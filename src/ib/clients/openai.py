import asyncio
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union

import openai
import tiktoken
from openai.types.chat.chat_completion import ChatCompletion

from src.ib.clients.base import Client, ClientConfig, ClientResponse, ClientSample
from src.ib.clients.usage import Usage, num_tokens_from_messages_openai
from src.ib.utils import get_logger

# Global event loop shared across all calls to Chat
loop: Optional[asyncio.AbstractEventLoop] = None


class OpenAIClient(Client):
    client_class: Type[openai.AsyncOpenAI] = openai.OpenAI

    class Config(ClientConfig):
        """Configuration options for the OpenAIClient."""

        _pass_as_config: bool = True

        model_name: str = "gpt-4o"
        api_key: Optional[str] = None
        base_url: Optional[str] = None

        # if we max out the context length, retry truncating the messages to fit in the
        # max length
        truncate_messages_and_retry: bool = True

    def __init__(self, config: Config):
        """
        Initialize the OpenAI client with the provided config.
        If config.api_key is set, it will override OPENAI_API_KEY in the environment.
        """
        self.config = config

        self.client = self.client_class(
            api_key=config.api_key if config.api_key else os.getenv("OPENAI_API_KEY"),
            base_url=(
                config.base_url if config.base_url else os.getenv("OPENAI_BASE_URL")
            ),
        )
        self.logger = get_logger("OpenAIClient")

        self.conversations = {}

        try:
            self.encoding = tiktoken.encoding_for_model(self.config.model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def complete(
        self,
        prompts: List[Union[str, List[int]]],
        temperature: float = 0.6,
        stop: List[str] = [],
        max_completion_tokens: int = 1,
        **kwargs,
    ) -> ClientSample:
        raise NotImplementedError("OpenAI does not support a completion API.")

    def chat(
        self,
        chats: List[List[Dict[str, Any]]],
        temperature: float = 0.6,
        stop: List[str] = [],
        max_completion_tokens: Optional[int] = None,
        conversation_id: Optional[str] = None,
        **kwargs,
    ) -> ClientResponse:
        assert len(chats) > 0
        # Flatten the top-level list of message lists (assuming only one chat set for demonstration)
        # If multiple chats are needed, you should call the API multiple times or adapt accordingly.
        if isinstance(chats[0], Dict):
            chats = [(chats, 1)]

        # find duplicate chats
        chat_to_count = defaultdict(list)
        for idx, messages in enumerate(chats):
            tup = tuple(frozenset(message.items()) for message in messages)
            chat_to_count[tup].append(idx)

        responses = [None] * len(chats)
        usage = Usage(prompt_tokens=0, completion_tokens=0)
        usages = []
        for messages, idxs in chat_to_count.items():
            messages = [dict(message) for message in messages]  # convert back

            def chat(m: List[Dict[str, Any]]) -> ChatCompletion:
                return self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=m,
                    max_tokens=max_completion_tokens,
                    temperature=temperature,
                    stop=stop if stop else None,
                    n=len(idxs),
                    logprobs=True,
                    **kwargs,
                )

            # SE (01/20): If a BadRequestError occurs due to messages being too long,
            # we handle it by retrying with progressively truncated messages.
            # This is only done if the config option 'truncate_messages_and_retry' is set to True.
            # We start by removing the first message and continue truncating until a
            # valid response is received or all messages are exhausted.
            error = None
            for num_truncated_messages in range(len(messages)):
                try:
                    used_messages = messages[num_truncated_messages:]
                    response: ChatCompletion = chat(used_messages)
                    break
                except openai.BadRequestError as e:
                    self.logger.warning(f"BadRequestError: {e}")
                    error = e
                    if e.body.get("code", "") not in (
                        "context_length_exceeded",
                        "too_many_messages",
                    ):
                        raise e

                    if not self.config.truncate_messages_and_retry:
                        raise ValueError(
                            f"OpenAI returned the following BadRequestError: {e}."
                            "Set truncate_messages_and_retry=True to retry with truncated messages."
                        )

                    self.logger.warning(
                        f"OpenAI returned the following BadRequestError: {e}. "
                        f"Truncating first {num_truncated_messages + 1} messages and retrying..."
                    )
                except Exception as e:
                    self.logger.error(f"Unexpected exception: {e}")
                    raise e
            else:
                raise ValueError(
                    f"OpenAI returned the following BadRequestError: {error}. "
                    "Even though you have set truncate_messages_and_retry=True, "
                    "the last message is still too long."
                )

            # Track conversation history for prompt token calculation
            curr_usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                # cached_prompt_tokens=response.usage.prompt_tokens_details.cached_tokens,
            )
            if conversation_id is not None:
                if conversation_id not in self.conversations:
                    self.conversations[conversation_id] = []

                max_matched_messages = []
                for prev_messages in self.conversations[conversation_id]:
                    matched_messages = []
                    for curr, prev in zip(used_messages, prev_messages):
                        if (curr["content"] != prev["content"]) or (
                            curr["role"] != prev["role"]
                        ):
                            break
                        matched_messages.append(curr)
                    if len(matched_messages) > len(max_matched_messages):
                        max_matched_messages = matched_messages

                curr_usage.seen_prompt_tokens = num_tokens_from_messages_openai(
                    max_matched_messages, encoding=self.encoding
                )

                # Add newly generated messages to conversation history
                self.conversations[conversation_id].extend(
                    [
                        used_messages
                        + [{"role": "assistant", "content": choice.message.content}]
                        for choice in response.choices
                    ]
                )
            usages.append(curr_usage)
            usage += curr_usage

            for idx, choice in zip(idxs, response.choices):
                # Extract tokens and logprobs from response
                tokens = []
                logprobs = []
                if choice.logprobs and choice.logprobs.content is not None:
                    for logprob in choice.logprobs.content:
                        tokens.append(logprob.token)
                        logprobs.append(logprob.logprob)

                if choice.finish_reason == "stop":
                    stop_reason = "stop_string"
                elif choice.finish_reason == "length":
                    stop_reason = "max_tokens"
                else:
                    stop_reason = "stop_string"  # fallback

                # Fallback when logprobs is not provided
                if not tokens:
                    text_content = choice.message.content
                    tokens = [text_content] if text_content else []

                responses[idx] = ClientSample(
                    text=choice.message.content,
                    tokens=tokens,
                    token_ids=None,
                    log_prob=logprobs,
                    stop_reason=stop_reason,
                    input_log_prob=None,
                )
        usage.cost = self._calculate_cost(usage)
        return ClientResponse(samples=responses, usage=usage, usages=usages)

    def _calculate_cost(self, usage: Union[Usage, None]) -> float:
        if usage:
            return openai_cost(self.config.model_name, usage.prompt_tokens, usage.completion_tokens)
        return 0.0

def openai_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    unit = 1_000_000  # 1 million tokens
    pricing = {
        "gpt-4.1-2025-04-14": {"input": 2.00 / unit, "output": 8.00 / unit},
        "gpt-4.1-mini-2025-04-14": {
            "input": 0.40 / unit,
            "output": 1.60 / unit,
        },
        "gpt-4.1-nano-2025-04-14": {
            "input": 0.10 / unit,
            "output": 0.40 / unit,
        },
        "gpt-4.5-preview-2025-02-27": {
            "input": 75.00 / unit,
            "output": 150.00 / unit,
        },
        "gpt-4o-2024-08-06": {"input": 2.50 / unit, "output": 10.00 / unit},
        "gpt-4o": {"input": 2.50 / unit, "output": 10.00 / unit},
        "gpt-4o-mini-2024-07-18": {"input": 0.15 / unit, "output": 0.60 / unit},
        "gpt-4o-mini": {"input": 0.15 / unit, "output": 0.60 / unit},
        "o1-2024-12-17": {"input": 15.00 / unit, "output": 60.00 / unit},
        "o1-pro-2025-03-19": {"input": 150.00 / unit, "output": 600.00 / unit},
        "o3-pro-2025-06-10": {"input": 20.00 / unit, "output": 80.00 / unit},
    }
    model_pricing = pricing.get(
        model_name, {"input": 0.0, "output": 0.0}
    )

    prompt_cost = model_pricing["input"] * prompt_tokens
    completion_cost = model_pricing["output"] * completion_tokens

    total_cost = prompt_cost + completion_cost
    return round(total_cost, 6)

class AsyncOpenAIClient(OpenAIClient):

    def chat(
        self,
        chats: List[List[Dict[str, Any]]],
        temperature: float = 0.6,
        stop: List[str] = [],
        max_completion_tokens: Optional[int] = None,
        **kwargs,
    ) -> ClientResponse:
        """
        Calls the OpenAI ChatCompletion endpoint, sending a single combined chat message array.
        If you want to handle multiple parallel chats, you'd need to adapt this accordingly.

        Add support async chats
        """
        assert len(chats) > 0
        # Flatten the top-level list of message lists (assuming only one chat set for demonstration)
        # If multiple chats are needed, you should call the API multiple times or adapt accordingly.
        if isinstance(chats[0], Dict):
            chats = [(chats, 1)]

        def _build_responses(chat_to_count, results: List[ChatCompletion]):
            # We transform the raw results into a final responses list
            # using your existing logic.
            usage = Usage(prompt_tokens=0, completion_tokens=0)
            responses = [None] * len(chats)
            for idxs, response in zip(chat_to_count.values(), results):
                usage += Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )
                for idx, choice in zip(idxs, response.choices):
                    tokens = []
                    logprobs = []
                    for logprob in getattr(choice.logprobs, "content", []):
                        tokens.append(logprob.token)
                        logprobs.append(logprob.logprob)
                    if choice.finish_reason == "stop":
                        stop_reason = "stop_string"
                    elif choice.finish_reason == "length":
                        stop_reason = "max_tokens"
                    else:
                        stop_reason = "stop_string"
                    if not tokens:
                        text_content = choice.message.content
                        tokens = [text_content] if text_content else []
                    responses[idx] = ClientSample(
                        text=choice.message.content,
                        tokens=tokens,
                        token_ids=None,
                        log_prob=logprobs,
                        input_log_prob=None,
                        stop_reason=stop_reason,
                    )
            return responses, usage

        async def _async_chat():
            chat_to_count = defaultdict(list)
            for idx, messages in enumerate(chats):
                tup = tuple(frozenset(message.items()) for message in messages)
                chat_to_count[tup].append(idx)
            tasks = []
            for messages, idxs in chat_to_count.items():
                messages = [dict(message) for message in messages]
                tasks.append(
                    asyncio.create_task(
                        self.client.chat.completions.create(
                            model=self.config.model_name,
                            messages=messages,
                            max_tokens=max_completion_tokens,
                            temperature=temperature,
                            stop=stop if stop else None,
                            n=len(idxs),
                            logprobs=True,
                            **kwargs,
                        )
                    )
                )
            results = await asyncio.gather(*tasks)
            return chat_to_count, results

        global loop
        if loop is None:
            loop = asyncio.get_event_loop()
        chat_to_count, results = loop.run_until_complete(_async_chat())
        final_responses, usage = _build_responses(chat_to_count, results)
        return ClientResponse(samples=final_responses, usage=usage)
