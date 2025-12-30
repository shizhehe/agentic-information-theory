import os
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union

import tiktoken
from fireworks.client import Fireworks

from src.ib.clients.base import Client, ClientConfig, ClientResponse, ClientSample
from src.ib.clients.usage import Usage, num_tokens_from_messages_openai
from src.ib.utils import get_logger


class FireworksClient(Client):
    """Fireworks AI client for chat completions."""
    
    client_class: Type[Fireworks] = Fireworks

    class Config(ClientConfig):
        """Configuration options for the FireworksClient."""

        _pass_as_config: bool = True

        model_name: str = "accounts/fireworks/models/llama-v3p3-70b-instruct"
        api_key: Optional[str] = None
        base_url: Optional[str] = None

        # If we max out the context length, retry truncating the messages to fit
        truncate_messages_and_retry: bool = True

    def __init__(self, config: Config):
        """
        Initialize the Fireworks client with the provided config.
        If config.api_key is set, it will override FIREWORKS_API_KEY in the environment.
        """
        self.config = config

        self.client = self.client_class(
            api_key=(
                config.api_key if config.api_key else os.getenv("FIREWORKS_API_KEY")
            ),
        )
        self.logger = get_logger("FireworksClient")

        self.conversations = {}

        # Try to get an appropriate encoding, fallback to cl100k_base if model-specific not found
        try:
            self.encoding = tiktoken.encoding_for_model("llama-v3")
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
        raise NotImplementedError("Fireworks client currently only supports chat API.")

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
        # Flatten the top-level list of message lists
        if isinstance(chats[0], Dict):
            chats = [(chats, 1)]

        # Find duplicate chats
        chat_to_count = defaultdict(list)
        for idx, messages in enumerate(chats):
            tup = tuple(frozenset(message.items()) for message in messages)
            chat_to_count[tup].append(idx)

        responses = [None] * len(chats)
        usage = Usage(prompt_tokens=0, completion_tokens=0)
        usages = []

        for messages, idxs in chat_to_count.items():
            messages = [dict(message) for message in messages]  # convert back

            def chat_completion(m: List[Dict[str, Any]]):
                max_retries = 5
                retry_count = 0
                while True:
                    try:
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
                    except Exception as e:
                        if retry_count < max_retries:
                            retry_count += 1
                            traceback.print_exc()
                            self.logger.warning(
                                f"Error with model {self.config.model_name}. Retrying ({retry_count}/{max_retries})..."
                            )
                            time.sleep(5)
                            continue
                        raise e

            # Handle potential context length error by progressively truncating messages
            error = None
            for num_truncated_messages in range(len(messages)):
                try:
                    used_messages = messages[num_truncated_messages:]
                    response = chat_completion(used_messages)
                    break
                except Exception as e:
                    error = e
                    if not hasattr(e, "code") or getattr(e, "code", "") not in (
                        "context_length_exceeded",
                        "too_many_messages",
                    ):
                        raise e

                    if not self.config.truncate_messages_and_retry:
                        raise ValueError(
                            f"Fireworks returned an error: {e}. "
                            "Set truncate_messages_and_retry=True to retry with truncated messages."
                        )

                    self.logger.warning(
                        f"Fireworks error: {e}. "
                        f"Truncating first {num_truncated_messages + 1} messages and retrying..."
                    )
            else:
                raise ValueError(
                    f"Fireworks error: {error}. "
                    "Even with truncate_messages_and_retry=True, "
                    "the last message is still too long."
                )

            # Calculate token usage
            curr_usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

            # Track conversation history if conversation_id is provided
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

            usage += curr_usage

            for idx, choice in zip(idxs, response.choices):
                # Extract tokens and logprobs from the response
                tokens = []
                logprobs = []
                for logprob in choice.logprobs.content:
                    tokens.append(logprob.token)
                    logprobs.append(logprob.logprob)

                if choice.finish_reason == "stop":
                    stop_reason = "stop_string"
                elif choice.finish_reason == "length":
                    stop_reason = "max_tokens"
                else:
                    stop_reason = "stop_string"  # fallback

                # Fallback if no tokens provided
                if not tokens:
                    text_content = choice.message.content
                    tokens = [text_content] if text_content else []

                responses[idx] = ClientSample(
                    text=choice.message.content,
                    tokens=tokens,
                    token_ids=None,  # Fireworks doesn't provide token IDs
                    log_prob=logprobs,
                    stop_reason=stop_reason,
                    input_log_prob=logprobs,
                )

                usages.append(Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens / len(idxs),
                ))

        return ClientResponse(samples=responses, usage=usage, usages=usages)
