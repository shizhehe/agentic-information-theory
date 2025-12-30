import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import sglang as sgl
from sglang import (
    RuntimeEndpoint,
    assistant,
    function,
    gen,
    set_default_backend,
    system,
    user,
)
from sglang.lang.chat_template import get_chat_template
from sglang.lang.interpreter import ProgramState
from transformers import AutoTokenizer

from src.ib.clients.base import Client, ClientConfig, ClientResponse, ClientSample
from src.ib.clients.mixins import ServerMixin
from src.ib.clients.usage import Usage
from src.ib.utils import get_logger


class SGLangClient(Client, ServerMixin):
    """
    SGLang client for remote API endpoints.
    
    Note: Usage tracking does not account for prefix sharing and represents
    the sum of prompt and completion tokens across the batch.
    Currently supports only modal endpoints.
    """

    class Config(ClientConfig):
        _pass_as_config: bool = True
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct"

        # if none, find a free port and launch the server
        # otherwise assume a server is already running on the given port
        port: Optional[int] = None
        capture_output: bool = False

        mem_fraction_static: float = 0.8

        timeout: int = 100

        enable_prefix_sharing: bool = True

        disable_thinking: Optional[bool] = True
        remove_thinking: Optional[bool] = False

        # Remote API configuration
        api_base_url: Optional[str] = None

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("SGLangClient")

        if config.api_base_url is not None and config.port is None:
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    self.logger.info(
                        f"Attempting to connect to sglang backend (attempt {attempt + 1}/{max_retries}): {config.api_base_url}"
                    )
                    self.backend = RuntimeEndpoint(config.api_base_url)
                    set_default_backend(self.backend)
                    model_info = self.backend.model_info
                    self.logger.info(
                        f"Successfully connected to sglang backend at: {config.api_base_url}"
                    )
                    self.logger.info(f"Model Info: {model_info}")
                    self.model_name = model_info["model_path"]

                    self.backend.chat_template = get_chat_template("qwen")

                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_info["tokenizer_path"],
                        cache_dir="/scr-ssd/shizhehe/models"
                    )

                    if self.config.disable_thinking and "Qwen3" in self.model_name and "30B" not in self.model_name:
                        self.logger.info(f"Disabling thinking for {self.model_name}")
                        # by default, remove the redundant thinking tokens
                        self.config.remove_thinking = True

                    break
                except Exception as e:
                    traceback.print_exc()
                    self.logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        self.logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Failed to connect after {max_retries} attempts")
                        raise ConnectionError(
                            f"Could not connect to SGLang backend at {config.api_base_url} after {max_retries} attempts."
                        )

    def complete(
        self,
        prompts: str,
        temperature: float = 0.6,
        max_completion_tokens: Optional[int] = None,
        **kwargs,
    ) -> ClientResponse:
        @function
        def parallel_completions(s: ProgramState, prompt: List[str]):
            s += prompt
            s += gen(
                "answer",
                max_tokens=max_completion_tokens,
                return_logprob=True,
                temperature=temperature,
            )

        states: List[ProgramState] = parallel_completions.run_batch(
            [{"prompt": prompt} for prompt in prompts], backend=self.backend
        )
        return self._parse_states(states)

    def chat(
        self,
        chats: List[List[Dict[str, Any]]],
        temperature: float = 0.6,
        stop: List[str] = [],
        max_completion_tokens: Optional[int] = None,
        logprob_start_len: Optional[int] = None,
        return_logprob: bool = True,
        **kwargs,
    ) -> ClientResponse:
        stop_strings, stop_token_ids = self._split_stop_strings_into_tokens(stop)

        if self.config.enable_prefix_sharing:
            chats = sorted(chats, key=len)

        @function
        def parallel_chats(s: ProgramState, messages: List[Dict[str, Any]]):
            if self.config.disable_thinking and "Qwen3" in self.model_name:
                s += system("/no_think")  # for qwen3 models
            for message in messages:
                s += getattr(sgl, message["role"])(message["content"])
            if not self.config.disable_thinking and "Qwen3" in self.model_name:
                s += assistant(
                    gen(
                        "answer",
                        max_tokens=max_completion_tokens,
                        return_logprob=return_logprob,
                        temperature=temperature,
                        logprob_start_len=0,
                        stop=stop_strings,
                        stop_token_ids=stop_token_ids,
                        return_text_in_logprobs=True,
                        **kwargs,
                    )
                    + "</think>"
                    + gen(
                        "answer",
                        max_tokens=max_completion_tokens,
                        return_logprob=return_logprob,
                        temperature=temperature,
                        logprob_start_len=0,
                        stop=stop_strings,
                        stop_token_ids=stop_token_ids,
                        return_text_in_logprobs=True,
                        **kwargs,
                    )
                )
            else:
                s += assistant(
                    gen(
                        "answer",
                        max_tokens=max_completion_tokens,
                        return_logprob=return_logprob,
                        temperature=temperature,
                        logprob_start_len=0,
                        stop=stop_strings,
                        stop_token_ids=stop_token_ids,
                        return_text_in_logprobs=True,
                        **kwargs,
                    )
                )

        states: List[ProgramState] = parallel_chats.run_batch(
            [{"messages": messages} for messages in chats], backend=self.backend
        )
        num_retries = 0
        max_retries = 5
        while any(state.error() is not None for state in states):
            for state in states:
                if state.error() is not None:
                    self.logger.error(f"SGLang Error: {state.error()}")
            num_retries += 1
            self.logger.warning(f"Retrying SGLang request {num_retries} times")
            states = parallel_chats.run_batch(
                [{"messages": messages} for messages in chats], backend=self.backend
            )
            if num_retries > max_retries:
                raise Exception(
                    f"SGLang failed to complete after {num_retries} retries"
                )
            if num_retries > 1:
                time.sleep(1)
        return self._parse_states(states)

    def _parse_states(self, states: List[ProgramState]) -> List[ClientSample]:

        responses = []
        usages = []
        usage = Usage(prompt_tokens=0, completion_tokens=0)

        for state in states:
            if state.error() is not None:
                self.logger.error(f"SGLang Error: {state.error()}")
                responses.append(
                    ClientSample(
                        text=state.messages()[-1]["content"],
                        tokens=[],
                        log_prob=[],
                        token_ids=[],
                        input_log_prob=[],
                        stop_reason="error",
                    )
                )
                continue

            answer = state.stream_executor.meta_info["answer"]
            logprob, token_ids, output_tokens = zip(*answer["output_token_logprobs"])
            input_logprob, input_token_ids, input_tokens = zip(*answer["input_token_logprobs"])

            # Join the input tokens with the output tokens
            tokens = list(input_tokens) + list(output_tokens)
            logprob = list(input_logprob) + list(logprob)
            token_ids = list(input_token_ids) + list(token_ids)

            # Remove tokens between <think> and </think>
            content = state.messages()[-1]["content"]
            if self.config.remove_thinking and len(output_tokens) > 1:
                try:
                    start_think_index = tokens.index("<think>")
                    end_think_index = len(tokens) - 1 - tokens[::-1].index("</think>")

                    completion_tokens_manual = len(tokens[end_think_index + 1 :])
                    tokens = tokens[:start_think_index] + tokens[end_think_index + 1 :]
                    logprob = logprob[:start_think_index] + logprob[end_think_index + 1 :]
                    token_ids = (
                        token_ids[:start_think_index] + token_ids[end_think_index + 1 :]
                    )
                    end_think_index_content = content.rindex("</think>")
                    content = content[end_think_index_content + len("</think>") :]
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Failed to remove thinking tokens: {e}")
                    # Keep original content if removal fails

            responses.append(
                ClientSample(
                    text=content,
                    tokens=tokens,
                    log_prob=logprob,
                    token_ids=token_ids,
                    input_log_prob=input_logprob,
                    stop_reason=answer["finish_reason"]["type"],
                )
            )
            usages.append(
                Usage(
                    prompt_tokens=answer["prompt_tokens"],
                    completion_tokens=answer["completion_tokens"],
                )
            )
            usage += usages[-1]
        return ClientResponse(samples=responses, usages=usages, usage=usage)

    def _split_stop_strings_into_tokens(
        self, stop: Optional[List[str]]
    ) -> Tuple[List[str], List[int]]:
        stop_strings, stop_tokens = [], []
        if stop is None:
            return stop_strings, stop_tokens
        for stop_string in stop:
            if stop_string in self.tokenizer.vocab:
                stop_tokens.append(self.tokenizer.convert_tokens_to_ids(stop_string))
            else:
                stop_strings.append(stop_string)
        return stop_strings, stop_tokens
