import subprocess
import time
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
from sglang.lang.interpreter import ProgramState
from transformers import AutoTokenizer

from src.ib.clients.base import Client, ClientConfig, ClientResponse, ClientSample
from src.ib.clients.mixins import ServerMixin
from src.ib.clients.usage import Usage
from src.ib.utils import get_logger


class SGLangClient(Client, ServerMixin):
    """
    SGLang client for local and remote endpoints.
    
    Note: Usage tracking does not account for prefix sharing and represents
    the sum of prompt and completion tokens across the batch.
    """

    class Config(ClientConfig):
        _pass_as_config: bool = True
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct"

        # If none, find a free port and launch the server
        # Otherwise assume a server is already running on the given port
        port: Optional[int] = None
        capture_output: bool = False

        mem_fraction_static: float = 0.8

        timeout: int = 100

        enable_prefix_sharing: bool = True

        # Remote API configuration
        api_base_url: Optional[str] = None
        api_key: Optional[str] = None

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("SGLangClient")

        if config.api_base_url is not None:
            # Connect to a remote endpoint
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.backend = RuntimeEndpoint(config.api_base_url)
        elif config.port is None:
            # Launch a local server
            self.port = self.find_free_port()
            launch_command = f"""python -m sglang.launch_server \
            --port {self.port} \
            --model-path {config.model_name} \
            --mem-fraction-static {config.mem_fraction_static} \
            """
            self.launch_server(
                launch_command, self.port, capture_output=config.capture_output
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.backend = RuntimeEndpoint(f"http://127.0.0.1:{self.port}")
        else:
            # Connect to an existing local server
            self.port = config.port
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    self.backend = RuntimeEndpoint(f"http://127.0.0.1:{self.port}")
                    set_default_backend(self.backend)
                    model_info = self.backend.model_info
                    self.logger.info(
                        f"Successfully connected to sglang backend at: {config.api_base_url}"
                    )
                    self.logger.info(f"Model Info: {model_info}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_info["tokenizer_path"]
                    )
                    self.model_name = model_info["model_path"]
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to connect to {config.api_base_url}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        self.logger.error(
                            f"Failed to connect to {config.api_base_url} after {max_retries} attempts"
                        )
                        raise ConnectionError(
                            f"Failed to connect to {config.api_base_url} after {max_retries} attempts"
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
        **kwargs,
    ) -> ClientResponse:
        stop_strings, stop_token_ids = self._split_stop_strings_into_tokens(stop)

        if self.config.enable_prefix_sharing:
            chats = sorted(chats, key=len)

        @function
        def parallel_chats(s: ProgramState, messages: List[Dict[str, Any]]):
            for message in messages:
                s += getattr(sgl, message["role"])(message["content"])
            s += assistant(
                gen(
                    "answer",
                    max_tokens=max_completion_tokens,
                    return_logprob=True,
                    temperature=temperature,
                    logprob_start_len=0,
                    stop=stop_strings,
                    stop_token_ids=stop_token_ids,
                )
            )

        states: List[ProgramState] = parallel_chats.run_batch(
            [{"messages": messages} for messages in chats], backend=self.backend
        )
        num_retries = 0
        max_retries = 5
        while any(state.error() is not None for state in states):
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
            logprob, token_ids, _ = zip(*answer["output_token_logprobs"])
            input_logprob, input_token_ids, _ = zip(*answer["input_token_logprobs"])
            tokens = [self.tokenizer.decode(token_id) for token_id in token_ids]
            input_tokens = [
                self.tokenizer.decode(token_id) for token_id in input_token_ids
            ]

            # Join the input tokens with the output tokens
            tokens = input_tokens + tokens
            logprob = list(input_logprob) + list(logprob)
            token_ids = list(input_token_ids) + list(token_ids)

            responses.append(
                ClientSample(
                    text=state.messages()[-1]["content"],
                    tokens=tokens,
                    log_prob=logprob,
                    token_ids=token_ids,
                    input_log_prob=input_logprob,
                    stop_reason=answer["finish_reason"]["type"],
                )
            )
            usage += Usage(
                prompt_tokens=answer["prompt_tokens"],
                completion_tokens=answer["completion_tokens"],
            )

        return ClientResponse(samples=responses, usage=usage)

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
