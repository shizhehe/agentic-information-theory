"""
SGLang Modal Client for ResSwarm.

Self-contained implementation adapted from minions-factory for connecting to
SGLang models served via Modal endpoints.
"""

from typing import Any, Dict, List, Optional, Tuple
import time
from transformers import AutoTokenizer

import sglang as sgl
from sglang import (
    function,
    system,
    user,
    assistant,
    gen,
    set_default_backend,
    RuntimeEndpoint,
)
from sglang.lang.interpreter import ProgramState

from src.deepresearch.clients.base import ResSwarmClient
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
from src.ib.clients.usage import Usage


class SGLangModalClient(ResSwarmClient):
    """
    SGLang client for connecting to Modal-served endpoints.

    This client connects to SGLang models served via Modal endpoints
    (e.g., YOUR_MODAL_ENDPOINT_HERE)
    """

    def __init__(
        self,
        model_name: str,
        api_base_url: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 100,
        enable_prefix_sharing: bool = True,
        **kwargs
    ):
        """
        Initialize SGLang Modal client.
        
        Args:
            model_name: Name of the model
            api_base_url: Modal endpoint URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            enable_prefix_sharing: Whether to enable prefix sharing optimization
        """
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens, **kwargs)
        
        self.api_base_url = api_base_url
        self.timeout = timeout
        self.enable_prefix_sharing = enable_prefix_sharing
        
        # Connect to the remote endpoint
        self.backend = RuntimeEndpoint(api_base_url)
        set_default_backend(self.backend)
        
        print(f"Successfully connected to sglang backend at: {api_base_url}")
        
        # Get model info and set up tokenizer
        model_info = self.backend.model_info
        print(f"Model Info: {model_info}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer_path"])
        self.actual_model_name = model_info["model_path"]

    def complete(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[str], Usage]:
        """
        Complete prompts using SGLang.
        
        Args:
            prompts: List of prompt strings
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Tuple of (completions, usage)
        """
        temp = temperature if temperature is not None else getattr(self, 'temperature', 0.7)
        max_completion_tokens = max_tokens if max_tokens is not None else getattr(self, 'max_tokens', None)

        @function
        def parallel_completions(s: ProgramState, prompt: str):
            s += prompt
            s += gen(
                "answer",
                max_tokens=max_completion_tokens,
                return_logprob=True,
                temperature=temp,
            )

        states: List[ProgramState] = parallel_completions.run_batch(
            [{"prompt": prompt} for prompt in prompts], backend=self.backend
        )
        
        completions = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for state in states:
            if state.error() is not None:
                print(f"[SGLangModal] Error: {state.error()}")
                completions.append("")
                continue
                
            answer = state.stream_executor.meta_info["answer"]
            completions.append(state.messages()[-1]["content"])
            
            # Extract token counts with fallback
            prompt_tokens = answer.get("prompt_tokens", 0)
            completion_tokens = answer.get("completion_tokens", 0)
            
            # If token counts are zero, estimate from text
            if prompt_tokens == 0 and completion_tokens == 0:
                # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                prompt_text = prompts[len(completions) - 1]
                response_text = state.messages()[-1]["content"]
                prompt_tokens = max(1, len(prompt_text) // 4)
                completion_tokens = max(1, len(response_text) // 4)
            
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
        
        usage = Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens
        )
        
        return completions, usage

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: List[str] = [],
        **kwargs
    ) -> Tuple[List[str], Usage]:
        """
        Chat completion using SGLang.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stop: Stop sequences
            
        Returns:
            Tuple of (response_text, usage)
        """
        temp = temperature if temperature is not None else getattr(self, 'temperature', 0.7)
        max_completion_tokens = max_tokens if max_tokens is not None else getattr(self, 'max_tokens', None)
        
        stop_strings, stop_token_ids = self._split_stop_strings_into_tokens(stop)

        @function
        def chat_completion(s: ProgramState, messages: List[Dict[str, Any]]):
            for message in messages:
                s += getattr(sgl, message["role"])(message["content"])
            s += assistant(
                gen(
                    "answer",
                    max_tokens=max_completion_tokens,
                    return_logprob=True,
                    temperature=temp,
                    logprob_start_len=0,
                    stop=stop_strings,
                    stop_token_ids=stop_token_ids,
                )
            )

        states: List[ProgramState] = chat_completion.run_batch(
            [{"messages": messages}], backend=self.backend
        )
        
        # Retry logic for errors
        num_retries = 0
        max_retries = 5
        while any(state.error() is not None for state in states):
            num_retries += 1
            print(f"[SGLangModal] Retrying {num_retries} times")
            states = chat_completion.run_batch(
                [{"messages": messages}], backend=self.backend
            )
            if num_retries > max_retries:
                raise Exception(f"[SGLangModal] Failed to complete after {num_retries} retries")
            if num_retries > 1:
                time.sleep(1)
        
        state = states[0]
        if state.error() is not None:
            raise Exception(f"[SGLangModal] Error: {state.error()}")
        
        answer = state.stream_executor.meta_info["answer"]
        response_text = state.messages()[-1]["content"]
        
        # Extract token counts with fallback to avoid zeros
        prompt_tokens = answer.get("prompt_tokens", 0)
        completion_tokens = answer.get("completion_tokens", 0)
        
        # If token counts are zero, estimate from text
        if prompt_tokens == 0 and completion_tokens == 0:
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            prompt_text = " ".join([msg["content"] for msg in messages])
            prompt_tokens = max(1, len(prompt_text) // 4)
            completion_tokens = max(1, len(response_text) // 4)
            print(f"[SGLangModal] Estimated tokens: prompt={prompt_tokens}, completion={completion_tokens}")
        
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        return [response_text], usage

    def _split_stop_strings_into_tokens(
        self, stop: Optional[List[str]]
    ) -> Tuple[List[str], List[int]]:
        """Split stop strings into strings and token IDs."""
        stop_strings, stop_tokens = [], []
        if stop is None:
            return stop_strings, stop_tokens
        for stop_string in stop:
            if stop_string in self.tokenizer.vocab:
                stop_tokens.append(self.tokenizer.convert_tokens_to_ids(stop_string))
            else:
                stop_strings.append(stop_string)
        return stop_strings, stop_tokens