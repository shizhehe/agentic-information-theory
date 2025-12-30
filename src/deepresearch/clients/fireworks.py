"""
Fireworks Client for ResSwarm.

Self-contained implementation adapted from minions-factory for connecting to
Fireworks AI models.
"""

import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from fireworks.client import Fireworks
import tiktoken

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
from src.ib.clients.usage import Usage
from src.deepresearch.clients.base import ResSwarmClient


class FireworksClient(ResSwarmClient):
    """
    Fireworks client for accessing Fireworks AI models.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        truncate_messages_and_retry: bool = True,
        **kwargs
    ):
        """
        Initialize Fireworks client.
        
        Args:
            model_name: Fireworks model name (e.g., "accounts/fireworks/models/llama-v3p3-70b-instruct")
            api_key: Fireworks API key (or use FIREWORKS_API_KEY env var)
            base_url: Optional custom base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            truncate_messages_and_retry: Whether to retry with truncated messages on context length errors
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
        
        self.truncate_messages_and_retry = truncate_messages_and_retry
        
        # Initialize Fireworks client
        self.client = Fireworks(
            api_key=api_key if api_key else os.getenv("FIREWORKS_API_KEY"),
            base_url=base_url
        )
        
        # Set up tokenizer for usage calculation
        try:
            self.encoding = tiktoken.encoding_for_model("llama-v3")
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def complete(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[str], Usage]:
        """
        Complete prompts using Fireworks.
        
        Note: Fireworks client currently only supports chat API, so this converts
        prompts to chat format.
        """
        # Convert prompts to chat format
        chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
        
        temp = temperature if temperature is not None else getattr(self, 'temperature', 0.7)
        max_completion_tokens = max_tokens if max_tokens is not None else getattr(self, 'max_tokens', None)
        
        responses = []
        total_usage = Usage(prompt_tokens=0, completion_tokens=0)
        
        for chat in chats:
            response_texts, usage = self.chat(
                messages=chat,
                temperature=temp,
                max_tokens=max_completion_tokens,
                **kwargs
            )
            responses.append(response_texts[0])  # Extract single response from list
            total_usage += usage
        
        return responses, total_usage

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: List[str] = [],
        **kwargs
    ) -> Tuple[List[str], Usage]:
        """
        Chat completion using Fireworks.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stop: Stop sequences
            
        Returns:
            Tuple of (response_texts_list, usage) - note: returns list for minions compatibility
        """
        temp = temperature if temperature is not None else getattr(self, 'temperature', 0.7)
        max_completion_tokens = max_tokens if max_tokens is not None else getattr(self, 'max_tokens', None)
        
        def make_request(m: List[Dict[str, Any]]):
            max_retries = 5
            retry_count = 0
            while True:
                try:
                    return self.client.chat.completions.create(
                        model=self.model_name,
                        messages=m,
                        max_tokens=max_completion_tokens,
                        temperature=temp,
                        stop=stop if stop else None,
                        logprobs=True,
                        **kwargs
                    )
                except Exception as e:
                    if retry_count < max_retries:
                        retry_count += 1
                        print(f"[Fireworks] Error encountered. Retrying ({retry_count}/{max_retries})...")
                        time.sleep(5)
                        continue
                    raise e
        
        # Handle potential context length error by progressively truncating messages
        error = None
        for num_truncated_messages in range(len(messages)):
            try:
                used_messages = messages[num_truncated_messages:]
                response = make_request(used_messages)
                break
            except Exception as e:
                error = e
                if not hasattr(e, 'code') or getattr(e, 'code', '') not in ("context_length_exceeded", "too_many_messages"):
                    raise e
                
                if not self.truncate_messages_and_retry:
                    raise ValueError(
                        f"Fireworks returned an error: {e}. "
                        "Set truncate_messages_and_retry=True to retry with truncated messages."
                    )
                
                print(
                    f"[Fireworks] Context length error: {e}. "
                    f"Truncating first {num_truncated_messages + 1} messages and retrying..."
                )
        else:
            raise ValueError(
                f"Fireworks returned an error: {error}. "
                "Even with truncate_messages_and_retry=True, the last message is still too long."
            )
        
        # Extract response
        choice = response.choices[0]
        response_text = choice.message.content
        
        # Calculate usage
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        
        # Return as list for minions compatibility (DeepResearch minion expects response[0])
        return [response_text], usage

    def batch_chat(
        self,
        chats: List[List[Dict[str, Any]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: List[str] = [],
        **kwargs
    ) -> Tuple[List[str], Usage]:
        """
        Batch chat completion for multiple conversations.
        
        Args:
            chats: List of conversation message lists
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stop: Stop sequences
            
        Returns:
            Tuple of (response_texts, total_usage)
        """
        temp = temperature if temperature is not None else getattr(self, 'temperature', 0.7)
        max_completion_tokens = max_tokens if max_tokens is not None else getattr(self, 'max_tokens', None)
        
        # Find duplicate chats to optimize requests
        chat_to_count = defaultdict(list)
        for idx, messages in enumerate(chats):
            tup = tuple(frozenset(message.items()) for message in messages)
            chat_to_count[tup].append(idx)
        
        responses = [None] * len(chats)
        total_usage = Usage(prompt_tokens=0, completion_tokens=0)
        
        for messages_tuple, idxs in chat_to_count.items():
            messages = [dict(message) for message in messages_tuple]  # convert back
            
            # Use single chat for this batch
            response_texts, usage = self.chat(
                messages=messages,
                temperature=temp,
                max_tokens=max_completion_tokens,
                stop=stop,
                **kwargs
            )
            
            # Assign response to all duplicate indices
            for idx in idxs:
                responses[idx] = response_texts[0]  # Extract single response from list
            
            total_usage += usage
        
        return responses, total_usage