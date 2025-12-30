"""
Together Client for ResSwarm.

ResSwarm-compatible implementation for connecting to Together AI models.
Returns only 2 values from chat() to match ResSwarm's expectations.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from together import Together

from src.deepreseach.clients.base import ResSwarmClient
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
from src.ib.clients.usage import Usage


class TogetherClient(ResSwarmClient):
    """
    Together client for accessing Together AI models.
    
    This client is designed to be compatible with ResSwarm's expectations,
    returning only (response, usage) from chat methods.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Together client.
        
        Args:
            model_name: Together model name (e.g., "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")
            api_key: Together API key (or use TOGETHER_API_KEY env var)
            base_url: Optional custom base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
        
        # Initialize Together client
        # Together SDK doesn't accept base_url parameter
        if api_key:
            self.client = Together(api_key=api_key)
        else:
            self.client = Together()  # Will use TOGETHER_API_KEY from environment

    def complete(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[str], Usage]:
        """
        Complete prompts using Together.
        
        Note: Together primarily supports chat API, so this converts
        prompts to chat format.
        
        Args:
            prompts: List of prompt strings
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Tuple of (completions, usage)
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
        Chat completion using Together.
        
        IMPORTANT: Returns only (response_list, usage) tuple to match ResSwarm expectations.
        The minions version returns 3 values, but ResSwarm expects only 2.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stop: Stop sequences
            
        Returns:
            Tuple of (response_texts_list, usage)
        """
        temp = temperature if temperature is not None else getattr(self, 'temperature', 0.7)
        max_completion_tokens = max_tokens if max_tokens is not None else getattr(self, 'max_tokens', None)
        
        # Make API request with retry logic
        max_retries = 5
        retry_count = 0
        
        while True:
            try:
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temp,
                }
                
                if max_completion_tokens:
                    params["max_tokens"] = max_completion_tokens
                
                if stop:
                    params["stop"] = stop
                
                # Add any additional kwargs
                params.update(kwargs)
                
                response = self.client.chat.completions.create(**params)
                break
                
            except Exception as e:
                if retry_count < max_retries:
                    retry_count += 1
                    print(f"[Together] Error encountered: {e}. Retrying ({retry_count}/{max_retries})...")
                    time.sleep(5)
                    continue
                raise e
        
        # Extract response
        choice = response.choices[0]
        response_text = choice.message.content
        
        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        
        # Return as list for ResSwarm compatibility
        # IMPORTANT: Only return 2 values, not 3 like the minions version
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
        
        responses = []
        total_usage = Usage(prompt_tokens=0, completion_tokens=0)
        
        # Process each chat sequentially
        # (Together doesn't have native batch support like some other providers)
        for messages in chats:
            response_texts, usage = self.chat(
                messages=messages,
                temperature=temp,
                max_tokens=max_completion_tokens,
                stop=stop,
                **kwargs
            )
            
            responses.append(response_texts[0])  # Extract single response from list
            total_usage += usage
        
        return responses, total_usage