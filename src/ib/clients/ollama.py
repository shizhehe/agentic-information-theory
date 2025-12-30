import asyncio
import logging
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union, Tuple

from src.ib.clients.usage import Usage
from src.ib.clients.base import ClientConfig
from src.ib.clients.base import ClientSample, ClientResponse

class OllamaClient:
    class Config(ClientConfig):
        model_name: str = "llama3.2"
        temperature: float = 0.0
        max_tokens: int = 2048
        num_ctx: int = 4096
        use_async: bool = False

        def instantiate(self):
            return OllamaClient(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                num_ctx=self.num_ctx,
                use_async=self.use_async,
            )
        
    def __init__(
            self,
            model_name: str = "llama-3.2",
            temperature: float = 0.0,
            max_tokens: int = 2048,
            num_ctx: int = 4096,
            structured_output_schema: Optional[BaseModel] = None,
            use_async: bool = False,
    ):
        """Initialize Ollama Client."""
        self.model_name = model_name
        self.logger = logging.getLogger("OllamaClient")
        self.logger.setLevel(logging.INFO)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx

        if self.model_name == "granite3.2-vision":
            self.num_ctx = 131072
            self.max_tokens = 131072

        self.use_async = use_async

        # If we want structured schema output:
        self.format_structured_output = None
        if structured_output_schema:
            self.format_structured_output = structured_output_schema.model_json_schema()

        # For async calls
        from ollama import AsyncClient

        self.client = AsyncClient() if use_async else None

        # Ensure model is pulled
        self._ensure_model_available()

    @staticmethod
    def get_available_models():
        """
        Get a list of available Ollama models

        Returns:
            List[str]: List of model names
        """
        try:
            import ollama
            models = ollama.list()

            # Extract model names from the list
            model_names = [model.model for model in models['models']]
            return model_names
        except Exception as e:
            logging.error(f"Failed to get Ollama model list: {e}")
            return []

    def _ensure_model_available(self):
        import ollama

        try:
            ollama.chat(
                model=self.model_name, messages=[{"role": "system", "content": "test"}]
            )
        except ollama.ResponseError as e:
            if e.status_code == 404:
                self.logger.info(
                    f"Model {self.model_name} not found locally. Pulling..."
                )
                ollama.pull(self.model_name)
                self.logger.info(f"Successfully pulled model {self.model_name}")
            else:
                raise

    def _prepare_options(self):
        """Common chat options for both sync and async calls."""
        opts = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "num_ctx": self.num_ctx,
        }
        chat_kwargs = {"options": opts}
        if self.format_structured_output:
            chat_kwargs["format"] = self.format_structured_output
        return chat_kwargs

    #
    #  ASYNC
    #
    def achat(
            self,
            messages: Union[List[Dict[str, Any]], Dict[str, Any]],
            **kwargs,
    ) -> Tuple[List[str], List[Usage], List[str]]:
        """
        Wrapper for async chat. Runs `asyncio.run()` internally to simplify usage.
        """
        if not self.use_async:
            raise RuntimeError(
                "This client is not in async mode. Set `use_async=True`."
            )

        try:
            return asyncio.run(self._achat_internal(messages, **kwargs))
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                # Create a new event loop and set it as the current one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self._achat_internal(messages, **kwargs)
                    )
                finally:
                    loop.close()
            raise

    async def _achat_internal(
            self,
            messages: Union[List[Dict[str, Any]], Dict[str, Any]],
            **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle async chat with multiple messages in parallel.
        """
        # If the user provided a single dictionary, wrap it in a list.
        if isinstance(messages, dict):
            messages = [messages]

        # Now we have a list of dictionaries. We'll call them in parallel.
        chat_kwargs = self._prepare_options()

        async def process_one(msg):
            resp = await self.client.chat(
                model=self.model_name,
                messages=[msg],  # each call with exactly one message
                **chat_kwargs,
                **kwargs,
            )
            return resp

        # Run them all in parallel
        results = await asyncio.gather(*(process_one(m) for m in messages))

        # Gather them back
        texts = []
        usage_total = Usage()
        done_reasons = []
        for r in results:
            texts.append(r["message"]["content"])
            usage_total += Usage(
                prompt_tokens=r["prompt_eval_count"], completion_tokens=r["eval_count"]
            )
            done_reasons.append(r["done_reason"])

        return texts, usage_total, done_reasons

    def schat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle synchronous chat completions. If you pass a list of message dicts,
        we do one call for that entire conversation. If you pass a single dict,
        we wrap it in a list so there's no error.
        """
        import ollama

        if isinstance(messages, dict):
            messages = [messages]

        chat_kwargs = self._prepare_options()  # base kwargs
        merged_kwargs = {**chat_kwargs, **kwargs}

        # 1. Separate valid top-level chat args
        ALLOWED_CHAT_ARGS = {"format", "stream", "template", "system", "keep_alive"}
        top_level_kwargs = {k: v for k, v in merged_kwargs.items() if k in ALLOWED_CHAT_ARGS}

        # 2. Funnel extra args into options={}
        GENERATION_ARGS = {"temperature", "top_p", "num_ctx", "num_predict", "seed", "repeat_penalty", "stop", "frequency_penalty", "presence_penalty"}
        options = {
            k: v for k, v in merged_kwargs.items()
            if k in GENERATION_ARGS
        }

        """print("Calling ollama.chat with:")
        print("  model:", self.model_name)
        print("  messages:", messages)
        print("  options:", options)
        print("  top_level_kwargs:", top_level_kwargs)"""

        if isinstance(messages, list) and len(messages) == 1 and isinstance(messages[0], list):
            messages = messages[0]

        print("Sending to ollama.chat")

        # Merge into final call
        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            options=options if options else None,
            **top_level_kwargs,
        )

        print("Response received from ollama.chat")

        responses = [response["message"]["content"]]
        usage_total = Usage(
            prompt_tokens=response.get("prompt_eval_count", 0),
            completion_tokens=response.get("eval_count", 0),
        )
        done_reasons = [response.get("done_reason", "stop")]

        # Wrap response in ClientResponse
        return ClientResponse(
            samples=[
                ClientSample(
                    text=response["message"]["content"],
                    tokens=[],
                    token_ids=None,
                    log_prob=[],
                    input_log_prob=None,    
                    stop_reason=response.get("done_reason", "stop"),
                )
            ],
            usage=Usage(
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=response.get("eval_count", 0),
            )
        )
        # return responses, usage_total, done_reasons

    def chat(
            self,
            messages: Union[List[Dict[str, Any]], Dict[str, Any]],
            **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle synchronous chat completions. If you pass a list of message dicts,
        we do one call for that entire conversation. If you pass a single dict,
        we wrap it in a list so there's no error.
        """
        if self.use_async:
            return self.achat(messages, **kwargs)
        else:
            return self.schat(messages, **kwargs)
