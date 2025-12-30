import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from src.ib.clients import ClientConfig, ClientResponse, Usage
from src.ib.clients.base import ClientSample
from src.ib.clients.sglang_modal import SGLangClient
from src.ib.pipeline.base import BaseProtocol, ProtocolResponse
from src.ib.tasks.base import Document
from src.ib.utils import get_logger


class PredictionResponse(BaseModel):
    explanation: str = Field(description="A brief explanation of the answer.")
    answer: str = Field(description="The final answer to the question.")


COMPRESSION_PROMPT = """
Summarize the following text to include ONLY information needed to answer the question.
Extract the key points relevant to the question.
DO NOT ANSWER THE QUESTION DIRECTLY.

Question: 
{query}

Text: 
{text}

Your summary (make sure to include all important details / background information related to the *question*. **DO NOT ANSWER THE QUESTION**)
"""

PREDICTION_PROMPT = """
Please answer the following question based on the provided summary.
Question:
{query}

Summary:
{summary}

Please respond in the following JSON format:
<briefly think about the information you have and the question you need to answer>

{{
    "explanation": "<brief explanation of the answer. explain how you arrived at the answer. 1-2 sentences>",
    "answer": "<your final answer>"
}}


Your answer (YOU MUST ONLY RESPOND WITH THE JSON OBJECT):
"""


class CompressionPredictionProtocol(BaseProtocol):
    class Config(BaseProtocol.Config):
        predictor_client: ClientConfig
        compressor_client: ClientConfig
        perplexity_client: Optional[ClientConfig] = None

        worker_max_new_tokens: int = 1024
        max_sentences: Optional[int] = None
        compression_prompt: str = COMPRESSION_PROMPT
        prediction_prompt: str = PREDICTION_PROMPT

        max_rounds: int = 3

        worker_temperature: float = 0.0
        worker_top_p: float = 1.0
        worker_top_k: int = -1

        supervisor_temperature: float = 0.6
        num_samples: int = 1
        num_samples_prediction: int = 1

        compression_path: Optional[str] = None
        return_num_tokens: bool = True
        
        # Dataset-specific configuration support
        dataset_type: Optional[str] = None  # e.g., "wildchat", "fineweb"

        def required_resources(self) -> Dict[str, Any]:
            return {
                "num_cpus": 1,
                "num_gpus": 1,
            }

    def __init__(self, config: Config):
        self.logger = get_logger(__name__)

        self.config = (
            config  # task specific config (specifies which clients are needed)
        )

        self.logger.info(f"Instantiating supervisor client...")
        self.predictor_client = config.predictor_client.instantiate()
        self.logger.info(f"Instantiating worker client...")
        self.compressor_client = config.compressor_client.instantiate()

        if hasattr(self.compressor_client.config, "api_base_url"):
            if self.compressor_client.config.api_base_url is not None:
                self.logger.info("Adapting compression prompt for Gemma models")
                self.config.compression_prompt = (
                    f"<start_of_turn>user\n{self.config.compression_prompt}\n"
                    f"<end_of_turn>\n<start_of_turn>model\n"
                )


        if config.perplexity_client:
            self.logger.info("Instantiating perplexity client...")
            self.perplexity_client = config.perplexity_client.instantiate()
        else:
            self.perplexity_client = None

        # Initialize tokenizer if needed for token counting
        if self.config.return_num_tokens:
            if hasattr(self.compressor_client, "tokenizer"):
                self.logger.info("Using worker client tokenizer")
                self.tokenizer = self.compressor_client.tokenizer
            elif hasattr(self.predictor_client, "tokenizer"):
                self.logger.info("Using supervisor client tokenizer")
                self.tokenizer = self.predictor_client.tokenizer
            else:
                self.logger.info("Defaulting to Llama tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Llama-3.1-8B-Instruct"
                )

        self.existing_compression_df = None
        if config.compression_path:
            self.existing_compression_df = pd.read_feather(
                os.path.join(config.compression_path, "results.feather")
            )
            self.logger.info(f"Using existing compression results from {config.compression_path}")

    def compress(
        self, id: str, query: str, context: str, num_samples: int = 1, **kwargs
    ):
        """Generate compressed summaries of the context for the given query."""
        if self.existing_compression_df is not None:
            existing_compressions = self._fetch_existing_compressions(id)
            if existing_compressions is not None:
                self.logger.info(
                    f"Found {len(existing_compressions.samples)} existing compressions for id {id}"
                )
                return existing_compressions

        # Remove question choices from query when creating summaries
        clean_query = query.split("Choose the correct answer from the following options:\n")[0]
        compression_prompt = self.config.compression_prompt.format(
            query=clean_query, text=context
        )
        
        if self.config.max_sentences is not None:
            self.logger.info(f"Instructing compressor to use at most {self.config.max_sentences} sentences")
            compression_prompt += (
                f"\n\nSummarize the text using EXACTLY {self.config.max_sentences} sentences. "
                f"Your summary:"
            )

        # Create identical messages for each sample
        compression_messages = [
            [{"role": "user", "content": compression_prompt}]
        ] * num_samples

        # Request multiple samples in a single call
        response = self.compressor_client.chat(
            compression_messages,
            temperature=self.config.worker_temperature,
            top_p=self.config.worker_top_p,
            top_k=self.config.worker_top_k,
            max_completion_tokens=self.config.worker_max_new_tokens,
        )
        return response

    def decompress(
        self, query: str, summary: str, num_samples_prediction: int = 1, **kwargs
    ):
        """Generate predictions based on the compressed summary."""
        if num_samples_prediction == 0:
            return ClientResponse(samples=[], usage=Usage())

        prediction_prompt = self.config.prediction_prompt.format(
            query=query, summary=summary
        )
        prediction_messages = [
            [{"role": "user", "content": prediction_prompt}]
        ] * num_samples_prediction

        if isinstance(self.predictor_client, SGLangClient):
            response = self.predictor_client.chat(
                prediction_messages,
                temperature=self.config.supervisor_temperature,
            )
        else:
            response_format = {"type": "json_object"}
            # Add schema for non-GPT models
            if not self.predictor_client.config.model_name.startswith("gpt"):
                response_format["schema"] = PredictionResponse.model_json_schema()
                
            response = self.predictor_client.chat(
                prediction_messages,
                temperature=self.config.supervisor_temperature,
                response_format=response_format,
            )
        return response

    def __call__(
        self,
        id: str,
        query: str,
        context: List[Document],
        num_samples: int = 1,
        **kwargs,
    ) -> ProtocolResponse:
        """Main protocol execution: compress context, then generate predictions."""
        # Use the config's num_samples or the provided one
        num_samples = self.config.num_samples if num_samples == 1 else num_samples

        # Convert context to string
        context_str = "\n\n".join([f"### {doc.content}" for doc in context])

        # Generate compressions
        compression_response = self.compress(id, query, context_str, num_samples=num_samples)

        # Process each compression sample and generate predictions
        outputs, pred_outputs, answers, supervisor_usage = self._process_compression_samples(
            compression_response, query
        )

        # Compute majority vote for most common answer
        majority_answer = max(set(answers), key=answers.count) if answers else ""

        return ProtocolResponse(
            text=majority_answer,
            remote_usage=supervisor_usage,
            edge_usage=outputs["compression_usage"],
            meta={
                "edge": {
                    "compression_input_log_prob": outputs["compression_input_log_prob"],
                    "compression_output_log_prob": outputs["compression_output_log_prob"],
                    "compression_num_unique_tokens": outputs["compression_num_unique_tokens"],
                    "compression_text": outputs["compression_text"],
                    "compression_usage": outputs["compression_usage"],
                    "all_samples": outputs["compression_text"],
                },
                "supervisor": {
                    "prediction_input_log_prob": None,
                    "prediction_output_log_prob": None,
                    "prediction_text": pred_outputs,
                    "prediction_usage": None,
                    "prediction_perplexity": outputs["prediction_perplexity"],
                },
            },
        )

    def _process_compression_samples(self, compression_response, query):
        """Process compression samples and generate predictions for each."""
        samples = compression_response.samples
        usages = compression_response.usages
        
        outputs = {
            "compression_text": [],
            "compression_usage": [],
            "compression_input_log_prob": [],
            "compression_output_log_prob": [],
            "compression_num_unique_tokens": [],
            "prediction_perplexity": [],
        }
        pred_outputs = {}
        answers = []
        supervisor_usage = Usage()

        for idx, sample in enumerate(samples):
            # Store compression metadata
            outputs["compression_text"].append(sample.text)
            outputs["compression_usage"].append(usages[idx].to_dict())
            if sample.input_log_prob:
                outputs["compression_input_log_prob"].append(sample.input_log_prob[1:])
                outputs["compression_output_log_prob"].append(sample.log_prob[1:])

            # Count unique tokens if requested
            if self.config.return_num_tokens:
                num_unique_tokens = len(set(self.tokenizer.encode(sample.text)))
                outputs["compression_num_unique_tokens"].append(num_unique_tokens)
            else:
                outputs["compression_num_unique_tokens"].append(0)

            # Generate predictions for this compression
            pred_outputs[str(idx)] = []
            prediction_response = self.decompress(
                query, sample.text, num_samples_prediction=self.config.num_samples_prediction
            )

            # Process each prediction sample
            for prediction_sample in prediction_response.samples:
                prediction_text = prediction_sample.text
                
                # Extract answer from JSON response
                prediction_answer = self._extract_answer_from_prediction(prediction_text)
                if prediction_answer is not None:
                    answers.append(prediction_answer)
                    
                supervisor_usage += prediction_response.usage
                pred_outputs[str(idx)].append(prediction_text)

                # Compute perplexity
                perplexity = self._compute_prediction_perplexity(
                    query, sample.text, prediction_sample
                )
                outputs["prediction_perplexity"].append(perplexity)

        return outputs, pred_outputs, answers, supervisor_usage

    def _extract_answer_from_prediction(self, prediction_text: str) -> Optional[str]:
        """Extract answer from JSON prediction response."""
        try:
            prediction_json = json.loads(prediction_text)
            return str(prediction_json.get("answer", None))
        except Exception as e:
            self.logger.error(
                f"Error parsing JSON in prediction: {e}, text: {prediction_text}",
                exc_info=True,
            )
            return None

    def _compute_prediction_perplexity(self, query: str, summary: str, prediction_sample) -> float:
        """Compute perplexity for a prediction sample."""
        if self.perplexity_client:
            return self._compute_perplexity_with_client(query, summary, prediction_sample)
        elif self.predictor_client.config.model_name.startswith("gpt"):
            return 0.0
        else:
            perplexity = self._compute_perplexity(prediction_sample.log_prob)
            self.logger.info(f"Perplexity: {perplexity}")
            return perplexity

    def _compute_perplexity_with_client(self, query: str, summary: str, prediction_sample) -> float:
        """Compute perplexity using dedicated perplexity client."""
        prediction_prompt = self.config.prediction_prompt.format(query=query, summary=summary)
        perplexity_messages = [
            [
                {"role": "user", "content": prediction_prompt},
                {"role": "assistant", "content": prediction_sample.text},
            ]
        ]
        
        response = self.generate_log_prob(perplexity_messages, self.perplexity_client)
        response = response.samples[0]
        
        # Find the offset where the actual answer starts
        offset = self._find_answer_offset(response.tokens)
        if offset == 0:
            offset = 1  # first logprob is None
            self.logger.info("Answer not found, defaulting to full answer")

        log_probs = response.input_log_prob[offset:-6]
        return self._compute_perplexity(log_probs)

    def _find_answer_offset(self, tokens: List[str]) -> int:
        """Find the offset where the JSON answer starts in the token sequence."""
        for i in range(len(tokens)):
            if (tokens[i : i + 5] == ["{\n", '"', "ex", "planation", '":'] or 
                tokens[i : i + 6] == ["{\n", "   ", ' "', "ex", "planation", '":"']):
                return i
        return 0

    def generate_log_prob(self, chat, client):
        """Generate log probabilities from the LLM model."""
        response = client.chat(chat, max_completion_tokens=1)
        return response

    def _compute_perplexity(self, log_probs: List[float]) -> float:
        """Compute perplexity from log probabilities."""
        return np.exp(-np.mean(log_probs))

    def _fetch_existing_compressions(self, id: str):
        """Fetch existing compression results from cached dataframe."""
        if self.existing_compression_df is None:
            return None

        try:
            problem_rows = self.existing_compression_df[
                self.existing_compression_df["id"] == id
            ]
            
            if problem_rows.empty:
                return None
                
            problem_row = problem_rows.iloc[0]
            
            # Extract compression data from stored response
            edge_meta = problem_row["response"]["meta"]["edge"]
            compression_texts = edge_meta["all_samples"].tolist()
            log_probs = edge_meta["compression_output_log_prob"].tolist()
            input_log_probs = edge_meta["compression_input_log_prob"].tolist()

            # Reconstruct ClientSample objects
            responses = [
                ClientSample(
                    text=compression_text,
                    tokens=None,
                    log_prob=logprob.tolist(),
                    token_ids=None,
                    input_log_prob=input_logprob.tolist(),
                    stop_reason=None,
                )
                for compression_text, logprob, input_logprob in zip(
                    compression_texts, log_probs, input_log_probs
                )
            ]

            # Reconstruct Usage objects
            edge_usage = problem_row["response"]["edge_usage"]
            usages = [
                Usage(
                    prompt_tokens=sample["prompt_tokens"],
                    completion_tokens=sample["completion_tokens"]
                )
                for sample in edge_usage
            ]

            return ClientResponse(samples=responses, usages=usages)
            
        except Exception as e:
            self.logger.error(f"Error fetching existing compressions for id {id}: {e}")
            return None


def create_compression_protocol(config) -> CompressionPredictionProtocol:
    """Factory function to create the appropriate compression protocol based on dataset type."""
    dataset_type = getattr(config, 'dataset_type', None)
    
    if dataset_type == "wildchat":
        from src.ib.pipeline.wildchat_compression import WildchatCompressionProtocol
        return WildchatCompressionProtocol(config)
    elif dataset_type == "fineweb":
        from src.ib.pipeline.fineweb_compression import FinewebCompressionProtocol
        return FinewebCompressionProtocol(config)
    else:
        # Default to base protocol
        return CompressionPredictionProtocol(config)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from text, handling various formats."""
    import re

    # Look for JSON in code blocks first
    block_matches = list(re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL))
    # Then look for JSON objects
    bracket_matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))

    # Take the last match as models often output multiple JSON blocks
    if block_matches:
        json_str = block_matches[-1].group(1).strip()
    elif bracket_matches:
        json_str = bracket_matches[-1].group(0)
    else:
        json_str = text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback for when JSON parsing fails
        return {"answer": "Failed to parse response as JSON."}
