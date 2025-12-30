import json
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from src.ib.clients import ClientConfig, ClientResponse, Usage
from src.ib.clients.base import ClientSample
from src.ib.pipeline.compression_prediction import CompressionPredictionProtocol
from src.ib.tasks.base import Document
from src.ib.utils import get_logger

logger = get_logger("fineweb_compression")


# Fineweb-specific prompts
GENERAL_COMPRESSION_PROMPT = """
Summarize the following text and produce a summary that preserves all details that could be needed to answer likely questions about the text. Do NOT invent facts.

Do NOT answer any question; just summarize potential answer-bearing info, and keep the summary as concise as possible.

Text: 
{text}

Your summary (make sure to include all important details / background information related. Just plain text, no formatting.)
""" 

GENERAL_COMPRESSION_PROMPT_GEMMA = """
<start_of_turn>user
Summarize the following text and produce a summary that preserves all details that could be needed to answer likely questions about the text. Do NOT invent facts.

Do NOT answer any question; just summarize potential answer-bearing info, and keep the summary as concise as possible.

Text: 
{text}

Your summary (make sure to include all important details / background information related. Just plain text, no formatting.):<end_of_turn>
<start_of_turn>model
""" 

QA_COMPRESSION_PROMPT = """
Summarize the following text to include ONLY information needed to answer the question.
Extract the key points relevant to the question.
DO NOT ANSWER THE QUESTION DIRECTLY.

Question: 
{question}

Text: 
{text}

Your summary (make sure to include all important details / background information related to the *question*. Just plain text, no formatting. **DO NOT ANSWER THE QUESTION**)
"""

QA_PREDICTION_PROMPT = """
Please answer the following question based on the provided {context_type}.

Question:
{query}

{context_type}:
{summary}

Please respond in the following JSON format:
<briefly think about the information you have and the question you need to answer>

{{
    "answer": "<your final answer>"
}}

Your answer (YOU MUST ONLY RESPOND WITH THE JSON OBJECT):
"""

QA_GENERATION_PROMPT = """
Please do the following based on the provided {context_type}.

Task:
{query}

{context_type}:
{summary}

Please respond in the following JSON format:
<briefly think about the information you have and the question you need to answer>

{{
    "answer": "<your final answer>"
}}

Your answer (YOU MUST ONLY RESPOND WITH THE JSON OBJECT):
"""


class FinewebPredictionResponse(BaseModel):
    answer: str = Field(description="The final answer to the question.")
    explanation: Optional[str] = Field(description="A brief explanation of the answer.") 


class FinewebCompressionProtocol(CompressionPredictionProtocol):
    """Specialized compression protocol for Fineweb document data."""
    
    class Config(CompressionPredictionProtocol.Config):
        # Fineweb-specific configuration options
        general_compression: bool = True  # vs question-specific compression
        context_type_name: str = "Summary"  # vs "Context" for baseline
        
        # Override default prompts
        compression_prompt: str = GENERAL_COMPRESSION_PROMPT
        qa_compression_prompt: str = QA_COMPRESSION_PROMPT
        prediction_prompt: str = QA_PREDICTION_PROMPT
        generation_prompt: str = QA_GENERATION_PROMPT

    def __init__(self, config: Config):
        super().__init__(config)
        self.logger = get_logger(__name__)
        
        # Adapt compression prompt for Gemma models
        if hasattr(self.compressor_client.config, "api_base_url"):
            if (self.compressor_client.config.api_base_url is not None and 
                "gemma-3" in self.compressor_client.config.model_name.lower()):
                self.logger.info("Using Gemma-specific compression prompt")
                self.config.compression_prompt = GENERAL_COMPRESSION_PROMPT_GEMMA


    def compress(
        self, id: str, query: str, context: str, num_samples: int = 1, **kwargs
    ) -> ClientResponse:
        """Compress document context for Fineweb data."""
        if self.existing_compression_df is not None:
            existing_compressions = self._fetch_existing_compressions(id)
            if existing_compressions is not None:
                self.logger.info(
                    f"Found {len(existing_compressions.samples)} existing compressions for id {id}"
                )
                return existing_compressions

        # Choose compression prompt based on configuration
        if self.config.general_compression:
            compression_prompt = self.config.compression_prompt.format(text=context)
        else:
            compression_prompt = self.config.qa_compression_prompt.format(
                text=context, question=query
            )
        
        # Create messages for compression
        compression_messages = [
            [{"role": "user", "content": compression_prompt}]
        ] * num_samples
        
        # Get compression
        response = self.compressor_client.chat(
            compression_messages,
            temperature=self.config.worker_temperature,
            max_completion_tokens=self.config.worker_max_new_tokens,
        )
        
        return response

    def decompress(
        self, query: str, summary: str, num_samples_prediction: int = 1, query_type: str = "qa", **kwargs
    ) -> ClientResponse:
        """Generate predictions using Fineweb-specific prompt format."""
        if num_samples_prediction == 0:
            return ClientResponse(samples=[], usage=Usage())

        # Use context type name from config
        context_type = self.config.context_type_name
        
        # Choose prediction prompt based on query type
        if query_type == "qa":
            prediction_prompt = self.config.prediction_prompt.format(
                query=query, summary=summary, context_type=context_type
            )
        elif query_type == "generation":
            prediction_prompt = self.config.generation_prompt.format(
                query=query, summary=summary, context_type=context_type
            )
        else:
            raise ValueError(f"Invalid query type: {query_type}")
        
        prediction_messages = [
            [{"role": "user", "content": prediction_prompt}]
        ] * num_samples_prediction

        response = self.predictor_client.chat(
            prediction_messages,
            temperature=self.config.supervisor_temperature,
            response_format={
                "type": "json_object",
                **(
                    {"schema": FinewebPredictionResponse.model_json_schema()}
                    if not self.predictor_client.config.model_name.startswith("gpt")
                    else {}
                ),
            },
        )
        return response

    def compute_baseline_perplexity(
        self, perplexity_client, context: str, query: str, target: str, query_type: str = "qa"
    ) -> tuple:
        """Compute perplexity using full document context as baseline."""
        return self._compute_perplexity_with_context(
            perplexity_client, context, query, target, query_type, baseline=True
        )

    def _compute_perplexity_with_context(
        self, perplexity_client, context: str, query: str, target: str, 
        query_type: str = "qa", baseline: bool = False
    ) -> tuple:
        """Compute perplexity for a given context and query."""
        context_type = "Context" if baseline else self.config.context_type_name
        
        # Choose prediction prompt based on query type
        if query_type == "qa":
            format_prompt = self.config.prediction_prompt.format(
                summary=context, query=query, context_type=context_type
            )
        elif query_type == "generation":
            format_prompt = self.config.generation_prompt.format(
                summary=context, query=query, context_type=context_type
            )
        else:
            raise ValueError(f"Invalid query type: {query_type}")
        
        json_answer = {"answer": target}
        
        chat = [[
            {"role": "user", "content": format_prompt},
            {"role": "assistant", "content": json.dumps(json_answer)}
        ]]
        
        # Generate log probabilities
        response = perplexity_client.chat(chat, max_completion_tokens=1)
        
        all_perplexity_scores = []
        all_perplexity_tokens = []
        all_perplexity_logprobs = []
        
        for sample in response.samples:
            tokens = sample.tokens
            perplexity_logprobs = sample.input_log_prob
            
            # Find assistant response tokens
            assistant_idx, end_idx = self._find_assistant_response_indices(tokens)
            
            if assistant_idx is None or end_idx is None:
                self.logger.warning("Can't find assistant response tokens for perplexity")
                perplexity = 0.0
                perplexity_tokens = []
                response_logprobs = []
            else:
                perplexity_tokens = tokens[assistant_idx:end_idx]
                response_logprobs = perplexity_logprobs[assistant_idx:end_idx]
                perplexity = np.exp(-np.mean(response_logprobs))
                
            all_perplexity_scores.append(perplexity)
            all_perplexity_tokens.append(perplexity_tokens)
            all_perplexity_logprobs.append(response_logprobs)
        
        return all_perplexity_scores, all_perplexity_tokens, all_perplexity_logprobs

    def _find_assistant_response_indices(self, tokens: List[str]) -> tuple:
        """Find the start and end indices of the assistant response in tokens."""
        assistant_idx = None
        end_idx = None
        
        for i in range(len(tokens)):
            # Look for assistant response start patterns
            # NOTE: This is dependent on the model/tokenizer used
            if tokens[i:i+5] == ("_start", "|", ">", "assistant", "\n"):
                assistant_idx = i + 5 + 3
            if tokens[i:i+3] == ("<|im_start|>", "assistant", "\n"):
                assistant_idx = i + 3 + 3
                
            # Look for response end patterns
            # NOTE: This is dependent on the model/tokenizer used
            if assistant_idx is not None:
                if (tokens[i:i+4] == ("}<", "|", "im", "_end") or
                    tokens[i:i+4] == ("<", "|", "im", "_end") or
                    tokens[i:i+5] == ('"}', '<', '|', 'im', '_end') or
                    tokens[i:i+1] == ("<|im_end|>",)):
                    end_idx = i
                    break
        
        return assistant_idx, end_idx

    def generate_summaries_with_qa_pairs(
        self, context_id: str, context: str, qa_pairs: List[Dict[str, str]], 
        num_samples: int = 1, general: bool = False
    ) -> List[Dict]:
        """Generate summaries for each QA pair (mirrors fineweb perplexity.py logic)."""
        all_qa_pairs = []
        chats = []
        gemma = "gemma-3" in self.compressor_client.config.model_name.lower()
        
        if general:
            # Generate general summary
            if gemma:
                general_prompt = GENERAL_COMPRESSION_PROMPT_GEMMA.format(text=context)
            else:
                general_prompt = GENERAL_COMPRESSION_PROMPT.format(text=context)
            
            for qa_idx, qa_pair in enumerate(qa_pairs.copy()):
                for i in range(num_samples):
                    cur_qa_pair = qa_pair.copy()
                    cur_qa_pair["S"] = i
                    all_qa_pairs.append(cur_qa_pair)

                    if qa_idx == 0:
                        chats.append([{"role": "user", "content": general_prompt}])
        else:
            # Generate question-specific summaries
            for qa_pair in qa_pairs.copy():
                prompt = QA_COMPRESSION_PROMPT.format(text=context, question=qa_pair["question"])
                for i in range(num_samples):
                    chats.append([{"role": "user", "content": prompt}])
                    cur_qa_pair = qa_pair.copy()
                    cur_qa_pair["S"] = i
                    all_qa_pairs.append(cur_qa_pair)

        self.logger.info(f"Generating {len(chats)} summaries")
        summary_responses = self.compressor_client.chat(
            chats=chats, 
            max_completion_tokens=4096-256, 
            temperature=1.0 if gemma else 0.6, 
        )
        
        summaries = [
            {
                "S": qa_pair["S"],
                "compressor_model_url": self.compressor_client.config.api_base_url,
                "context_id": context_id,
                "context": context,
                "topic": qa_pair["topic"],
                "question": qa_pair["question"],
                "summary": summary_responses.samples[idx].text if not general else summary_responses.samples[(idx % len(qa_pairs))].text,
                "answer": qa_pair["answer"],
                "type": qa_pair["type"],
                "num_input_tokens": summary_responses.usages[idx].prompt_tokens if not general else summary_responses.usages[idx % len(qa_pairs)].prompt_tokens,
                "num_output_tokens": summary_responses.usages[idx].completion_tokens if not general else summary_responses.usages[idx % len(qa_pairs)].completion_tokens
            } for idx, qa_pair in enumerate(all_qa_pairs)
        ]
        return summaries