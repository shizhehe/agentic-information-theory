import ast
import json
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from src.ib.clients import ClientConfig, ClientResponse, Usage
from src.ib.clients.base import ClientSample
from src.ib.pipeline.compression_prediction import CompressionPredictionProtocol
from src.ib.tasks.base import Document
from src.ib.utils import get_logger

logger = get_logger("wildchat_compression")


# Wildchat-specific prompts
GENERAL_COMPRESSION_PROMPT = """
You are a memory compression assistant, tasked with summarizing a chat conversation. 
Produce a summary that preserves all details that could be useful as memory for a language model, and keep the summary as concise as possible.
DO NOT invent any information.

CHAT: 
{conversation}

Your summary (Just plain text, no formatting.):
""" 

GENERAL_COMPRESSION_PROMPT_GEMMA = """
<start_of_turn>user
You are a memory compression assistant, tasked with summarizing a chat conversation. 
Produce a summary that preserves all details that could be useful as memory for a language model, and keep the summary as concise as possible.
DO NOT invent any information.

CHAT: 
{conversation}

Your summary (Just plain text, no formatting.):<end_of_turn>
<start_of_turn>model
""" 

QA_SPECIFIC_COMPRESSION_PROMPT = """
You are a memory compression assistant, tasked with summarizing a chat conversation to answer a specific question. 
Produce a summary that preserves all details that could be useful as memory for a language model to answer the question, and keep the summary as concise as possible.
DO NOT invent any information.

QUESTION:
{question}

CHAT: 
{conversation}

Your summary (Just plain text, no formatting.)
"""

QA_PREDICTION_PROMPT = """
Please answer the following question based on the provided {memory_type}.

Question:
{query}

{memory_type}:
{memory}

Please respond in the following JSON format:
<briefly think about the information you have and the question you need to answer>

{{
    "explanation": "<brief explanation of the answer. explain how you arrived at the answer. 1-2 sentences>",
    "answer": "<your final answer>"
}}

Your answer (YOU MUST ONLY RESPOND WITH THE JSON OBJECT):
"""


class WildchatPredictionResponse(BaseModel):
    answer: str = Field(description="The final answer to the question.")
    explanation: str = Field(description="A brief explanation of the answer.")

class WildchatCompressionProtocol(CompressionPredictionProtocol):
    """Specialized compression protocol for Wildchat conversation data."""
    
    class Config(CompressionPredictionProtocol.Config):
        # Wildchat-specific configuration options
        general_compression: bool = True  # vs question-specific compression
        per_conversation_compression: bool = True  # compress each conversation separately
        memory_type_name: str = "chat memory"  # vs "chat histories" for baseline
        
        # Override default prompts
        compression_prompt: str = GENERAL_COMPRESSION_PROMPT
        prediction_prompt: str = QA_PREDICTION_PROMPT

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
        """Compress conversations individually for Wildchat data."""
        if self.existing_compression_df is not None:
            existing_compressions = self._fetch_existing_compressions(id)
            if existing_compressions is not None:
                self.logger.info(
                    f"Found {len(existing_compressions.samples)} existing compressions for id {id}"
                )
                return existing_compressions

        # Parse the context as conversation data
        conversations = self._parse_conversations_from_context(context)

        self.logger.info(f"Compressing {len(conversations)} conversations")
        
        if self.config.per_conversation_compression:
            self.logger.info("Compressing conversations individually")
            return self._compress_conversations_individually(conversations, query, num_samples)
        else:
            # Fall back to standard compression
            self.logger.info("Compressing all conversations together")
            return super().compress(id, query, context, num_samples, **kwargs)

    def decompress(
        self, query: str, summary: str, num_samples_prediction: int = 1, **kwargs
    ) -> ClientResponse:
        """Generate predictions using Wildchat-specific prompt format."""
        if num_samples_prediction == 0:
            return ClientResponse(samples=[], usage=Usage())

        # Use memory type name from config
        memory_type = self.config.memory_type_name
        prediction_prompt = self.config.prediction_prompt.format(
            query=query, memory=summary, memory_type=memory_type
        )
        
        prediction_messages = [
            [{"role": "user", "content": prediction_prompt}]
        ] * num_samples_prediction

        response = self.predictor_client.chat(
            prediction_messages,
            temperature=self.config.supervisor_temperature,
            response_format={
                "type": "json_object",
                **(
                    {"schema": WildchatPredictionResponse.model_json_schema()}
                    if not self.predictor_client.config.model_name.startswith("gpt")
                    else {}
                ),
            },
        )
        return response

    def _parse_conversations_from_context(self, context: str) -> List[List[Dict[str, str]]]:
        """Parse conversation data from the context string."""
        try:
            # Context should be formatted as conversation data
            # Extract conversations from Document content
            conversations = []
            
            # Split context by document separators
            doc_sections = context.split("### ")
            for section in doc_sections:
                if section.strip():
                    # Try to parse as conversation data
                    try:
                        # Look for conversation data in the section
                        lines = section.strip().split('\n')
                        for line in lines:
                            if line.startswith("CHAT ") and ":" in line:
                                # This looks like formatted conversation data
                                chat_data = self._extract_conversation_from_formatted_text(section)
                                if chat_data:
                                    conversations.extend(chat_data)
                                break
                    except Exception as e:
                        self.logger.warning(f"Failed to parse conversation section: {e}")
                        continue
            
            if not conversations:
                self.logger.warning("No conversations found in context, treating as single conversation")
                # Fallback: treat the entire context as a single conversation
                conversations = [[{"role": "user", "content": context}]]
                
            return conversations
            
        except Exception as e:
            self.logger.error(f"Error parsing conversations from context: {e}")
            # Fallback to treating context as single conversation
            return [[{"role": "user", "content": context}]]

    def _extract_conversation_from_formatted_text(self, text: str) -> List[List[Dict[str, str]]]:
        """Extract conversations from formatted text like 'User: ... Assistant: ...'"""
        conversations = []
        current_conversation = []
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("CHAT ") and current_conversation:
                # Start of new conversation
                conversations.append(current_conversation)
                current_conversation = []
            elif line.startswith("User:") or line.startswith("user:"):
                content = line[5:].strip()  # Remove "User:" prefix
                current_conversation.append({"role": "user", "content": content})
            elif line.startswith("Assistant:") or line.startswith("assistant:"):
                content = line[10:].strip()  # Remove "Assistant:" prefix
                current_conversation.append({"role": "assistant", "content": content})
        
        # Add the last conversation if it exists
        if current_conversation:
            conversations.append(current_conversation)
            
        return conversations

    def _compress_conversations_individually(
        self, conversations: List[List[Dict[str, str]]], query: str, num_samples: int
    ) -> ClientResponse:
        """Compress each conversation individually, then combine summaries."""
        all_summaries = []
        all_usages = []
        
        for conversation in conversations:
            # Format conversation for compression
            conversation_text = self._format_conversation(conversation)
            
            # Choose compression prompt based on configuration
            if self.config.general_compression:
                compression_prompt = self.config.compression_prompt.format(
                    conversation=conversation_text
                )
            else:
                compression_prompt = QA_SPECIFIC_COMPRESSION_PROMPT.format(
                    conversation=conversation_text, question=query
                )
            
            # Create messages for this conversation
            compression_messages = [
                [{"role": "user", "content": compression_prompt}]
            ] * num_samples
            
            # Get compression for this conversation
            response = self.compressor_client.chat(
                compression_messages,
                temperature=self.config.worker_temperature,
                max_completion_tokens=self.config.worker_max_new_tokens,
            )
            
            # Store summaries from this conversation
            for i, sample in enumerate(response.samples):
                if i >= len(all_summaries):
                    all_summaries.append([])
                all_summaries[i].append(sample.text.strip())
            
            # Accumulate usage
            all_usages.extend(response.usages)
        
        # Combine summaries for each sample
        combined_samples = []
        for summary_list in all_summaries:
            combined_text = "\n\n".join(summary_list)
            combined_samples.append(ClientSample(
                text=combined_text,
                tokens=None,
                log_prob=None,
                token_ids=None,
                input_log_prob=None,
                stop_reason=None,
            ))
        
        return ClientResponse(samples=combined_samples, usages=all_usages)

    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format a conversation for compression."""
        formatted = ""
        for message in conversation:
            role = message['role'].capitalize()
            formatted += f"{role}: {message['content']}\n\n"
        return formatted.strip()

    def compute_baseline_perplexity(
        self, perplexity_client, conversations: List[List[Dict[str, str]]], 
        query: str, target: str
    ) -> tuple:
        """Compute perplexity using full conversation history as baseline."""
        # Format all conversations as baseline memory
        memory = ""
        for i, conversation in enumerate(conversations):
            memory += f"CHAT {i+1}:\n"
            memory += self._format_conversation(conversation)
            memory += "\n\n"
        
        return self._compute_perplexity_with_memory(
            perplexity_client, memory.strip(), query, target, baseline=True
        )

    def _compute_perplexity_with_memory(
        self, perplexity_client, memory: str, query: str, target: str, baseline: bool = False
    ) -> tuple:
        """Compute perplexity for a given memory and query."""
        memory_type = "chat histories" if baseline else self.config.memory_type_name
        
        # Prepare the chat for perplexity computation
        format_prompt = QA_PREDICTION_PROMPT.format(
            memory=memory, query=query, memory_type=memory_type
        )
        
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
                perplexity_logprobs = []
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
            # Look for assistant response start
            # NOTE: This is dependent on the model/tokenizer used
            if (tokens[i:i+5] == ["_start", "|", ">", "assistant", "\n"] or
                tokens[i:i+3] == ["<|im_start|>", "assistant", "\n"]):
                assistant_idx = i + (5 if tokens[i] == "_start" else 3) + 3
                
            # Look for response end
            # NOTE: This is dependent on the model/tokenizer used
            if assistant_idx is not None:
                if (tokens[i:i+4] == ["}<", "|", "im", "_end"] or
                    tokens[i:i+5] == ['"}', '<', '|', 'im', '_end'] or
                    tokens[i:i+1] == ["<|im_end|>"]):
                    end_idx = i
                    break
        
        return assistant_idx, end_idx