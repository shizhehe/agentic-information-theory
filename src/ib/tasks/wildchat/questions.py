import ast
import json
import os
import random
from typing import Dict, List, Optional

import pandas as pd
import pydrantic
import torch
from pydantic import BaseModel, Field
from pydrantic import ObjectConfig
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.ib.clients.openai import OpenAIClient, openai_cost
from src.ib.utils import get_logger, seed_everything
from src.ib.tasks.wildchat.dataset import load_wildchat_raw, load_conversations_from_string, generate_wildchat_raw_data

logger = get_logger("wildchat_questions")

# System prompt for synthetic QA generation
SYNTHETIC_QA_SYSTEM_PROMPT = """
SYSTEM:
Your are a data generation assistant tasked with creating synthetic QA pairs for a memory-benchmark dataset.
Goal:
- Produce a **realistic new user question** and a **helpful assistant answer**.
- The question must be *novel* (never asked in the chats) yet naturally build on, or refer to, at least one topic that appears in the chats ("memory anchoring").
- The answer should draw on that same memory in a smooth, human-like way (no awkward references, no verbatim quotes). 
- The question should contain enough information as a standalone question.

Rules (must-follow):
1. Do **not** copy text verbatim from the chats; paraphrase instead.
2. Do **not** reveal private or sensitive info that appears in the chats.  
3. Think step-by-step internally but **output only** the JSON described below.
"""

# User prompt for synthetic QA generation
SYNTHETIC_QA_PROMPT = """
Here are the prior multi-turn chats (all from the same user). Use them as potential memory:

CHATS
{chats}

Please respond in the following JSON format:
{{
    "question": "<single user question>",
    "answer": "<assistant reply to that question>"
}}
Your answer (YOU MUST ONLY RESPOND WITH THE JSON OBJECT):
"""

# Example QA pairs for few-shot prompting
WILDCHAT_EXAMPLES = [
    {
        "question": "Can you help me understand the concept of machine learning that we discussed earlier?",
        "answer": "Based on our previous conversation about machine learning, it's a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. The key is that algorithms can identify patterns in data and make predictions or decisions based on those patterns.",
        "user_id": "example_1"
    },
    {
        "question": "What was that Python library you mentioned for data visualization?",
        "answer": "You're thinking of matplotlib, which we discussed for creating charts and graphs. It's one of the most popular Python libraries for data visualization, allowing you to create everything from simple line plots to complex multi-subplot figures.",
        "user_id": "example_2"
    },
    {
        "question": "Can you expand on the cooking technique we talked about last time?",
        "answer": "You're referring to the sautéing technique we discussed. Sautéing involves cooking food quickly in a small amount of oil or butter over relatively high heat, while stirring or tossing frequently. The key is to keep the food moving to ensure even cooking and prevent burning.",
        "user_id": "example_3"
    }
]

# Pydantic model for structured QA generation
class SyntheticQAResponse(BaseModel):
    question: str = Field(description="The question to be answered.")
    answer: str = Field(description="The answer to the question.")

class GenerateConfig(pydrantic.RunConfig):
    """Configuration for generating QA pairs from wildchat conversations."""
    
    name: str = "generate-wildchat-questions"
    
    # Data configuration
    wildchat_data_path: str = "data/wildchat/raw/wildchat_1m_4rounds_100_userwise.csv"
    output_dir: str = "data/wildchat/qa"
    
    # Generation configuration
    generator: ObjectConfig
    num_users: int = 1  # Number of users to process
    min_rounds: int = 4   # Minimum conversation rounds
    max_rounds: int = 8   # Maximum conversation rounds
    conversations_per_user: int = 10
    
    # Model configuration
    temperature: float = 0.4
    top_p: float = 0.95
    max_tokens: int = 512
    
    seed: int = 42

    def run(self):
        return generate_wildchat_questions(self)

class OpenAIGenerator:
    """OpenAI client for generating QA pairs from conversations."""
    
    class Config(ObjectConfig):
        _pass_as_config: bool = True
        model_name: str = "gpt-4o"
        api_key: Optional[str] = None
        timeout: int = 30
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAIClient.Config(
            model_name=config.model_name,
            api_key=config.api_key or os.environ.get("OPENAI_API_KEY")
        ).instantiate()
    
    def generate_qa_pair(self, conversations: List[List[Dict[str, str]]]) -> Dict[str, str]:
        """Generate a single QA pair from conversation history."""
        formatted_chats = self._format_chats(conversations)
        prompt = SYNTHETIC_QA_PROMPT.format(chats=formatted_chats)
        
        try:
            # Use structured output parsing for reliable JSON
            response = self.client.client.beta.chat.completions.parse(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": SYNTHETIC_QA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format=SyntheticQAResponse,
                timeout=self.config.timeout
            )
            
            cost = openai_cost(
                self.config.model_name, 
                response.usage.prompt_tokens, 
                response.usage.completion_tokens
            )
            logger.debug(f"Generated QA pair, cost: ${cost:.4f}")
            
            return response.choices[0].message.parsed.model_dump()
        except Exception as e:
            logger.warning(f"Failed to generate QA pair: {e}")
            return {}
    
    def generate_simple_qa(self, conversations: List[List[Dict[str, str]]]) -> Dict[str, str]:
        """Fallback method using regular chat completion."""
        formatted_chats = self._format_chats(conversations)
        prompt = SYNTHETIC_QA_PROMPT.format(chats=formatted_chats)
        
        try:
            response = self.client.chat(
                chats=[[
                    {"role": "system", "content": SYNTHETIC_QA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]]
            )
            
            json_response = json.loads(response.samples[0].text)
            return json_response
        except Exception as e:
            logger.warning(f"Failed to generate simple QA pair: {e}")
            return {}
    
    def _format_chat(self, chat: List[Dict[str, str]]) -> str:
        """Format a single conversation."""
        formatted = ""
        for message in chat:
            role = message['role']
            formatted += f"{role.capitalize()}: {message['content']}\\n\\n"
        return formatted
    
    def _format_chats(self, conversations: List[List[Dict[str, str]]]) -> str:
        """Format multiple conversations."""
        formatted = ""
        for i, conversation in enumerate(conversations):
            formatted += f"CHAT {i+1}:\\n"
            formatted += self._format_chat(conversation)
        return formatted

def load_conversations(conversations_str: str) -> List[List[Dict[str, str]]]:
    """Load and parse conversation data using the dataset module."""
    return load_conversations_from_string(conversations_str)

def extract_conversation(conversation: List[Dict]) -> List[Dict[str, str]]:
    """Extract relevant fields from conversation."""
    return [
        {
            "content": message["content"],
            "role": message["role"]
        } for message in conversation
    ]

def load_wildchat_data(data_path: str, num_users: int, min_rounds: int) -> pd.DataFrame:
    """Load wildchat dataset using the dataset module."""
    return load_wildchat_raw(data_path, min_rounds, num_users)

def generate_wildchat_questions(config: GenerateConfig):
    """Main function to generate QA pairs from wildchat conversations."""
    seed_everything(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    
    logger.info(f"Starting wildchat question generation with config: {config.name}")

    # Generate raw data if it doesn't exist
    if not os.path.exists(config.wildchat_data_path):
        logger.info(f"Generating raw wildchat data from HuggingFace")
        # get folder name from config.wildchat_data_path
        folder_name = os.path.dirname(config.wildchat_data_path)
        generate_wildchat_raw_data(folder_name, config.num_users, config.min_rounds, config.max_rounds, config.conversations_per_user)
    
    # Load wildchat data
    df = load_wildchat_data(config.wildchat_data_path, config.num_users, config.min_rounds)
    logger.info(f"Loaded {len(df)} users")
    
    # Initialize generator
    generator = config.generator.instantiate()
    
    # Generate QA pairs for each user
    all_qa_pairs = {}
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating QA pairs"):
        user_id = row["user_id"]
        conversations_str = row["conversations"]
        
        try:
            # Load and parse conversations
            conversations = load_conversations(conversations_str)
            logger.debug(f"Processing user {user_id} with {len(conversations)} conversations")
            
            # Generate QA pair for this user
            qa_pair = generator.generate_qa_pair(conversations)
            
            if qa_pair and "question" in qa_pair and "answer" in qa_pair:
                all_qa_pairs[user_id] = qa_pair
                
                # Add to results for DataFrame
                results.append({
                    "user_id": user_id,
                    "conversations": conversations_str,
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "num_conversations": len(conversations)
                })
            else:
                logger.warning(f"No valid QA pair generated for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error processing user {user_id}: {e}")
            continue
    
    # Save results
    qa_pairs_path = os.path.join(config.output_dir, "wildchat_qa_pairs.json")
    with open(qa_pairs_path, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    df_results = pd.DataFrame(results)
    results_path = os.path.join(config.output_dir, "wildchat_questions.feather")
    df_results.to_feather(results_path)
    
    logger.info(f"Generated {len(results)} QA pairs from {len(all_qa_pairs)} users")
    logger.info(f"Saved QA pairs to {qa_pairs_path}")
    logger.info(f"Saved results DataFrame to {results_path}")
    
    return df_results

if __name__ == "__main__":
    config = GenerateConfig(
        generator=OpenAIGenerator.Config()
    )
    pydrantic.main([config])