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
from src.ib.tasks.fineweb.dataset import load_fineweb_raw, generate_fineweb_raw_data

logger = get_logger("fineweb_questions")

# Synthetic QA generation prompt adapted for fineweb content
SYNTHETIC_QA_PROMPT = """
You are generating synthetic question-answer (QA) pairs from a source text.

SOURCE_TEXT:
{context}

Use only information from SOURCE_TEXT. No hallucinated facts. 
Generate five questions and answers:
- Question 1: What is {topic} and why is it important? (type = "qa")
- Question 2: What is {topic} and how does it work? (type = "qa")
- Question 3: Write an email to a colleague summarizing the findings and take-aways. (type = "generation")
- Question 4: Generate rap lyrics that teach the core concepts. (type = "generation")
- Question 5: Generate a poem about the topic. (type = "generation")

Please respond in the following JSON format:
<briefly think about the information you have and questions you can generate from it>

{{
    "questions": [
        {{
            "topic": "<topic 1>",
            "question": "<question 1>",
            "answer": "<answer 1>",
            "type": "qa"
        }},
        {{
            "topic": "<topic 2>",
            "question": "<question 2>",
            "answer": "<answer 2>",
            "type": "qa"
        }},
        {{
            "topic": "<topic 3>",
            "question": "<question 3>",
            "answer": "<answer 3>",
            "type": "generation"
        }},
        {{
            "topic": "<topic 4>",
            "question": "<question 4>",
            "answer": "<answer 4>",
            "type": "generation"
        }},
        {{
            "topic": "<topic 5>",
            "question": "<question 5>",
            "answer": "<answer 5>",
            "type": "generation"
        }}
    ]
}}

Your answer (YOU MUST ONLY RESPOND WITH THE JSON OBJECT):
"""

# Pydantic models for structured QA generation
class SyntheticQAPair(BaseModel):
    topic: str = Field(description="The topic of the question.")
    question: str = Field(description="The question to be answered.")
    answer: str = Field(description="The answer to the question.")
    type: str = Field(description="The type of the question (qa or generation).")

class SyntheticQAResponse(BaseModel):
    questions: List[SyntheticQAPair] = Field(
        description="The list of synthetic QA pairs.", 
        min_length=5, # NOTE: change based on the number of questions you want to generate
        max_length=5
    )

class GenerateConfig(pydrantic.RunConfig):
    """Configuration for generating QA pairs from fineweb data."""
    
    name: str = "generate-fineweb-questions"
    
    # Data configuration
    fineweb_data_path: str = "data/fineweb/raw/fineweb_15k_28k_1.csv"
    output_dir: str = "data/fineweb/qa"
    
    # Generation configuration
    generator: ObjectConfig
    num_samples: int = 1  # Number of fineweb documents to process
    
    # Model configuration
    temperature: float = 0.4
    top_p: float = 0.95
    max_tokens: int = 1024
    
    seed: int = 42

    def run(self):
        return generate_fineweb_questions(self)

class OpenAIGenerator:
    """OpenAI client for generating QA pairs."""
    
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
    
    def generate_qa_pairs(self, context: str) -> List[Dict[str, str]]:
        """Generate structured QA pairs from context using OpenAI."""
        prompt = SYNTHETIC_QA_PROMPT.format(context=context, topic="the main topic")
        
        try:
            # Use structured output parsing for reliable JSON
            response = self.client.client.beta.chat.completions.parse(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=SyntheticQAResponse,
                timeout=self.config.timeout
            )
            
            return [
                qa_pair.model_dump() 
                for qa_pair in response.choices[0].message.parsed.questions
            ]
        except Exception as e:
            logger.warning(f"Failed to generate QA pairs: {e}")
            return []
    
    def generate_simple_qa(self, context: str) -> List[Dict[str, str]]:
        """Fallback method using regular chat completion."""
        prompt = SYNTHETIC_QA_PROMPT.format(context=context, topic="the main topic")
        
        try:
            response = self.client.chat(
                chats=[[{"role": "user", "content": prompt}]]
            )
            
            json_response = json.loads(response.samples[0].text)
            return json_response.get("questions", [])
        except Exception as e:
            logger.warning(f"Failed to generate simple QA pairs: {e}")
            return []

def load_fineweb_data(data_path: str, num_samples: int) -> pd.DataFrame:
    """Load fineweb dataset using the dataset module."""
    return load_fineweb_raw(data_path, num_samples)

def generate_fineweb_questions(config: GenerateConfig):
    """Main function to generate QA pairs from fineweb data."""
    seed_everything(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    if not os.path.exists(config.fineweb_data_path):
        logger.info(f"Generating raw fineweb data from HuggingFace")
        # get folder name from config.fineweb_data_path
        folder_name = os.path.dirname(config.fineweb_data_path)
        generate_fineweb_raw_data(folder_name, config.num_samples)
    
    logger.info(f"Starting fineweb question generation with config: {config.name}")
    
    # Load fineweb data
    df = load_fineweb_data(config.fineweb_data_path, config.num_samples)
    logger.info(f"Loaded {len(df)} fineweb documents")
    
    # Initialize generator
    generator = config.generator.instantiate()
    
    # Generate QA pairs for each document
    all_qa_pairs = {}
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating QA pairs"):
        doc_id = row["id"]
        text = row["text"]
        
        # Generate QA pairs for this document
        qa_pairs = generator.generate_qa_pairs(text)
        
        if qa_pairs:
            all_qa_pairs[doc_id] = qa_pairs
            
            # Add to results for DataFrame
            for qa_pair in qa_pairs:
                results.append({
                    "doc_id": doc_id,
                    "text": text,
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "topic": qa_pair["topic"],
                    "type": qa_pair["type"]
                })
        else:
            logger.warning(f"No QA pairs generated for document {doc_id}")
    
    # Save results
    qa_pairs_path = os.path.join(config.output_dir, "fineweb_qa_pairs.json")
    with open(qa_pairs_path, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    df_results = pd.DataFrame(results)
    results_path = os.path.join(config.output_dir, "fineweb_questions.feather")
    df_results.to_feather(results_path)
    
    logger.info(f"Generated {len(results)} QA pairs from {len(all_qa_pairs)} documents")
    logger.info(f"Saved QA pairs to {qa_pairs_path}")
    logger.info(f"Saved results DataFrame to {results_path}")
    
    return df_results

if __name__ == "__main__":
    config = GenerateConfig(
        generator=OpenAIGenerator.Config()
    )
    pydrantic.main([config])