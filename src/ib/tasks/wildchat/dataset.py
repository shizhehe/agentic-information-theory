import ast
import json
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd
from datasets import load_dataset
from pydantic import BaseModel, field_validator

from src.ib.tasks.base import Document, Problem
from src.ib.utils import get_logger

logger = get_logger("wildchat_dataset")


class WildchatMessage(BaseModel):
    role: str
    content: str
    
    
class WildchatConversation(BaseModel):
    conversation_hash: str
    model: str
    timestamp: str
    messages: List[WildchatMessage]
    turn_count: int
    

class WildchatUser(BaseModel):
    user_id: int
    conversations: List[WildchatConversation]
    question: Optional[str] = None
    answer: Optional[str] = None
    
    def get_total_conversations(self) -> int:
        return len(self.conversations)
    
    def get_conversation_text(self) -> str:
        """Format all conversations as a single text block."""
        conversation_text = ""
        for i, conversation in enumerate(self.conversations):
            conversation_text += f"CHAT {i+1}:\n"
            for message in conversation.messages:
                role = message.role.capitalize()
                conversation_text += f"{role}: {message.content}\n\n"
        return conversation_text.strip()


def load_wildchat_dataset(
    user_ids: Optional[List[int]] = None,
    qa_data_path: str = "data/wildchat/qa/wildchat_questions.feather",
    raw_data_path: Optional[str] = None,
    cache_dir: str = "/scr-ssd/shizhehe/datasets"
) -> List[WildchatUser]:
    """
    Load the WildChat dataset with automatic data location handling.
    
    This function follows the LongHealth pattern - it automatically:
    1. Checks for existing QA data at qa_data_path
    2. Falls back to raw data if QA data doesn't exist
    3. Can generate raw data from HuggingFace if needed
    
    Args:
        user_ids: Optional list of user IDs to filter by
        qa_data_path: Path to QA data (feather file)
        raw_data_path: Optional path to raw CSV data
        cache_dir: Directory for HuggingFace dataset cache
        
    Returns:
        List of WildchatUser objects
    """
    logger.info("Loading WildChat dataset")
    
    # First, try to load from QA data if it exists
    print(f"QA data path: {qa_data_path}")
    if os.path.exists(qa_data_path):
        logger.info(f"Loading WildChat QA data from {qa_data_path}")
        
        df = pd.read_feather(qa_data_path)
        
        # Filter by user IDs if specified
        if user_ids is not None:
            df = df[df['user_id'].isin(user_ids)]
            logger.info(f"Filtered to {len(df)} users")
        
        users = []
        for _, row in df.iterrows():
            # Parse conversations 
            conversations_data = ast.literal_eval(row["conversations"])
            conversations = []
            
            for conv_json in conversations_data:
                conv_data = json.loads(conv_json)
                messages = [
                    WildchatMessage(role=msg["role"], content=msg["content"])
                    for msg in conv_data
                ]
                
                conversations.append(WildchatConversation(
                    conversation_hash=f"qa_hash_{len(conversations)}",
                    model="unknown",
                    timestamp="unknown", 
                    messages=messages,
                    turn_count=len(messages)
                ))
            
            users.append(WildchatUser(
                user_id=row["user_id"],
                conversations=conversations,
                question=row["question"],
                answer=row["answer"]
            ))
        
        logger.info(f"Loaded {len(users)} users with QA pairs")
        return users
    
    # If QA data doesn't exist, try to load from raw data
    if raw_data_path and os.path.exists(raw_data_path):
        logger.info(f"QA data not found, loading raw data from {raw_data_path}")
        
        df = pd.read_csv(raw_data_path)
        
        # Filter by user IDs if specified
        if user_ids is not None:
            df = df[df['user_id'].isin(user_ids)]
            logger.info(f"Filtered to {len(df)} users")
        
        users = []
        for _, row in df.iterrows():
            # Parse conversations
            conversations_data = ast.literal_eval(row["conversations"])
            conversations = []
            
            for i, conv_json in enumerate(conversations_data):
                conv_data = json.loads(conv_json)
                messages = [
                    WildchatMessage(role=msg["role"], content=msg["content"])
                    for msg in conv_data
                ]
                
                # Get metadata for this conversation
                conversation_hash = row["conversation_hashes"][i] if isinstance(row["conversation_hashes"], list) else f"hash_{i}"
                model = row["models"][i] if isinstance(row["models"], list) else "unknown"
                timestamp = row["timestamps"][i] if isinstance(row["timestamps"], list) else "unknown"
                turn_count = row["turns"][i] if isinstance(row["turns"], list) else len(messages)
                
                conversations.append(WildchatConversation(
                    conversation_hash=conversation_hash,
                    model=model,
                    timestamp=timestamp,
                    messages=messages,
                    turn_count=turn_count
                ))
            
            users.append(WildchatUser(
                user_id=row["user_id"],
                conversations=conversations
            ))
        
        logger.info(f"Loaded {len(users)} users from raw data (no QA pairs)")
        return users
    
    # If no existing data found, provide helpful error message
    raise FileNotFoundError(
        f"No WildChat data found. Please ensure one of the following exists:\n"
        f"  - QA data: {qa_data_path}\n"
        f"  - Raw data: {raw_data_path if raw_data_path else 'Not specified'}\n\n"
        f"To generate data, use:\n"
        f"  - generate_wildchat_raw_data() for raw conversation data\n"
        f"  - WildChat questions.py script for QA generation"
    )

def extract_conversation(conversation):
    """Extract relevant fields from conversation data."""
    return [
        {
            "content": message["content"],
            "role": message["role"]
        } for message in conversation
    ]

def generate_wildchat_raw_data(
    output_path: str = "data/wildchat/raw",
    num_users: int = 100,
    min_rounds: int = 4,
    max_rounds: int = 8,
    conversations_per_user: int = 10,
    cache_dir: str = "data/cache"
) -> pd.DataFrame:
    """
    Generate raw wildchat data from HuggingFace WildChat-1M dataset.
    
    Args:
        output_path: Directory to save the generated data
        num_samples: Number of conversation samples to collect
        min_rounds: Minimum number of conversation rounds
        max_rounds: Maximum number of conversation rounds
        conversations_per_user: Number of conversations to group per user
        cache_dir: Directory for HuggingFace dataset cache
        
    Returns:
        DataFrame with user-grouped conversation data
    """
    logger.info(f"Loading WildChat-1M dataset from HuggingFace")
    
    # Load the WildChat-1M dataset
    fineweb_dataset = load_dataset(
        "allenai/WildChat-1M", 
        split="train", 
        streaming=True, 
        cache_dir=cache_dir
    )
    
    samples = []
    
    logger.info(f"Collecting {num_users * conversations_per_user} samples with {min_rounds}-{max_rounds} rounds")
    for sample in fineweb_dataset:
        if len(samples) >= num_users * conversations_per_user:
            break
            
        # Filter by turn count
        if sample["turn"] > max_rounds:
            continue
            
        # Filter by language
        if sample["language"] != "English":
            continue
            
        # Only include conversations with sufficient rounds
        if sample["turn"] >= min_rounds:
            samples.append({
                "conversation_hash": sample["conversation_hash"],
                "model": sample["model"],
                "timestamp": sample["timestamp"],
                "conversation": json.dumps(extract_conversation(sample["conversation"])),
                "turn": sample["turn"],
            })
    
    logger.info(f"Collected {len(samples)} conversation samples")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save individual conversation dataset
    dataset_df = pd.DataFrame(samples)
    dataset_path = os.path.join(output_path, f"wildchat_1m_{min_rounds}rounds_{len(dataset_df)}.csv")
    dataset_df.to_csv(dataset_path, index=False)
    logger.info(f"Saved conversation dataset to {dataset_path}")
    
    # Group conversations into users (every N conversations = 1 user)
    user_samples = []
    for i in range(0, len(dataset_df), conversations_per_user):
        user_df = dataset_df.iloc[i:i+conversations_per_user]
        user_samples.append({
            "user_id": i // conversations_per_user,
            "conversation_hashes": user_df.conversation_hash.tolist(),
            "models": user_df.model.tolist(),
            "timestamps": user_df.timestamp.tolist(),
            "conversations": user_df.conversation.tolist(),
            "turns": user_df.turn.tolist(),
        })
    
    # Save user-grouped dataset
    user_dataset_df = pd.DataFrame(user_samples)
    user_dataset_path = os.path.join(
        output_path, 
        f"wildchat_1m_{min_rounds}rounds_{len(user_dataset_df)}_userwise.csv"
    )
    user_dataset_df.to_csv(user_dataset_path, index=False)
    
    logger.info(f"Created {len(user_samples)} users from {len(samples)} conversations")
    logger.info(f"Saved user dataset to {user_dataset_path}")
    
    return user_dataset_df

def load_conversations_from_string(conversations_str: str) -> List[List[Dict[str, str]]]:
    """Parse conversation data from string format."""
    if isinstance(conversations_str, str):
        conversations_data = ast.literal_eval(conversations_str)
    else:
        conversations_data = conversations_str
    
    return [
        json.loads(conversation) for conversation in conversations_data
    ]

def load_wildchat_questions(
    data_path: str = "data/wildchat/qa/wildchat_questions.feather",
    user_ids: Optional[List[int]] = None
) -> List[Problem]:
    """
    Load wildchat QA dataset from feather file and convert to Problem objects.
    
    Args:
        data_path: Path to the feather file containing QA data
        user_ids: Optional list of user IDs to filter by
        
    Returns:
        List of Problem objects
    """
    # Load the dataset using the main loader function
    users = load_wildchat_dataset(
        user_ids=user_ids,
        qa_data_path=data_path
    )
    
    # Convert to Problem objects
    problems = []
    for user in users:
        if user.question is None or user.answer is None:
            logger.warning(f"User {user.user_id} has no QA pair, skipping")
            continue
            
        # Create document from conversation history
        document = Document(
            title=f"User {user.user_id} Conversations",
            content=user.get_conversation_text()
        )
        
        problem = Problem(
            id=f"wildchat_user_{user.user_id}",
            query=user.question,
            context=[document],
            target=user.answer,
            metadata={
                "user_id": user.user_id,
                "num_conversations": user.get_total_conversations()
            }
        )
        problems.append(problem)
    
    logger.info(f"Created {len(problems)} Problem objects from WildChat dataset")
    return problems

def load_wildchat_raw(
    data_path: str = "/scr-ssd/shizhehe/m07d23_wildchat/wildchat_1m_4rounds_100_userwise.csv",
    min_rounds: int = 4,
    num_users: Optional[int] = None
) -> pd.DataFrame:
    """
    Load raw wildchat conversation data from CSV file.
    
    Args:
        data_path: Path to the raw wildchat CSV file
        min_rounds: Minimum number of conversation rounds required
        num_users: Optional number of users to load
        
    Returns:
        DataFrame with wildchat conversation data
    """
    logger.info(f"Loading raw wildchat data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw wildchat data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    if num_users:
        df = df.head(num_users)
        logger.info(f"Limited to {num_users} users")
    
    logger.info(f"Loaded {len(df)} users with min {min_rounds} conversation rounds")
    return df
