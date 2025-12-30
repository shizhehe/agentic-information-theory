import os
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
from datasets import load_dataset
from pydantic import BaseModel, field_validator

from src.ib.tasks.base import Document, Problem
from src.ib.utils import get_logger

logger = get_logger("fineweb_dataset")


class FinewebQAPair(BaseModel):
    topic: str
    question: str
    answer: str
    type: str  # "qa" or "generation"


class FinewebDocument(BaseModel):
    doc_id: str
    text: str
    url: Optional[str] = None
    language: str = "en"
    language_score: Optional[float] = None
    token_count: int
    dump: Optional[str] = None  # CommonCrawl dump identifier
    date: Optional[str] = None
    qa_pairs: Optional[List[FinewebQAPair]] = None
    
    def get_text_preview(self, max_chars: int = 200) -> str:
        """Get a preview of the document text."""
        if len(self.text) <= max_chars:
            return self.text
        return self.text[:max_chars] + "..."
    
    def get_qa_pairs_by_type(self, qa_type: str) -> List[FinewebQAPair]:
        """Get QA pairs filtered by type (qa or generation)."""
        if self.qa_pairs is None:
            return []
        return [qa for qa in self.qa_pairs if qa.type == qa_type]


def load_fineweb_dataset(
    doc_ids: Optional[List[str]] = None,
    question_types: Optional[List[str]] = None,
    qa_data_path: str = "data/fineweb/qa/fineweb_questions.feather",
    raw_data_path: Optional[str] = None,
    cache_dir: str = "data/cache"
) -> List[FinewebDocument]:
    """
    Load the FineWeb dataset with automatic data location handling.
    
    This function follows the LongHealth pattern - it automatically:
    1. Checks for existing QA data at qa_data_path
    2. Falls back to raw data if QA data doesn't exist
    3. Can generate raw data from HuggingFace if needed
    
    Args:
        doc_ids: Optional list of document IDs to filter by
        question_types: Optional list of question types to filter by ("qa", "generation")
        qa_data_path: Path to QA data (feather file)
        raw_data_path: Optional path to raw CSV data
        cache_dir: Directory for HuggingFace dataset cache
        
    Returns:
        List of FinewebDocument objects
    """
    logger.info("Loading FineWeb dataset")
    
    # First, try to load from QA data if it exists
    if os.path.exists(qa_data_path):
        logger.info(f"Loading FineWeb QA data from {qa_data_path}")
        
        df = pd.read_feather(qa_data_path)
        
        # Filter by document IDs if specified
        if doc_ids is not None:
            df = df[df['doc_id'].isin(doc_ids)]
            logger.info(f"Filtered to {len(df)} QA pairs by doc_id")
        
        # Filter by question types if specified
        if question_types is not None:
            df = df[df['type'].isin(question_types)]
            logger.info(f"Filtered to {len(df)} QA pairs by question type")
        
        # Group by document
        documents = []
        for doc_id, doc_group in df.groupby('doc_id'):
            if len(doc_group) == 0:
                continue
                
            # Get document info from first row
            first_row = doc_group.iloc[0]
            
            # Collect all QA pairs for this document
            qa_pairs = []
            for _, qa_row in doc_group.iterrows():
                qa_pairs.append(FinewebQAPair(
                    topic=qa_row["topic"],
                    question=qa_row["question"],
                    answer=qa_row["answer"],
                    type=qa_row["type"]
                ))
            
            documents.append(FinewebDocument(
                doc_id=doc_id,
                text=first_row["text"],
                token_count=len(first_row["text"].split()),  # Approximate
                qa_pairs=qa_pairs
            ))
        
        logger.info(f"Loaded {len(documents)} documents with QA pairs")
        return documents
    
    # If QA data doesn't exist, try to load from raw data
    if raw_data_path and os.path.exists(raw_data_path):
        logger.info(f"QA data not found, loading raw data from {raw_data_path}")
        
        df = pd.read_csv(raw_data_path)
        
        # Filter by document IDs if specified
        if doc_ids is not None:
            df = df[df['id'].isin(doc_ids)]
            logger.info(f"Filtered to {len(df)} documents")
        
        documents = []
        for _, row in df.iterrows():
            documents.append(FinewebDocument(
                doc_id=row["id"],
                text=row["text"],
                url=row.get("url"),
                language=row.get("language", "en"),
                language_score=row.get("language_score"),
                token_count=row["token_count"],
                dump=row.get("dump"),
                date=row.get("date")
            ))
        
        logger.info(f"Loaded {len(documents)} documents from raw data (no QA pairs)")
        return documents
    
    # If no existing data found, provide helpful error message
    raise FileNotFoundError(
        f"No FineWeb data found. Please ensure one of the following exists:\n"
        f"  - QA data: {qa_data_path}\n"
        f"  - Raw data: {raw_data_path if raw_data_path else 'Not specified'}\n\n"
        f"To generate data, use:\n"
        f"  - generate_fineweb_raw_data() for raw document data\n"
        f"  - FineWeb questions.py script for QA generation"
    )


def generate_fineweb_raw_data(
    output_path: str = "data/fineweb/raw",
    num_samples: int = 100,
    min_token_count: int = 15000,
    max_token_count: int = 28000,
    cache_dir: str = "data/cache"
) -> pd.DataFrame:
    """
    Generate raw fineweb data from HuggingFace FineWeb dataset.
    
    Args:
        output_path: Directory to save the generated data
        num_samples: Number of document samples to collect
        min_token_count: Minimum token count for documents
        max_token_count: Maximum token count for documents
        cache_dir: Directory for HuggingFace dataset cache
        
    Returns:
        DataFrame with fineweb document data
    """
    logger.info(f"Loading FineWeb dataset from HuggingFace")
    
    # Load the FineWeb dataset
    fineweb_dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        split="train", 
        streaming=True, 
        cache_dir=cache_dir
    )
    
    samples = []
    
    logger.info(f"Collecting {num_samples} samples with {min_token_count}-{max_token_count} tokens")
    for sample in fineweb_dataset:
        if len(samples) >= num_samples:
            break
            
        # Filter by language (English only)
        if sample["language"] != "en":
            continue
            
        # Filter by token count range
        if (sample["token_count"] >= min_token_count and 
            sample["token_count"] <= max_token_count):
            samples.append(sample)
    
    logger.info(f"Collected {len(samples)} document samples")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save raw document dataset
    dataset_df = pd.DataFrame(samples)
    dataset_path = os.path.join(
        output_path, 
        f"fineweb_{min_token_count//1000}k_{max_token_count//1000}k_{len(dataset_df)}.csv"
    )
    dataset_df.to_csv(dataset_path, index=False)
    
    logger.info(f"Saved fineweb dataset to {dataset_path}")
    
    return dataset_df


def load_fineweb_questions(
    data_path: str = "data/fineweb/qa/fineweb_questions.feather",
    doc_ids: Optional[List[str]] = None,
    question_types: Optional[List[str]] = None
) -> List[Problem]:
    """
    Load fineweb QA dataset from feather file and convert to Problem objects.
    
    Args:
        data_path: Path to the feather file containing QA data
        doc_ids: Optional list of document IDs to filter by
        question_types: Optional list of question types to filter by
        
    Returns:
        List of Problem objects
    """
    # Load the dataset using the main loader function
    documents = load_fineweb_dataset(
        doc_ids=doc_ids,
        question_types=question_types,
        qa_data_path=data_path
    )
    
    # Convert to Problem objects
    problems = []
    for doc in documents:
        if doc.qa_pairs is None:
            logger.warning(f"Document {doc.doc_id} has no QA pairs, skipping")
            continue
            
        # Create a Problem for each QA pair
        for i, qa_pair in enumerate(doc.qa_pairs):
            document = Document(
                title=f"Document {doc.doc_id}",
                content=doc.text
            )
            
            problem = Problem(
                id=f"{doc.doc_id}_{i}",
                query=qa_pair.question,
                context=[document],
                target=qa_pair.answer,
                metadata={
                    "doc_id": doc.doc_id,
                    "topic": qa_pair.topic,
                    "type": qa_pair.type,
                    "token_count": doc.token_count
                }
            )
            problems.append(problem)
    
    logger.info(f"Created {len(problems)} Problem objects from FineWeb dataset")
    return problems

def load_fineweb_raw(
    data_path: str = "data/fineweb/raw/fineweb_15k_28k_1.csv",
    num_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Load raw fineweb data from CSV file.
    
    Args:
        data_path: Path to the raw fineweb CSV file
        num_samples: Optional number of documents to load
        
    Returns:
        DataFrame with fineweb documents
    """
    logger.info(f"Loading raw fineweb data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw fineweb data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    if num_samples:
        df = df.head(num_samples)
        logger.info(f"Limited to {num_samples} documents")
    
    logger.info(f"Loaded {len(df)} raw fineweb documents")
    return df

if __name__ == "__main__":
    print("Testing FineWeb dataset loading...")
    
    # Test the new load_dataset function
    try:
        # Load QA data if available (both factual and creative questions)
        documents = load_fineweb_dataset(
            qa_data_path="data/fineweb/qa/fineweb_questions.feather",
            question_types=["qa", "generation"],
            doc_ids=None  # Load all documents
        )
        print(f"Loaded {len(documents)} documents with QA pairs")
        
        # Show sample document
        if documents:
            doc = documents[0]
            print(f"Sample Document {doc.doc_id}:")
            print(f"  - Token count: {doc.token_count}")
            print(f"  - QA pairs: {len(doc.qa_pairs) if doc.qa_pairs else 0}")
            print(f"  - Factual QA: {len(doc.get_qa_pairs_by_type('qa'))}")
            print(f"  - Creative tasks: {len(doc.get_qa_pairs_by_type('generation'))}")
            print(f"  - Text preview: {doc.get_text_preview(100)}")
        
        # Convert to Problem objects for different question types
        factual_problems = load_fineweb_questions(
            "data/fineweb/qa/fineweb_questions.feather",
            question_types=["qa"]
        )
        print(f"Created {len(factual_problems)} factual Problem objects")
        
        creative_problems = load_fineweb_questions(
            "data/fineweb/qa/fineweb_questions.feather",
            question_types=["generation"]
        )
        print(f"Created {len(creative_problems)} creative Problem objects")
        
        if factual_problems:
            problem = factual_problems[0]
            print(f"Sample Factual Problem:")
            print(f"  - ID: {problem.id}")
            print(f"  - Query: {problem.query[:50]}...")
            print(f"  - Target: {problem.target[:50]}...")
            print(f"  - Metadata: {problem.metadata}")
            
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("\nTo generate FineWeb data:")
        print("  1. Use generate_fineweb_raw_data() to create raw document data")
        print("  2. Run the FineWeb questions.py script to generate QA pairs")
    
    print("\nFineWeb dataset testing completed!")