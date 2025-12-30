from typing import List, Optional
from datasets import load_dataset


def list_to_string(array: List[str]) -> str:
    """Convert a list of strings to a single string."""
    return " ".join(array)


def get_context(row: dict) -> str:
    """
    Extract and format the context from a QASPER dataset row.
    
    Args:
        row: A row from the QASPER dataset
        
    Returns:
        Formatted context string with sections and figures/tables
    """
    section_names = row["full_text"]["section_name"]
    paragraphs = row["full_text"]["paragraphs"]

    formatted_sections = []
    for section_name, para_list in zip(section_names, paragraphs):
        section_text = list_to_string(para_list)
        formatted_sections.append(f"### {section_name}\n\n{section_text}")

    figures_text = row["figures_and_tables"]["caption"]

    return list_to_string(formatted_sections + figures_text)


def load_qasper_dataset(split: str = "train"):
    """
    Load the QASPER dataset from HuggingFace.
    
    Args:
        split: Dataset split to load (e.g., "train", "validation", "test")
        
    Returns:
        List of processed QASPER entries with context, questions, and answers
    """
    dataset = load_dataset("allenai/qasper", split=split)
    
    processed_dataset = []
    
    for entry in dataset:
        context = get_context(entry)
        title = entry['title']
        doc_id = entry["id"]
        questions = entry['qas']['question']
        question_ids = entry['qas']['question_id']
        all_answers = entry['qas']['answers']
        
        for q, qa_id, a in zip(questions, question_ids, all_answers):
            answer = None
            evidence = None
            highlighted_evidence = None
            
            for single_answer in a['answer']:
                # Skip unanswerable questions
                if single_answer['unanswerable']:
                    continue
                
                # Extract answer from extractive spans
                if single_answer['extractive_spans']:
                    answer = ", ".join(single_answer['extractive_spans'])
                    evidence = ", ".join(single_answer['evidence'])
                    highlighted_evidence = ", ".join(single_answer['highlighted_evidence'])
                    break  # Use the first valid answer
            
            # Only add entries with valid answers
            if answer:
                processed_dataset.append({
                    "doc_id": doc_id,
                    "title": title,
                    "context": context,
                    "qa_id": qa_id,
                    "question": q,
                    "answer": answer,
                    "evidence": evidence,
                    "highlighted_evidence": highlighted_evidence
                })
    
    return processed_dataset


if __name__ == "__main__":
    # Test loading the dataset
    train_data = load_qasper_dataset(split="train")
    print(f"Loaded {len(train_data)} examples from QASPER train split")
    
    if train_data:
        print("\nExample entry:")
        print(f"Doc ID: {train_data[0]['doc_id']}")
        print(f"Title: {train_data[0]['title']}")
        print(f"Question: {train_data[0]['question']}")
        print(f"Answer: {train_data[0]['answer']}")
        print(f"Context length: {len(train_data[0]['context'])} characters")
