import random
from typing import Any, Dict, List, Optional, Tuple
import json
from pydantic import BaseModel, field_validator
import requests
import pandas as pd


class LongHealthAnswerLocation(BaseModel):
    start: List[float]
    end: List[float]
    
    def __str__(self):
        """String representation for serialization."""
        return json.dumps({"start": self.start, "end": self.end})
    
    @classmethod
    def from_str(cls, s):
        """Create instance from serialized string."""
        data = json.loads(s)
        return cls(start=data["start"], end=data["end"])
        
    # Add field validator to ensure the object is serializable
    @field_validator('start', 'end')
    def validate_coordinates(cls, v):
        # Ensure all values are basic Python types
        return [float(x) for x in v]


# dict_keys(['No', 'question', 'answer_a', 'answer_b', 'answer_c', 'answer_d', 'answer_e', 'correct', 'answer_location'])
class LongHealthQuestion(BaseModel):
    question_id: str
    question: str
    correct: str

    answer_a: str
    answer_b: str
    answer_c: str
    answer_d: str
    answer_e: str

    # text_id -> answer_location in that text
    answer_location: Optional[Dict[str, LongHealthAnswerLocation]]

class LongHealthPatient(BaseModel):
    patient_id: str
    texts: Dict[str, str]
    name: str 
    birthday: str
    diagnosis: str
    questions: List[LongHealthQuestion]


    
DATASET_PATH = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"

def load_longhealth_dataset(patient_ids: Optional[List[str]] = None, problem_ids: Optional[List[str]] = None) -> List[LongHealthPatient]:
    """
    Load the LongHealth dataset from GitHub URL.
    
    Args:
        patient_ids: Optional list of patient IDs to filter by.
        problem_ids: Optional list of problem IDs to filter by.
        
    Returns:
        List of LongHealthPatient objects.
    """
    
    # Load the dataset from GitHub URL
    response = requests.get(DATASET_PATH)
    response.raise_for_status()  # Raise an exception for HTTP errors
    data = json.loads(response.text)
    
    # The data is now a dictionary with patient_ids as keys
    patients = []
    
    for patient_id, patient_data in data.items():
        # Skip if we're filtering by patient ID and this one isn't in the list
        if patient_ids and patient_id not in patient_ids:
            continue
            
        # Process questions
        questions = []
        for question in patient_data["questions"]:
            # Create question_id from patient_id and question number
            question_id = f"{patient_id}_{question['No']}"
            
            # Skip if we're filtering by problem ID and this one isn't in the list
            if problem_ids and question_id not in problem_ids:
                continue
                
            # Process answer location if it exists
            answer_location = None
            if "answer_location" in question and question["answer_location"]:
                answer_location = {
                    text_id: LongHealthAnswerLocation(**loc_data)
                    for text_id, loc_data in question["answer_location"].items()
                }
                
            questions.append(
                LongHealthQuestion(
                    question_id=question_id,
                    question=question["question"],
                    answer_a=question["answer_a"],
                    answer_b=question["answer_b"],
                    answer_c=question["answer_c"],
                    answer_d=question["answer_d"],
                    answer_e=question["answer_e"],
                    correct=question["correct"],
                    answer_location=answer_location
                )
            )
        
        # Skip creating patient if no questions match our filters
        if problem_ids and not questions:
            continue
            
        # Create patient object
        patients.append(
            LongHealthPatient(
                patient_id=patient_id,
                name=patient_data["name"],
                birthday=patient_data["birthday"],
                diagnosis=patient_data["diagnosis"],
                texts=patient_data["texts"],
                questions=questions
            )
        )
    
    return patients


if __name__ == "__main__":
    dataset = load_longhealth_dataset()

    # print some info about the dataset
    print(f"Number of patients: {len(dataset)}")
    # print num questions per patient (print all unique values)
    print(f"Number of questions per patient: {sorted(list(set(len(patient.questions) for patient in dataset)))}")
    # print num texts per patient
    print(f"Number of texts per patient: {sorted(list(set(len(patient.texts) for patient in dataset)))}")
    
    

    # write dataset locally to parquet
    df = pd.DataFrame([patient.model_dump() for patient in dataset])
    df.to_parquet("longhealth_dataset.parquet")
