from typing import Dict, List, Optional, Union
import json
import datasets
import random
import re
import os

from src.ib.tasks.base import Document, Problem, BaseTask
from src.ib.tasks.longhealth.dataset import load_longhealth_dataset
from src.ib.utils import get_logger
from src.ib.clients.openai import OpenAIClient


class LongHealthTask(BaseTask):
    name: str = "LongHealth"

    class Config(BaseTask.Config):
        patient_ids: Optional[List[str]] = None
        problem_ids: Optional[List[str]] = None
        shuffle: bool = False
        use_llm_judge: bool = False
        limit: Optional[int] = None
        include_answer_choices_in_query: bool = True

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        self.evaluator_client = None

        # Initialize OpenAI client if LLM judge is enabled
        if self.config.use_llm_judge:
            try:
                self.evaluator_client = OpenAIClient(
                    config=OpenAIClient.Config(
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        model_name="gpt-4o-mini-2024-07-18",
                    )
                )
                self.logger.info("Initialized OpenAI client for GPT-4o scoring")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
                self.logger.warning(
                    "Falling back to simple letter matching for scoring"
                )

    def build_dataset(self) -> List[Problem]:
        # Load the LongHealth dataset
        patients = load_longhealth_dataset(
            patient_ids=self.config.patient_ids, problem_ids=self.config.problem_ids
        )
        self.logger.info(f"Loaded {len(patients)} patients")

        data = []
        for patient in patients:
            # Create documents from patient texts
            documents = [
                Document(title=text_id, content=text_content)
                for text_id, text_content in patient.texts.items()
            ]

            # Create a problem for each question
            for question in patient.questions:
                # Format the multiple choice options
                options = {
                    "A": question.answer_a,
                    "B": question.answer_b,
                    "C": question.answer_c,
                    "D": question.answer_d,
                    "E": question.answer_e,
                }

                # Format the question with options
                if self.config.include_answer_choices_in_query:
                    formatted_question = f"{question.question}\n\nChoose the correct answer from the following options:\n"
                    for key, value in options.items():
                        formatted_question += f"{key}. {value}\n"
                else:
                    formatted_question = question.question

                # Create metadata without answer_location
                metadata = {
                    "patient_id": patient.patient_id,
                    "patient_name": patient.name,
                    "patient_birthday": patient.birthday,
                    "patient_diagnosis": patient.diagnosis,
                    "options": options,
                }

                # Only add answer_location if needed for internal processing, not for serialization
                # Commenting out this line to avoid serialization issues
                # metadata["answer_location"] = question.answer_location

                data.append(
                    Problem(
                        id=question.question_id,
                        query=formatted_question,
                        context=documents,
                        target=question.correct,
                        task="LongHealth",
                        metadata=metadata,
                    )
                )

        if self.config.shuffle:
            # set random seed

            random.seed(42)
            random.shuffle(data)

        # Apply limit directly here if specified
        if hasattr(self.config, "limit") and self.config.limit is not None:
            data = data[: self.config.limit]

        self.logger.info(f"Built dataset with {len(data)} problems")
        return data

    def score(
        self,
        pred: Union[str, list[str]],
        item: Problem,
        default_critical: bool = False,
    ) -> Union[float, int]:
        """
        Use GPT-4o to evaluate if the predicted answer matches the ground truth if use_llm_judge is True.
        Falls back to simple letter matching otherwise.

        If default_critical is True, kill the process if the the judge fails.
        """
        if isinstance(pred, str):
            preds = [pred]
            single = True
        else:
            preds = pred
            single = False

        if not self.config.use_llm_judge or not self.evaluator_client:
            print(
                f"Using simple matching, {self.config.use_llm_judge}, {self.evaluator_client}"
            )
            # Simple fallback if LLM judge is not enabled or client initialization failed
            sanitized_pred = pred.strip().upper().strip(".")
            sanitized_target = item.target.strip().upper().strip(".")

            # Direct match with the correct letter
            if sanitized_target in sanitized_pred:
                return 1
            return 0

        # Use GPT-4o as judge
        options = item.metadata.get("options", {})
        correct_letter = item.target.strip().upper()
        correct_answer = options.get(correct_letter, "")

        prompt = """
        You are an expert evaluator. Your task is to determine if a predicted answer matches the correct answer for a multiple-choice question.
        
        Question: {query}
        
        Predicted Answer: {pred}
        
        Correct Answer: {correct_letter}. {correct_answer}
        
        Is the predicted answer correct? 
        
        Respond with a JSON object with the following fields:
        - "explanation": brief explanation of your reasoning
        - "is_correct": true or false
        """

        prompts = []
        for pred in preds:
            prompts.append(
                prompt.format(
                    query=item.query,
                    pred=pred,
                    correct_letter=correct_letter,
                    correct_answer=correct_answer,
                )
            )

        try:
            print("Trying to judge with ", self.evaluator_client.config.model_name)
            response = self.evaluator_client.chat(
                chats=[[{"role": "user", "content": prompt}] for prompt in prompts],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            print("Done judging with ", self.evaluator_client.config.model_name)

            results = [
                json.loads(response.samples[i].text)
                for i in range(len(response.samples))
            ]
            cost = response.usage.cost
            print(
                f"Judging {len(preds)} predictions using {self.evaluator_client.config.model_name}, cost: {cost}"
            )
            if single:
                return 1 if results[0].get("is_correct", False) else 0
            else:
                return [
                    1 if result.get("is_correct", False) else 0 for result in results
                ]
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            print("Fallback to simple matching")
            if default_critical:
                raise Exception(
                    "Error during evaluation, default_critical is True, exiting"
                )
            # Fall back to simple matching in case of error
            if single:
                sanitized_pred = pred.strip().upper()
                sanitized_target = item.target.strip().upper()
                return 1 if sanitized_target in sanitized_pred else 0
            else:
                sanitized_preds = [pred.strip().upper() for pred in preds]
                sanitized_target = item.target.strip().upper()
                return [
                    1 if sanitized_target in sanitized_pred else 0
                    for sanitized_pred in sanitized_preds
                ]
