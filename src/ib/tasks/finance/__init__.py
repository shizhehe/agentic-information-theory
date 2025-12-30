from typing import List, Union

import pandas as pd
import json
import os
import random

import fireworks.client

from mfactory.tasks.base import Document, Problem, BaseTask
from mfactory.tasks.datasets.finance import load_finance
from mfactory.utils import get_logger
from mfactory.clients.openai import OpenAIClient

fireworks.client.api_key = os.getenv("FIREWORKS_API_KEY")


class FinanceTask(BaseTask):
    name: str = "Finance"

    class Config(BaseTask.Config):
        use_llm_judge: bool = True
        evidence_only: bool = False

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        self.evaluator_client = None

        # Initialize OpenAI client if LLM judge is enabled
        if self.config.use_llm_judge:
            try:
                import openai

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
        df = load_finance(
            tokenizer_model=self.config.tokenizer_model,
            max_context_length=self.config.max_context_tokens,
            evidence_only=self.config.evidence_only,
        )

        data = []

        for i, row in enumerate(df.iterrows()):
            if isinstance(row, tuple):  # When using iterrows()
                _, row = row

            row_dict = row.to_dict() if not isinstance(row, dict) else row

            data.append(
                Problem(
                    id=str(i),
                    query=row_dict.get("question", ""),
                    context=[
                        Document(
                            title="Financial Document",
                            content=row_dict.get("pdf_text", ""),
                        )
                    ],
                    target=row_dict.get("answer", ""),
                    task="Finance",
                )
            )
        if self.config.shuffle:
            # set random seed

            random.seed(42)
            random.shuffle(data)
        return self._subset_dataset(data)

    def _sanitize(self, answer: str) -> str:
        return answer.strip().lower()

    def score(
        self,
        pred: Union[str, list[str]],
        item: Problem,
        default_critical: bool = False,
    ) -> Union[float, int]:
        """
        Use Fireworks to evaluate if the predicted answers are correct.
        """
        if isinstance(pred, str):
            preds = [pred]
        else:
            preds = pred

        if not self.config.use_llm_judge:
            # Simple fallback if LLM judge is not enabled or client initialization failed
            if len(preds) == 1:
                return int(self._sanitize(pred) == self._sanitize(item.target))
            else:
                return [
                    int(self._sanitize(pred) == self._sanitize(item.target))
                    for pred in preds
                ]

        # Use Llama 70B as judge
        prompt = """
        You are an expert evaluator. Your task is to determine if a predicted answer matches the correct answer for a QA question.
        
        
        A predicted answer can have three evaluation outcomes. First, correct answer. This is the 'desired' behavior of models. 
        To ensure a good-faith understanding of models' capabilities we allow minor deviations, such as giving the answer in 
        billions when the unit was given in the question as millions. We also allow very small rounding errors. 
        
        Second, incorrect answer. Incorrect answers vary, from calculations that are off by small margins to several orders 
        of magnitude, and from making up legal information to giving the wrong direction for an effect (e.g. reporting negative
        growth when it is actually positive). If a model gives the right answer but with logic or calculations that explicitly 
        contradict the evidence in the gold standard answer, we label it Incorrect. 
        
        Third, failure to answer. If the model explicitly states that it cannot answer because it does not have access to the 
        right information then it is a failure to answer (e.g. \"As an AI, I don't have real-time data access capabilities to 
        provide information on Boeing's production rate forecast for FY2023.\").
        
        Question: {query}
        
        Predicted Answer: {pred}
        
        Correct Answer: {target}.
        
        Is the predicted answer correct, incorrect, or failure to answer? 
        
        Respond with a JSON object with the following fields:
        - "explanation": brief explanation of your reasoning
        - "is_correct": true or false
        - "failure_to_answer": true or false
        """

        prompts = []
        for pred in preds:
            prompts.append(
                prompt.format(
                    query=item.query,
                    pred=pred,
                    target=item.target,
                )
            )

        try:
            response = self.evaluator_client.chat(
                chats=[[{"role": "user", "content": prompt}] for prompt in prompts],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            results = [
                json.loads(response.samples[i].text)
                for i in range(len(response.samples))
            ]
            cost = response.usage.cost
            print(
                f"Judging {len(preds)} predictions using {self.evaluator_client.config.model_name}, cost: {cost}"
            )
            if len(preds) == 1:
                return (
                    1
                    if result.get("is_correct", False)
                    and not result.get("failure_to_answer", False)
                    else 0
                )
            else:
                return [
                    (
                        1
                        if result.get("is_correct", False)
                        and not result.get("failure_to_answer", False)
                        else 0
                    )
                    for result in results
                ]
        except Exception as e:
            print(f"Error during evaluation: {e}")
            print("Fallback to simple matching")
            if default_critical:
                raise Exception(
                    "Error during evaluation, default_critical is True, exiting"
                )
            # Fall back to simple matching in case of error
            if len(preds) == 1:
                return int(self._sanitize(pred) == self._sanitize(item.target))
            else:
                return [
                    int(self._sanitize(pred) == self._sanitize(item.target))
                    for pred in preds
                ]
