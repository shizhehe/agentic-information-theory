from typing import List, Optional, Union
import json
import os
import random
import re

from mfactory.tasks.base import Document, Problem, BaseTask
from mfactory.tasks.qasper.dataset import load_qasper_dataset
from mfactory.utils import get_logger
from mfactory.clients.openai import OpenAIClient
from mfactory.clients.fireworks import FireworksClient


class QasperTask(BaseTask):
    name: str = "Qasper"

    class Config(BaseTask.Config):
        split: str = "train"
        use_llm_judge: bool = False
        shuffle: bool = False
        problem_ids: Optional[List[str]] = None
        limit: Optional[int] = None

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        self.openai_client = None
        self.fireworks_client = None

        # Initialize both clients if LLM judge is enabled
        # They will be used as fallbacks in the score method
        if self.config.use_llm_judge:
            # Try to initialize OpenAI client
            try:
                self.openai_client = OpenAIClient(
                    config=OpenAIClient.Config(
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        model_name="gpt-4o-mini-2024-07-18",
                    )
                )
                self.logger.info("Initialized OpenAI client for GPT-4o scoring")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI client: {e}")

            # Try to initialize FireworksClient
            try:
                self.fireworks_client = FireworksClient(
                    config=FireworksClient.Config(
                        api_key=os.environ.get("FIREWORKS_API_KEY"),
                        model_name="accounts/fireworks/models/llama-v3p1-405b-instruct",
                    )
                )
                self.logger.info("Initialized FireworksClient for scoring")
            except Exception as e2:
                self.logger.warning(f"Failed to initialize FireworksClient: {e2}")

            if not self.openai_client and not self.fireworks_client:
                self.logger.warning(
                    "Neither OpenAI nor Fireworks clients initialized. Will use simple string matching for scoring"
                )

    def build_dataset(self) -> List[Problem]:
        # Load the QASPER dataset
        dataset = load_qasper_dataset(split=self.config.split)
        self.logger.info(f"Loaded {len(dataset)} examples from QASPER {self.config.split} split")

        data = []
        for entry in dataset:
            # Create document from context
            documents = [
                Document(
                    title=entry["title"],
                    content=entry["context"]
                )
            ]

            # Create metadata
            metadata = {
                "doc_id": entry["doc_id"],
                "qa_id": entry["qa_id"],
                "evidence": entry.get("evidence"),
                "highlighted_evidence": entry.get("highlighted_evidence"),
            }

            data.append(
                Problem(
                    id=entry["qa_id"],
                    query=entry["question"],
                    context=documents,
                    target=entry["answer"],
                    task="Qasper",
                    metadata=metadata,
                )
            )

        if self.config.shuffle:
            random.seed(42)
            random.shuffle(data)

        self.logger.info(f"Built dataset with {len(data)} problems")
        return self._subset_dataset(data)

    def _sanitize(self, answer: str) -> str:
        """Sanitize answer for comparison."""
        return answer.strip().lower()

    def score(
        self,
        pred: Union[str, list[str]],
        item: Problem,
        default_critical: bool = False,
    ) -> Union[float, int]:
        """
        Score the prediction against the ground truth answer.
        Uses LLM judge if enabled, otherwise falls back to simple string matching.
        """
        if isinstance(pred, str):
            preds = [pred]
            single = True
        else:
            preds = pred
            single = False

        if not self.config.use_llm_judge:
            # Simple fallback if LLM judge is not enabled
            if single:
                return int(self._sanitize(pred) == self._sanitize(item.target))
            else:
                return [
                    int(self._sanitize(p) == self._sanitize(item.target))
                    for p in preds
                ]

        # Prepare prompt for LLM judge
        prompt = """
        You are an expert evaluator. Your task is to determine if a predicted answer matches the correct answer for a question-answering task.
        
        Question: {query}
        
        Predicted Answer: {pred}
        
        Correct Answer: {target}
        
        Is the predicted answer correct? Consider that answers may be phrased differently but convey the same information.
        
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
                    target=item.target,
                )
            )

        # Try OpenAI first, then Fireworks, then simple matching
        # Try OpenAI client
        # if self.openai_client:
        #     try:
        #         self.logger.info(f"Judging with {self.openai_client.config.model_name}")
        #         response = self.openai_client.chat(
        #             chats=[[{"role": "user", "content": prompt}] for prompt in prompts],
        #             response_format={"type": "json_object"},
        #             temperature=0.0,
        #         )

        #         results = [
        #             json.loads(response.samples[i].text)
        #             for i in range(len(response.samples))
        #         ]
        #         cost = response.usage.cost
        #         self.logger.info(
        #             f"Judging {len(preds)} predictions using {self.openai_client.config.model_name}, cost: {cost}"
        #         )
                
        #         if single:
        #             return 1 if results[0].get("is_correct", False) else 0
        #         else:
        #             return [
        #                 1 if result.get("is_correct", False) else 0 for result in results
        #             ]
        #     except Exception as e:
        #         self.logger.warning(f"OpenAI evaluation failed: {e}")
        #         self.logger.info("Falling back to FireworksClient")

        # Try Fireworks client
        if self.fireworks_client:
            try:
                self.logger.info(f"Judging with {self.fireworks_client.config.model_name}")
                # Note: Fireworks may not support response_format, so we'll try without it first
                response = self.fireworks_client.chat(
                    chats=[[{"role": "user", "content": prompt}] for prompt in prompts],
                    temperature=0.0,
                )

                results = []
                for i in range(len(response.samples)):
                    try:
                        results.append(json.loads(response.samples[i].text))
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try to extract JSON from the text
                        text = response.samples[i].text
                        # Try to find JSON object in the response
                        json_match = re.search(r'\{[^{}]*"is_correct"[^{}]*\}', text)
                        if json_match:
                            results.append(json.loads(json_match.group()))
                        else:
                            # Fallback: check if text contains "true" or "false" for is_correct
                            is_correct = "true" in text.lower() and "false" not in text.lower()
                            results.append({"is_correct": is_correct})

                cost = response.usage.cost if hasattr(response.usage, 'cost') else 0
                self.logger.info(
                    f"Judging {len(preds)} predictions using {self.fireworks_client.config.model_name}, cost: {cost}"
                )
                
                if single:
                    return 1 if results[0].get("is_correct", False) else 0
                else:
                    return [
                        1 if result.get("is_correct", False) else 0 for result in results
                    ]
            except Exception as e:
                self.logger.warning(f"Fireworks evaluation failed: {e}")
                self.logger.info("Falling back to simple string matching")

        # Final fallback to simple matching
        self.logger.warning("All LLM judges failed, using simple string matching")
        if default_critical:
            raise Exception(
                "Error during evaluation, default_critical is True, exiting"
            )
        if single:
            return int(self._sanitize(pred) == self._sanitize(item.target))
        else:
            return [
                int(self._sanitize(p) == self._sanitize(item.target))
                for p in preds
            ]
