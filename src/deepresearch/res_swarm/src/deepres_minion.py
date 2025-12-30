"""
ResSwarm Minion Protocol

Adapted from minions/minions/minion.py for DeepResearch Bench integration.
Provider-agnostic supervisor-worker architecture with web search capabilities.
Part of ResSwarm - Research Swarm Intelligence.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import json
import re
import os
import time
import concurrent.futures
from datetime import datetime
from pydantic import BaseModel

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

from src.ib.clients.usage import Usage
from src.deepresearch.clients.base import LMClient

# Import advanced prompts
from src.deepresearch.res_swarm.src.prompts.deepres_prompts import (
    INITIAL_QUERY_PROMPT,
    WORKER_SUMMARIZE_PROMPT, 
    WORKER_MULTI_QUERY_EXTRACT_PROMPT,
    ASSESSMENT_PROMPT,
    FINAL_SYNTHESIS_PROMPT,
    SUPERVISOR_DECOMPOSITION_PROMPT,
    WORKER_RESPONSE_PROMPT,
    SUPERVISOR_SYNTHESIS_PROMPT,
    MULTI_QUERY_GENERATION_PROMPT,
    SYNTHESIS_PLANNING_PROMPT,
    CHUNKED_SYNTHESIS_PROMPT
)


def _escape_newlines_in_strings(json_str: str) -> str:
    """Escape newlines in JSON strings for better parsing."""
    return re.sub(
        r'(".*?")',
        lambda m: m.group(1).replace("\n", "\\n"),
        json_str,
        flags=re.DOTALL,
    )


def _extract_balanced_json(text: str) -> str:
    """Extract the first complete JSON object using balanced brace counting."""
    text = text.strip()
    
    # Find the first opening brace
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    # Count balanced braces to find the complete JSON object
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the complete JSON object
                    return text[start_idx:i+1]
    
    return None


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from text that may be wrapped in markdown code blocks."""
    
    # Handle markdown formatted responses
    if text.strip().startswith("###"):
        # Extract content after markdown header
        lines = text.strip().split('\n')
        header = lines[0].strip('#').strip().lower()
        content = '\n'.join(lines[1:]).strip()
        
        # Return as explanation with relevant answer
        return {
            "explanation": content,
            "answer": "relevant"
        }
    
    # First try: if the text is already clean JSON, parse it directly
    text_stripped = text.strip()
    if text_stripped.startswith('{') and text_stripped.endswith('}'):
        try:
            # Try parsing directly first (for well-formed JSON)
            result = json.loads(text_stripped)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            try:
                # Escape newlines only within quoted JSON strings as fallback
                cleaned_text = _escape_newlines_in_strings(text_stripped)
                result = json.loads(cleaned_text)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError as e:
                # Log the specific error for debugging
                print(f"[DEBUG] JSON parsing failed: {e}")
                pass  # Fall through to extraction strategies
    
    # Try multiple extraction strategies for more complex text
    strategies = [
        # 1. Markdown code blocks
        lambda t: re.finditer(r"```(?:json)?\s*(.*?)```", t, re.DOTALL),
        # 2. Find the outermost complete JSON object using balanced braces
        lambda t: _extract_balanced_json(t),
        # 3. Simple curly braces (last resort, may fail with nested objects)
        lambda t: re.finditer(r"\{.*?\}", t, re.DOTALL),
    ]
    
    json_str = None
    for strategy in strategies:
        if callable(strategy):
            try:
                if strategy == _extract_balanced_json:
                    # Special handling for balanced JSON extraction
                    json_str = strategy(text)
                    if json_str:
                        break
                else:
                    matches = list(strategy(text))
                    if matches:
                        if hasattr(matches[0], 'group'):
                            # For regex matches
                            if matches[0].groups():
                                json_str = matches[-1].group(1).strip()  # Code block content
                            else:
                                json_str = matches[-1].group(0)  # Full match
                        break
            except Exception:
                continue
    
    # Fallback: use the whole text
    if json_str is None:
        json_str = text.strip()
    
    # Escape newlines only within quoted JSON strings
    json_str = _escape_newlines_in_strings(json_str)
    
    try:
        result = json.loads(json_str)
        # Ensure we return a dict
        if isinstance(result, dict):
            return result
        else:
            # If it's a string or other type, wrap it in a dict
            return {"explanation": str(result), "answer": "relevant"}
    except json.JSONDecodeError as e:
        print(f"[DEBUG] Failed to parse JSON: {json_str[:200]}...")
        print(f"[DEBUG] JSONDecodeError: {e}")
        # Return the raw text as explanation
        return {"explanation": json_str, "answer": "relevant"}


def chunk_by_section(doc: str, max_chunk_size: int = 3000, overlap: int = 50) -> List[str]:
    """
    Chunk document by sections with overlap for context preservation.
    
    Args:
        doc: Document to chunk
        max_chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of document chunks with preserved context
    """
    if not doc or len(doc) <= max_chunk_size:
        return [doc] if doc else []
    
    chunks = []
    start = 0
    while start < len(doc):
        end = start + max_chunk_size
        chunk = doc[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)
        
        start += max_chunk_size - overlap
        
        # Avoid infinite loops
        if start >= len(doc):
            break
    
    return chunks


class JobManifest(BaseModel):
    """
    Represents a job manifest for content processing with relevance assessment.
    """
    chunk: str
    chunk_id: int
    task_id: int
    job_id: int


class JobOutput(BaseModel):
    """
    Represents the output of content relevance assessment.
    Enhanced with multi-dimensional analysis fields.
    """
    # Original fields for backward compatibility
    explanation: Optional[str] = None
    answer: str  # "relevant" or "not relevant"
    
    # Enhanced analysis fields (accept both string and list formats)
    key_findings: Optional[Union[str, List[str]]] = None
    quantitative_analysis: Optional[Union[str, List[str]]] = None
    evidence_data: Optional[Union[str, List[str]]] = None
    comparative_framework: Optional[Union[str, List[str]]] = None
    trend_analysis: Optional[Union[str, List[str]]] = None
    context_implications: Optional[Union[str, List[str]]] = None
    strategic_insights: Optional[Union[str, List[str]]] = None
    information_gaps: Optional[Union[str, List[str]]] = None
    relevance_score: Optional[float] = None


class AssessmentOutput(BaseModel):
    """
    Represents the output of research completeness assessment.
    """
    more_info_required: bool
    search_query: Optional[str] = None


class DeepResearchMinion:
    """
    Advanced DeepResearch protocol for supervisor-worker model evaluation.
    
    Features:
    - Multi-round adaptive research with quality assessment
    - Intelligent content chunking and relevance filtering
    - Batch processing for efficient API usage
    - Backward compatible single-round mode for evaluation
    - Provider-agnostic supervisor and worker model support
    - Output format compatible with DeepResearch Bench evaluation
    """
    
    def __init__(
        self,
        supervisor_client: MinionsClient,
        worker_client: MinionsClient,
        web_search_enabled: bool = True,
        log_dir: str = "outputs/logs",
        callback=None,
        max_rounds: int = 3,
        research_mode: str = "adaptive",
        max_sources_per_round: int = 20,
        worker_batch_size: int = 10,
        synthesis_strategy: str = "single",
        sections_per_chunk: int = 2,
    ):
        """
        Initialize DeepResearch Minion with supervisor and worker clients.
        
        Args:
            supervisor_client: Cloud LLM client (OpenAI, Anthropic, Fireworks, etc.)
            worker_client: Worker LLM client (SGLang Modal, Ollama, etc.)
            web_search_enabled: Whether to enable web search tools for workers
            log_dir: Directory for logging conversation history
            callback: Optional callback function for real-time updates
            max_rounds: Maximum research rounds (default 3 for adaptive mode)
            research_mode: "adaptive" for multi-round or "evaluation" for single-round
            max_sources_per_round: Maximum web sources per research round
            worker_batch_size: Batch size for worker processing
            synthesis_strategy: "single" (default) or "chunked" for sectioned synthesis
            sections_per_chunk: Number of sections per chunk for chunked synthesis
        """
        self.supervisor_client = supervisor_client
        self.worker_client = worker_client
        self.web_search_enabled = web_search_enabled
        self.log_dir = log_dir
        self.callback = callback
        self.research_mode = research_mode
        self.max_sources_per_round = max_sources_per_round
        self.synthesis_strategy = synthesis_strategy
        self.sections_per_chunk = sections_per_chunk
        if "qwen" in self.worker_client.model_name.lower():
            self.worker_batch_size = 5
        else:
            self.worker_batch_size = worker_batch_size
        
        # Set max_rounds based on research mode
        if research_mode == "evaluation":
            self.max_rounds = 1  # Single-round for benchmark compatibility
        else:
            self.max_rounds = max_rounds  # Multi-round adaptive research
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize web search tools if enabled
        if web_search_enabled:
            from .deepres_tools import WebSearchTool
            self.search_tool = WebSearchTool()
        else:
            self.search_tool = None
    
    def __call__(
        self,
        task: str,
        query_id: Optional[int] = None,
        logging_id: Optional[str] = None,
        research_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the DeepResearch protocol on a single task.
        
        Args:
            task: The research question/task to answer
            query_id: Optional ID from DeepResearch Bench query
            logging_id: Optional identifier for logging
            research_mode: Override instance research mode ("adaptive" or "evaluation")
            
        Returns:
            Dict containing final_answer, conversation log, usage stats
        """
        # Override research mode if specified
        current_mode = research_mode or self.research_mode
        
        print(f"\n========== DEEPRESEARCH TASK STARTED ==========")
        print(f"Task: {task}")
        print(f"Query ID: {query_id}")
        print(f"Research mode: {current_mode}")
        print(f"Max rounds: {1 if current_mode == 'evaluation' else self.max_rounds}")
        print(f"Web search enabled: {self.web_search_enabled}")
        
        # Initialize timing and usage tracking
        start_time = time.time()
        timing = {
            "supervisor_time": 0.0,
            "worker_time": 0.0,
            "search_time": 0.0,
            "total_time": 0.0,
        }
        
        # Track granular token usage per call instead of aggregated Usage objects
        supervisor_calls = []  # List of {"input_tokens": int, "output_tokens": int}
        worker_calls = []      # List of {"input_tokens": int, "output_tokens": int}
        
        # Initialize conversation log
        conversation_log = {
            "query_id": query_id,
            "task": task,
            "research_mode": current_mode,
            "max_rounds": 1 if current_mode == "evaluation" else self.max_rounds,
            "conversation": [],
            "final_answer": "",
            "usage": {
                "supervisor": {},
                "worker": {},
            },
            "timing": timing,
            "search_queries": [],
            "search_results": [],
            "research_rounds": [],
        }
        
        try:
            if current_mode == "evaluation":
                # Backward compatible single-round mode for evaluation
                final_answer = self._run_evaluation_mode(
                    task, conversation_log, timing, supervisor_calls, worker_calls
                )
            elif current_mode == "multi_query":
                # Multi-query mode: 5 queries with 10 sources each
                final_answer = self._run_multi_query_mode(
                    task, conversation_log, timing, supervisor_calls, worker_calls
                )
            else:
                # Advanced multi-round adaptive research mode
                final_answer = self._run_adaptive_research_mode(
                    task, conversation_log, timing, supervisor_calls, worker_calls
                )
            
            conversation_log["final_answer"] = final_answer
            
        except Exception as e:
            print(f"Error during DeepResearch protocol: {e}")
            final_answer = f"Error processing task: {str(e)}"
            conversation_log["final_answer"] = final_answer
        
        # Calculate total time
        end_time = time.time()
        timing["total_time"] = end_time - start_time
        
        # Debug: Print final token tracking
        print(f"[DEBUG] Supervisor calls: {len(supervisor_calls)} total")
        print(f"[DEBUG] Worker calls: {len(worker_calls)} total")
        
        # Calculate summary metrics
        total_supervisor_input = sum(call["input_tokens"] for call in supervisor_calls)
        total_supervisor_output = sum(call["output_tokens"] for call in supervisor_calls)
        total_worker_input = sum(call["input_tokens"] for call in worker_calls)
        total_worker_output = sum(call["output_tokens"] for call in worker_calls)
        
        print(f"[DEBUG] Supervisor total: {total_supervisor_input + total_supervisor_output} tokens")
        print(f"[DEBUG] Worker total: {total_worker_input + total_worker_output} tokens")
        
        # Add granular usage statistics to log
        conversation_log["usage"]["supervisor_calls"] = supervisor_calls
        conversation_log["usage"]["worker_calls"] = worker_calls
        conversation_log["usage"]["summary"] = {
            "supervisor_total_input": total_supervisor_input,
            "supervisor_total_output": total_supervisor_output,
            "worker_total_input": total_worker_input,
            "worker_total_output": total_worker_output,
        }
        
        # Save conversation log
        self._save_log(conversation_log, task, query_id, logging_id)
        
        print(f"\n=== DEEPRESEARCH TASK COMPLETED ===")
        
        # Create aggregated Usage objects for backward compatibility
        supervisor_usage = Usage(
            prompt_tokens=sum(call.get('input_tokens', 0) for call in supervisor_calls),
            completion_tokens=sum(call.get('output_tokens', 0) for call in supervisor_calls)
        )
        
        worker_usage = Usage(
            prompt_tokens=sum(call.get('input_tokens', 0) for call in worker_calls),
            completion_tokens=sum(call.get('output_tokens', 0) for call in worker_calls)
        )
        
        # Return format with both granular calls (new) and aggregated usage (backward compatibility)
        return {
            "final_answer": final_answer,
            "conversation_log": conversation_log,
            "supervisor_model": getattr(self.supervisor_client, 'model_name', 'unknown'),
            "worker_model": getattr(self.worker_client, 'model_name', 'unknown'),
            "supervisor_calls": supervisor_calls,
            "worker_calls": worker_calls,
            "worker_call_count": len(worker_calls),
            "supervisor_usage": supervisor_usage,
            "worker_usage": worker_usage,
            "timing": timing,
        }
    
    def _run_adaptive_research_mode(
        self,
        task: str,
        conversation_log: Dict,
        timing: Dict,
        supervisor_calls: List[Dict],
        worker_calls: List[Dict]
    ) -> str:
        """
        Run advanced multi-round adaptive research protocol.
        
        This method implements the sophisticated research approach from the original
        minions_deep_research.py with iterative quality assessment and gap filling.
        """
        print(f"🔬 ADAPTIVE RESEARCH MODE: Up to {self.max_rounds} rounds")
        
        current_round = 0
        query_results = {}
        visited_urls = set()
        
        # Step 1: Generate initial search query
        assessment = self._assess_and_generate_query(task, None, conversation_log, timing, supervisor_calls)
        
        # Step 2: Iterative research loop
        while assessment.more_info_required and current_round < self.max_rounds:
            current_round += 1
            print(f"\n🔍 RESEARCH ROUND {current_round}/{self.max_rounds}")
            
            
            if self.callback:
                self.callback("supervisor", f"🔍 Round {current_round}: {assessment.search_query}")
            
            # Extract metadata from web search
            search_urls, metadata = self._extract_metadata(assessment.search_query, conversation_log, timing)
            visited_urls.update(search_urls)
            
            # Summarize and filter metadata for relevance
            summaries = self._summarize_metadata(assessment.search_query, metadata, conversation_log, timing, worker_calls)
            
            
            if self.callback and summaries:
                preview_text = f"📚 Round {current_round}: Found {len(summaries)} relevant sources"
                self.callback("worker", preview_text)
            
            # Store results for this query
            query_results[assessment.search_query] = summaries
            
            # Log this research round
            conversation_log["research_rounds"].append({
                "round": current_round,
                "search_query": assessment.search_query,
                "sources_found": len(search_urls),
                "relevant_summaries": len(summaries),
                "urls": list(search_urls)
            })
            
            # Assess if we need more information
            assessment = self._assess_and_generate_query(task, query_results, conversation_log, timing, supervisor_calls)
            
            
            if not assessment.more_info_required:
                print(f"✅ Research complete after {current_round} rounds - sufficient information gathered")
                break
        
        if current_round >= self.max_rounds:
            print(f"⏰ Research stopped after {self.max_rounds} rounds (max limit reached)")
        
        # Step 3: Synthesize final answer from all gathered information
        if self.callback:
            self.callback("supervisor", "📊 Synthesizing comprehensive research report...")
        
        final_answer = self._synthesize_final_answer_adaptive(
            task, query_results, conversation_log, timing, supervisor_calls
        )
        
        return final_answer
    
    def _run_evaluation_mode(
        self,
        task: str,
        conversation_log: Dict,
        timing: Dict,
        supervisor_calls: List[Dict],
        worker_calls: List[Dict]
    ) -> str:
        """
        Run backward-compatible single-round evaluation mode.
        
        This preserves the original ResSwarm behavior for benchmark compatibility.
        """
        print(f"📊 EVALUATION MODE: Single-round research")
        final_answer = self._run_supervisor_worker_protocol(
            task, conversation_log, timing, supervisor_calls, worker_calls
        )
        return final_answer
    
    def _run_multi_query_mode(
        self,
        task: str,
        conversation_log: Dict,
        timing: Dict,
        supervisor_calls: List[Dict],
        worker_calls: List[Dict]
    ) -> str:
        """
        Run multi-query mode: 5 queries with 10 sources each.
        
        This mode generates 5 search queries with corresponding sub-tasks
        and processes them all in a single round with 10 sources per query.
        """
        print(f"🔍 MULTI-QUERY MODE: 5 queries x 10 sources")
        
        # Retry loop for supervisor query generation
        max_retries = 3
        queries_data = []
        research_plan = ""
        synthesis_strategy = ""
        supervisor_response = None
        
        for attempt in range(max_retries):
            try:
                # Step 1: Supervisor generates 5 queries with sub-tasks
                print("📝 Generating 5 search queries with sub-tasks...")
                supervisor_prompt = MULTI_QUERY_GENERATION_PROMPT.format(query=task)
                
                # Log supervisor prompt (only on first attempt)
                if attempt == 0:
                    conversation_log["conversation"].append({
                        "role": "supervisor",
                        "type": "multi_query_generation",
                        "content": supervisor_prompt,
                        "response": None
                    })
                
                if self.callback and attempt == 0:
                    self.callback("supervisor", None, is_final=False)
                
                # Get supervisor's multi-query generation
                start_time = time.time()
                supervisor_response, usage = self.supervisor_client.chat([
                    {"role": "user", "content": supervisor_prompt}
                ])
                timing["supervisor_time"] += time.time() - start_time
                
                # Track supervisor call tokens
                supervisor_call = {
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens
                }
                supervisor_calls.append(supervisor_call)
                
                # Parse supervisor's multi-query response
                supervisor_json = _extract_json(supervisor_response[0])
                research_plan = supervisor_json.get("research_plan", "")
                queries_data = supervisor_json.get("queries", [])
                synthesis_strategy = supervisor_json.get("synthesis_strategy", "")
                
                # Validate that we got queries
                if not queries_data:
                    if attempt < max_retries - 1:
                        print(f"⚠️ No queries extracted, retrying...")
                        time.sleep(2)  # Brief delay before retry
                        continue
                    else:
                        raise ValueError("Failed to generate queries after all retries")
                
                # Success - break out of retry loop
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Error in query generation, retrying...")
                    time.sleep(2)
                    continue
                else:
                    # Final attempt failed
                    raise ValueError(f"Multi-query generation failed: {str(e)}")
        
        # Log supervisor response (final successful attempt)
        if supervisor_response:
            conversation_log["conversation"][-1]["response"] = supervisor_response[0]
            conversation_log["conversation"][-1]["parsed"] = {
                "research_plan": research_plan,
                "queries": queries_data,
                "synthesis_strategy": synthesis_strategy
            }
            
            if self.callback:
                self.callback("supervisor", {"role": "assistant", "content": supervisor_response[0]}, is_final=False)
        
        print(f"📋 Research Plan: {research_plan}")
        print(f"💬 Generated {len(queries_data)} queries")
        for i, query_data in enumerate(queries_data, 1):
            print(f"  {i}. Query: {query_data['search_query']}")
            print(f"     Sub-task: {query_data['sub_task']}")
        
        # Step 2: Process all queries with web search (10 sources each) - PARALLEL PROCESSING
        all_findings = {}
        query_results = {}
        
        # Ensure we have queries to process
        if len(queries_data) == 0:
            raise ValueError("No queries available for processing")
        
        print(f"\n🚀 Processing {len(queries_data)} queries in parallel...")
        
        # Process all queries in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries_data), 5)) as executor:
            # Submit all query processing tasks
            future_to_query = {}
            for i, query_data in enumerate(queries_data):
                future = executor.submit(
                    self._process_single_query,
                    query_data, i + 1, len(queries_data), conversation_log, timing, worker_calls
                )
                future_to_query[future] = query_data
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_query):
                query_data = future_to_query[future]
                search_query = query_data["search_query"]
                
                try:
                    result = future.result()
                    query_results[search_query] = result
                    all_findings[search_query] = result["summaries"]
                    print(f"✅ Completed query: {search_query[:50]}...")
                except Exception as e:
                    print(f"❌ Error processing query '{search_query}': {e}")
                    # Store empty result to prevent synthesis failure
                    query_results[search_query] = {
                        "sub_task": query_data["sub_task"],
                        "summaries": [],
                        "metadata": []
                    }
                    all_findings[search_query] = []
        
        # Step 3: Supervisor synthesizes all findings
        print("\n📊 Synthesizing all findings...")
        
        # Check synthesis strategy
        if self.synthesis_strategy == "chunked":
            # Use chunked synthesis approach
            final_answer = self._synthesize_chunked_multi_query(
                task, research_plan, query_results, synthesis_strategy,
                conversation_log, timing, supervisor_calls
            )
        else:
            # Use existing single synthesis approach
            # Format findings for synthesis
            findings_text = self._format_multi_query_findings(query_results)
            
            synthesis_prompt = SUPERVISOR_SYNTHESIS_PROMPT.format(
                original_task=task,
                research_plan=research_plan,
                qa_pairs=findings_text,
                synthesis_strategy=synthesis_strategy
            )
            
            # Log synthesis prompt
            conversation_log["conversation"].append({
                "role": "supervisor",
                "type": "synthesis",
                "content": synthesis_prompt,
                "response": None
            })
            
            if self.callback:
                self.callback("supervisor", None, is_final=False)
            
            # Get final synthesis
            start_time = time.time()
            synthesis_response, usage = self.supervisor_client.chat([
                {"role": "user", "content": synthesis_prompt}
            ])
            timing["supervisor_time"] += time.time() - start_time
            
            # Track supervisor call tokens
            supervisor_call = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens
            }
            supervisor_calls.append(supervisor_call)
            
            final_answer = synthesis_response[0]
            
            # Log synthesis response
            conversation_log["conversation"][-1]["response"] = final_answer
            
            if self.callback:
                self.callback("supervisor", {"role": "assistant", "content": final_answer}, is_final=True)
        
        print("✅ Multi-query research completed")
        
        return final_answer
    
    def _run_supervisor_worker_protocol(
        self, 
        task: str, 
        conversation_log: Dict,
        timing: Dict,
        supervisor_calls: List[Dict],
        worker_calls: List[Dict]
    ) -> str:
        """
        Execute the supervisor-worker protocol with max_rounds=1.
        
        Args:
            task: Research task
            conversation_log: Conversation logging dict
            timing: Timing tracking dict
            supervisor_calls: List to collect supervisor token usage per call
            worker_calls: List to collect worker token usage per call
            
        Returns:
            Final answer string
        """
        
        # Supervisor initial prompt for task decomposition
        supervisor_prompt = self._get_supervisor_initial_prompt(task)
        
        # Log supervisor prompt
        conversation_log["conversation"].append({
            "role": "supervisor",
            "type": "initial_prompt",
            "content": supervisor_prompt,
            "response": None
        })
        
        if self.callback:
            self.callback("supervisor", None, is_final=False)
        
        # Get supervisor's task decomposition
        start_time = time.time()
        supervisor_response, usage = self.supervisor_client.chat([
            {"role": "user", "content": supervisor_prompt}
        ])
        timing["supervisor_time"] += time.time() - start_time
        
        # Track supervisor call tokens
        supervisor_call = {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens
        }
        supervisor_calls.append(supervisor_call)
        
        # Parse supervisor's decomposition
        try:
            supervisor_json = _extract_json(supervisor_response[0])
            subqueries = supervisor_json.get("subqueries", [])
            synthesis_strategy = supervisor_json.get("synthesis_strategy", "")
        except Exception as e:
            print(f"Error parsing supervisor response: {e}")
            # Fallback: treat the entire response as a single subquery
            subqueries = [supervisor_response[0]]
            synthesis_strategy = "Summarize the worker's findings"
        
        # Log supervisor response
        conversation_log["conversation"][-1]["response"] = supervisor_response[0]
        conversation_log["conversation"][-1]["parsed"] = {
            "subqueries": subqueries,
            "synthesis_strategy": synthesis_strategy
        }
        
        if self.callback:
            self.callback("supervisor", {"role": "assistant", "content": supervisor_response[0]}, is_final=False)
        
        print(f"💬 SUPERVISOR DECOMPOSITION: {len(subqueries)} subqueries")
        for i, subquery in enumerate(subqueries, 1):
            print(f"  {i}. {subquery}")
        
        # Worker processes subqueries with web search
        worker_responses = []
        for i, subquery in enumerate(subqueries):
            worker_response = self._process_worker_subquery(
                subquery, i, conversation_log, timing, worker_calls
            )
            worker_responses.append(worker_response)
        
        # Supervisor synthesizes final answer
        final_answer = self._synthesize_final_answer(
            task, subqueries, worker_responses, synthesis_strategy,
            conversation_log, timing, supervisor_calls
        )
        
        return final_answer
    
    def _assess_and_generate_query(
        self, 
        task: str, 
        query_results: Optional[Dict[str, List[Dict[str, Any]]]], 
        conversation_log: Dict,
        timing: Dict,
        supervisor_calls: List[Dict]
    ) -> AssessmentOutput:
        """
        Assess research completeness and generate next search query if needed.
        
        Args:
            task: Original research task
            query_results: Dictionary of search queries and their results
            conversation_log: Conversation logging dict
            timing: Timing tracking dict
            supervisor_usage: Supervisor usage tracking
            
        Returns:
            AssessmentOutput with next action decision
        """
        start_time = time.time()
        
        try:
            if not query_results:
                # First round - generate initial search query
                messages = [{
                    "role": "user", 
                    "content": INITIAL_QUERY_PROMPT.format(query=task)
                }]
                response, usage = self.supervisor_client.chat(messages)
                timing["supervisor_time"] += time.time() - start_time
                # Track granular call-level usage
                supervisor_calls.append({
                    "input_tokens": getattr(usage, 'prompt_tokens', 0),
                    "output_tokens": getattr(usage, 'completion_tokens', 0)
                })
                
                # Parse JSON response to extract clean search query
                try:
                    parsed_response = _extract_json(response[0])
                    search_query = parsed_response.get("search_query", "").strip()
                    explanation = parsed_response.get("explanation", "")
                    print(f"[DEBUG] Extracted search query: '{search_query}'")
                except Exception as e:
                    print(f"[DEBUG] Failed to parse JSON response, using fallback: {e}")
                    # Fallback to original method
                    search_query = response[0].strip()
                
                # Log initial query generation
                conversation_log["conversation"].append({
                    "role": "supervisor",
                    "type": "initial_query_generation",
                    "prompt": messages[0]["content"],
                    "response": response[0],
                    "extracted_query": search_query
                })
                
                return AssessmentOutput(
                    more_info_required=True,
                    search_query=search_query
                )
            else:
                # Subsequent rounds - assess information completeness
                formatted_information = self._format_query_results(query_results)
                
                messages = [{
                    "role": "user",
                    "content": ASSESSMENT_PROMPT.format(
                        query=task,
                        information=formatted_information
                    )
                }]
                
                response, usage = self.supervisor_client.chat(messages)
                timing["supervisor_time"] += time.time() - start_time
                # Track granular call-level usage
                supervisor_calls.append({
                    "input_tokens": getattr(usage, 'prompt_tokens', 0),
                    "output_tokens": getattr(usage, 'completion_tokens', 0)
                })
                
                # Parse assessment response
                try:
                    assessment_json = _extract_json(response[0])
                    assessment = AssessmentOutput.model_validate(assessment_json)
                    
                    # Log assessment
                    conversation_log["conversation"].append({
                        "role": "supervisor",
                        "type": "information_assessment",
                        "prompt": messages[0]["content"],
                        "response": response[0],
                        "parsed": assessment_json
                    })
                    
                    return assessment
                    
                except Exception as e:
                    print(f"Error parsing assessment response: {e}")
                    # Default to stopping research if can't parse
                    return AssessmentOutput(more_info_required=False)
                    
        except Exception as e:
            timing["supervisor_time"] += time.time() - start_time
            print(f"Error in assessment: {e}")
            # Default to stopping research on error
            return AssessmentOutput(more_info_required=False)
    
    def _extract_metadata(
        self, 
        search_query: str, 
        conversation_log: Dict,
        timing: Dict,
        max_results: Optional[int] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Extract metadata from web search with parallel processing.
        
        Args:
            search_query: Query to search for
            conversation_log: Conversation logging dict
            timing: Timing tracking dict
            
        Returns:
            Tuple of (urls, metadata_content)
        """
        if not self.web_search_enabled or not self.search_tool:
            return [], []
        
        print(f"🔍 Searching for: {search_query}")
        start_time = time.time()
        
        try:
            # Perform web search and scraping
            search_results = self.search_tool.search_and_scrape(
                search_query, 
                max_results=max_results or self.max_sources_per_round,
                scrape_content=True
            )
            
            timing["search_time"] += time.time() - start_time
            
            # Extract URLs and content
            urls = []
            metadata = []
            
            for result in search_results:
                urls.append(result["url"])
                if result.get("scraped") and result.get("content"):
                    metadata.append(result["content"])
                else:
                    metadata.append(f"Content not available for {result['url']}")
            
            # Log search results
            conversation_log["search_queries"].append(search_query)
            conversation_log["search_results"].append(search_results)
            
            print(f"✅ Found {len(urls)} sources, {len([m for m in metadata if len(m) > 100])} with content")
            
            return urls, metadata
            
        except Exception as e:
            timing["search_time"] += time.time() - start_time
            print(f"Error in metadata extraction: {e}")
            return [], []
    
    def _summarize_metadata(
        self,
        search_query: str,
        metadata: List[str],
        conversation_log: Dict,
        timing: Dict,
        worker_calls: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Summarize metadata using worker models with relevance filtering.
        
        Args:
            search_query: The search query these results are for
            metadata: List of scraped content to summarize
            conversation_log: Conversation logging dict
            timing: Timing tracking dict
            worker_usage: Worker usage tracking
            
        Returns:
            List of relevant summaries with metadata
        """
        if not metadata:
            return []
        
        start_time = time.time()
        
        try:
            # Create job manifests for each content chunk
            job_manifests = []
            
            for source_id, content in enumerate(metadata):
                if not content or len(content) < 100:
                    continue
                
                # Chunk content for processing
                chunks = chunk_by_section(content, max_chunk_size=6000, overlap=200)
                
                for chunk_id, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 100:
                        continue
                    
                    job_manifest = JobManifest(
                        chunk=chunk,
                        chunk_id=chunk_id,
                        task_id=source_id,
                        job_id=len(job_manifests)
                    )
                    job_manifests.append(job_manifest)
            
            if not job_manifests:
                return []
            
            # Process in batches for efficiency
            summaries = []
            batch_size = min(self.worker_batch_size, len(job_manifests))
            total_batches = (len(job_manifests) + batch_size - 1) // batch_size
            
            # Import tqdm for progress bar
            try:
                from tqdm import tqdm
                use_tqdm = True
            except ImportError:
                use_tqdm = False
                print(f"🔄 Processing {len(job_manifests)} content chunks in {total_batches} batches...")
            
            # Track sources for progress indication
            sources_processed = set()
            total_sources = len(set(jm.task_id for jm in job_manifests))
            
            if use_tqdm:
                pbar = tqdm(total=len(job_manifests), desc="Processing chunks", unit="chunk")
            
            for i in range(0, len(job_manifests), batch_size):
                batch = job_manifests[i:i+batch_size]
                
                # Prepare worker messages for batch processing
                worker_messages = []
                for job_manifest in batch:
                    prompt = WORKER_SUMMARIZE_PROMPT.format(
                        query=search_query,
                        content=job_manifest.chunk
                    )
                    worker_messages.append({"role": "user", "content": prompt})
                
                # Process batch
                if not use_tqdm:
                    print(f"⏳ Sending batch to worker ({len(worker_messages)} messages)...")
                worker_responses, usage = self.worker_client.chat(worker_messages)
                if not use_tqdm:
                    print(f"✅ Worker batch complete")
                # Track granular call-level usage for each message in batch
                for _ in worker_messages:
                    worker_calls.append({
                        "input_tokens": getattr(usage, 'prompt_tokens', 0) // len(worker_messages),
                        "output_tokens": getattr(usage, 'completion_tokens', 0) // len(worker_messages)
                    })
                
                # Process responses
                for worker_response, job_manifest in zip(worker_responses, batch):
                    try:
                        # Extract JSON from potentially markdown-wrapped response
                        parsed_response = _extract_json(worker_response)
                        
                        # Try to validate with JobOutput schema
                        try:
                            job_output = JobOutput.model_validate(parsed_response)
                            is_relevant = job_output.answer.lower() == "relevant"
                        except Exception as validation_error:
                            # Create fallback JobOutput for simple responses
                            if isinstance(parsed_response, str):
                                # If it's just a string, assume it's the explanation
                                job_output = JobOutput(
                                    explanation=parsed_response,
                                    answer="relevant" if "relevant" in parsed_response.lower() else "not relevant"
                                )
                            else:
                                # If it's a dict but missing required fields, fill in defaults
                                if isinstance(parsed_response, dict):
                                    # Extract fields that exist in JobOutput model, excluding duplicates
                                    valid_fields = {k: v for k, v in parsed_response.items() if k in JobOutput.model_fields}
                                    
                                    # Ensure required fields are present
                                    if "explanation" not in valid_fields:
                                        valid_fields["explanation"] = str(parsed_response)
                                    if "answer" not in valid_fields:
                                        valid_fields["answer"] = "relevant"
                                    
                                    job_output = JobOutput(**valid_fields)
                                else:
                                    # If it's not a dict, treat as explanation string
                                    job_output = JobOutput(
                                        explanation=str(parsed_response),
                                        answer="relevant"
                                    )
                            is_relevant = job_output.answer.lower() == "relevant"
                        
                        if is_relevant:
                            # Use enhanced fields if available, fallback to explanation
                            if job_output.key_findings:
                                # Helper function to format field (handles both string and list)
                                def format_field(field):
                                    if isinstance(field, list):
                                        return "\n".join(f"• {item}" for item in field)
                                    return field
                                
                                # Enhanced multi-dimensional summary
                                summary_text = f"KEY FINDINGS: {format_field(job_output.key_findings)}\n\n"
                                if job_output.evidence_data:
                                    summary_text += f"EVIDENCE: {format_field(job_output.evidence_data)}\n\n"
                                if job_output.context_implications:
                                    summary_text += f"CONTEXT: {format_field(job_output.context_implications)}\n\n"
                                if job_output.information_gaps:
                                    summary_text += f"GAPS: {format_field(job_output.information_gaps)}"
                            else:
                                # Fallback to simple explanation
                                summary_text = job_output.explanation or "No summary available"
                            
                            summary = {
                                "source_id": job_manifest.task_id,
                                "chunk_id": job_manifest.chunk_id,
                                "summary": summary_text,
                                "raw_chunk": job_manifest.chunk,
                                "relevant": True,
                                "search_query": search_query,
                                "relevance_score": job_output.relevance_score
                            }
                            summaries.append(summary)
                        
                        # Track source completion
                        sources_processed.add(job_manifest.task_id)
                        
                    except Exception as e:
                        print(f"Error processing worker response: {e}")
                        # Still include but mark as potentially unreliable
                        summary = {
                            "source_id": job_manifest.task_id,
                            "chunk_id": job_manifest.chunk_id,
                            "summary": f"Processing error: {str(e)}",
                            "raw_chunk": job_manifest.chunk,
                            "relevant": False,
                            "search_query": search_query
                        }
                        summaries.append(summary)
                        sources_processed.add(job_manifest.task_id)
                    
                    # Update progress bar
                    if use_tqdm:
                        pbar.update(1)
                        pbar.set_postfix({
                            "sources": f"{len(sources_processed)}/{total_sources}",
                            "relevant": len([s for s in summaries if s.get("relevant")])
                        })
                
                # Simple progress for non-tqdm case
                if not use_tqdm:
                    current_batch = i//batch_size + 1
                    print(f"✅ Batch {current_batch}/{total_batches} complete - {len(sources_processed)}/{total_sources} sources processed")
            
            if use_tqdm:
                pbar.close()
            
            timing["worker_time"] += time.time() - start_time
            
            # Filter for relevant summaries only
            relevant_summaries = [s for s in summaries if s.get("relevant")]
            
            print(f"📊 Summarized {len(job_manifests)} chunks → {len(relevant_summaries)} relevant")
            
            return relevant_summaries
            
        except Exception as e:
            timing["worker_time"] += time.time() - start_time
            print(f"Error in metadata summarization: {e}")
            print(f"Error type: {type(e)}")
            print(f"Error args: {e.args}")
            import traceback
            print(f"Error traceback: {traceback.format_exc()}")
            return []
    
    def _format_query_results(self, query_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Format query results for supervisor assessment.
        
        Args:
            query_results: Dictionary mapping search queries to their results
            
        Returns:
            Formatted string with all research findings
        """
        formatted_sections = []
        
        for query, summaries in query_results.items():
            formatted_sections.append(f"SEARCH QUERY: {query}")
            
            if summaries:
                summary_lines = []
                for i, summary in enumerate(summaries, 1):
                    content = summary.get("summary", "").strip()
                    if content:
                        summary_lines.append(f"Source {i}: {content}")
                
                if summary_lines:
                    formatted_sections.append("\n".join(summary_lines))
                else:
                    formatted_sections.append("No relevant information found.")
            else:
                formatted_sections.append("No information found for this query.")
            
            formatted_sections.append("----------")
        
        return "\n\n".join(formatted_sections)
    
    def _synthesize_final_answer_adaptive(
        self,
        task: str,
        query_results: Dict[str, List[Dict[str, Any]]],
        conversation_log: Dict,
        timing: Dict,
        supervisor_calls: List[Dict]
    ) -> str:
        """
        Synthesize final answer from all research rounds in adaptive mode.
        
        Args:
            task: Original research task
            query_results: All search results from multiple rounds
            conversation_log: Conversation logging dict
            timing: Timing tracking dict
            supervisor_calls: Supervisor call-level usage tracking
            
        Returns:
            Comprehensive final answer
        """
        print("🔎 SUPERVISOR SYNTHESIZING COMPREHENSIVE ANSWER")
        
        start_time = time.time()
        
        try:
            # Format all gathered information
            formatted_information = self._format_query_results(query_results)
            
            # Create synthesis prompt
            synthesis_prompt = FINAL_SYNTHESIS_PROMPT.format(
                query=task,
                information=formatted_information
            )
            
            # Log synthesis prompt
            conversation_log["conversation"].append({
                "role": "supervisor",
                "type": "final_synthesis",
                "prompt": synthesis_prompt,
                "response": None
            })
            
            # Get final synthesis
            messages = [{"role": "user", "content": synthesis_prompt}]
            response, usage = self.supervisor_client.chat(messages)
            
            timing["supervisor_time"] += time.time() - start_time
            # Track granular call-level usage
            supervisor_calls.append({
                "input_tokens": getattr(usage, 'prompt_tokens', 0),
                "output_tokens": getattr(usage, 'completion_tokens', 0)
            })
            
            final_answer = response[0]
            
            # Log synthesis response
            conversation_log["conversation"][-1]["response"] = final_answer
            
            print("✅ COMPREHENSIVE ANSWER SYNTHESIZED")
            
            return final_answer
            
        except Exception as e:
            timing["supervisor_time"] += time.time() - start_time
            print(f"Error in final synthesis: {e}")
            return f"Error synthesizing final answer: {str(e)}"
    
    def _get_supervisor_initial_prompt(self, task: str) -> str:
        """Generate the initial prompt for supervisor task decomposition."""
        return f"""You are a research supervisor tasked with decomposing complex research questions into focused subqueries that can be answered by a worker with web search capabilities.

Your task: {task}

Please decompose this task into 2-4 specific, focused subqueries that when answered together will provide comprehensive coverage of the research question. Each subquery should be:
1. Specific and answerable with web search
2. Focused on a particular aspect of the main question  
3. Clear and unambiguous

Also provide a synthesis strategy explaining how the worker's answers should be combined.

Respond in JSON format:
{{
    "subqueries": [
        "First focused subquery...",
        "Second focused subquery...",
        "Third focused subquery..."
    ],
    "synthesis_strategy": "How to combine the answers into a comprehensive response..."
}}"""

    def _process_worker_subquery(
        self, 
        subquery: str, 
        index: int,
        conversation_log: Dict,
        timing: Dict,
        worker_calls: List[Dict]
    ) -> str:
        """
        Process a single subquery with the worker, including web search.
        
        Args:
            subquery: The subquery to process
            index: Index of the subquery
            conversation_log: Conversation logging dict
            timing: Timing tracking dict
            worker_calls: List to collect worker token usage per call
            
        Returns:
            Worker's response to the subquery
        """
        print(f"🔨 WORKER PROCESSING SUBQUERY {index + 1}: {subquery}")
        
        # Prepare worker prompt with search capability info
        worker_prompt = self._get_worker_prompt(subquery)
        
        # Log worker prompt
        conversation_log["conversation"].append({
            "role": "worker",
            "type": "subquery",
            "subquery_index": index,
            "subquery": subquery,
            "prompt": worker_prompt,
            "search_results": [],
            "response": None
        })
        
        # Perform web search if enabled
        search_context = ""
        if self.web_search_enabled and self.search_tool:
            search_context = self._perform_web_search(subquery, index, conversation_log, timing)
            
        # Add search context to worker prompt
        if search_context:
            worker_prompt_with_context = f"""{worker_prompt}

**Web Search Results:**
{search_context}

Please use the above search results to answer the question. Cite specific sources when possible."""
        else:
            worker_prompt_with_context = worker_prompt
        
        if self.callback:
            self.callback("worker", None, is_final=False)
        
        # Get worker response
        start_time = time.time()
        worker_response, usage = self.worker_client.chat([
            {"role": "user", "content": worker_prompt_with_context}
        ])
        timing["worker_time"] += time.time() - start_time
        
        # Track worker call tokens
        worker_call = {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens
        }
        worker_calls.append(worker_call)
        
        # Log worker response
        conversation_log["conversation"][-1]["response"] = worker_response[0]
        
        if self.callback:
            self.callback("worker", {"role": "assistant", "content": worker_response[0]}, is_final=False)
        
        print(f"✅ WORKER RESPONSE {index + 1} COMPLETED")
        
        return worker_response[0]
    
    def _get_worker_prompt(self, subquery: str) -> str:
        """Generate prompt for worker to answer a subquery."""
        return f"""You are a research assistant tasked with answering focused research questions using web search capabilities.

Question: {subquery}

Please provide a comprehensive, well-researched answer to this question. If web search results are provided, use them to support your answer with specific citations and references."""

    def _perform_web_search(
        self, 
        subquery: str, 
        index: int,
        conversation_log: Dict,
        timing: Dict
    ) -> str:
        """
        Perform web search for a subquery and return formatted results.
        
        Args:
            subquery: The search query
            index: Index of the subquery
            conversation_log: Conversation logging dict
            timing: Timing tracking dict
            
        Returns:
            Formatted search results context
        """
        print(f"🔍 SEARCHING WEB FOR: {subquery}")
        
        start_time = time.time()
        try:
            search_results = self.search_tool.search_and_scrape(subquery, max_results=3)
            timing["search_time"] += time.time() - start_time
            
            # Log search results
            conversation_log["search_queries"].append(subquery)
            conversation_log["search_results"].append(search_results)
            conversation_log["conversation"][-1]["search_results"] = search_results
            
            # Format search results for worker
            search_context = ""
            for i, result in enumerate(search_results, 1):
                search_context += f"\n**Source {i}: {result['url']}**\n"
                search_context += f"Title: {result.get('title', 'N/A')}\n"
                search_context += f"Content: {result.get('content', 'Content not available')[:1000]}...\n\n"
            
            print(f"✅ SEARCH COMPLETED: {len(search_results)} results")
            return search_context
            
        except Exception as e:
            timing["search_time"] += time.time() - start_time
            print(f"❌ SEARCH ERROR: {e}")
            return ""
    
    def _synthesize_final_answer(
        self,
        original_task: str,
        subqueries: List[str],
        worker_responses: List[str],
        synthesis_strategy: str,
        conversation_log: Dict,
        timing: Dict,
        supervisor_calls: List[Dict]
    ) -> str:
        """
        Have supervisor synthesize final answer from worker responses.
        
        Args:
            original_task: The original research task
            subqueries: List of subqueries
            worker_responses: List of worker responses
            synthesis_strategy: How to combine responses
            conversation_log: Conversation logging dict
            timing: Timing tracking dict
            supervisor_calls: List to collect supervisor token usage per call
            
        Returns:
            Final synthesized answer
        """
        print("🔎 SUPERVISOR SYNTHESIZING FINAL ANSWER")
        
        # Prepare synthesis prompt
        synthesis_prompt = self._get_synthesis_prompt(
            original_task, subqueries, worker_responses, synthesis_strategy
        )
        
        # Log synthesis prompt
        conversation_log["conversation"].append({
            "role": "supervisor",
            "type": "synthesis",
            "prompt": synthesis_prompt,
            "response": None
        })
        
        if self.callback:
            self.callback("supervisor", None, is_final=False)
        
        # Get supervisor's final synthesis
        start_time = time.time()
        final_response, usage = self.supervisor_client.chat([
            {"role": "user", "content": synthesis_prompt}
        ])
        timing["supervisor_time"] += time.time() - start_time
        
        # Track supervisor synthesis call tokens
        supervisor_call = {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens
        }
        supervisor_calls.append(supervisor_call)
        
        final_answer = final_response[0]
        
        # Log synthesis response
        conversation_log["conversation"][-1]["response"] = final_answer
        
        if self.callback:
            self.callback("supervisor", {"role": "assistant", "content": final_answer}, is_final=True)
        
        print("✅ FINAL ANSWER SYNTHESIZED")
        
        return final_answer
    
    def _get_synthesis_prompt(
        self,
        original_task: str,
        subqueries: List[str],
        worker_responses: List[str],
        synthesis_strategy: str
    ) -> str:
        """Generate prompt for supervisor to synthesize final answer."""
        
        # Combine subqueries with responses
        qa_pairs = ""
        for i, (subquery, response) in enumerate(zip(subqueries, worker_responses), 1):
            qa_pairs += f"\n**Research Question {i}:** {subquery}\n"
            qa_pairs += f"**Answer {i}:** {response}\n\n"
        
        return f"""You are a research supervisor tasked with synthesizing comprehensive answers from focused research findings.

**Original Research Task:** {original_task}

**Research Findings:**
{qa_pairs}

**Synthesis Strategy:** {synthesis_strategy}

Please synthesize these research findings into a comprehensive, well-structured answer to the original research task. The answer should:
1. Be comprehensive and address all aspects of the original question
2. Integrate insights from multiple research findings
3. Include specific citations and references where available
4. Be well-organized with clear structure and headings
5. Provide actionable insights and conclusions

Write a detailed research report that directly answers the original task."""

    def _save_log(
        self, 
        conversation_log: Dict, 
        task: str, 
        query_id: Optional[int] = None,
        logging_id: Optional[str] = None
    ):
        """Save conversation log to file."""
        
        if logging_id:
            log_filename = f"{logging_id}_deepres.json"
        elif query_id:
            log_filename = f"query_{query_id}_deepres.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = re.sub(r"[^a-zA-Z0-9]", "_", task[:15])
            log_filename = f"{timestamp}_{safe_task}_deepres.json"
        
        log_path = os.path.join(self.log_dir, log_filename)
        
        print(f"\n=== SAVING LOG TO {log_path} ===")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(conversation_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving log to {log_path}: {e}")
    
    def _summarize_metadata_with_subtask(
        self,
        search_query: str,
        sub_task: str,
        metadata: List[str],
        conversation_log: Dict,
        timing: Dict,
        worker_calls: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Summarize metadata with awareness of the sub-task.
        Similar to _summarize_metadata but includes sub-task in the prompt.
        """
        if not metadata:
            return []
        
        summaries = []
        print(f"  📝 Workers summarizing {len(metadata)} sources for sub-task: {sub_task}")
        
        # Process in batches
        batch_size = self.worker_batch_size
        
        for i in range(0, len(metadata), batch_size):
            batch = metadata[i:i + batch_size]
            
            for j, content in enumerate(batch):
                try:
                    # Use the new extraction prompt for multi-query mode
                    worker_prompt = WORKER_MULTI_QUERY_EXTRACT_PROMPT.format(
                        query=search_query,
                        sub_task=sub_task,
                        content=content[:8000]  # Limit content length
                    )
                    
                    # Log worker prompt
                    conversation_log["conversation"].append({
                        "role": "worker",
                        "type": "multi_query_extraction",
                        "subquery_index": i * batch_size + j,
                        "subquery": search_query,
                        "sub_task": sub_task,
                        "prompt": worker_prompt,
                        "search_results": [],
                        "response": None
                    })
                    
                    # Get worker response
                    start_time = time.time()
                    response, usage = self.worker_client.chat([
                        {"role": "user", "content": worker_prompt}
                    ])
                    timing["worker_time"] += time.time() - start_time
                    
                    # Track worker call tokens
                    worker_call = {
                        "input_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens
                    }
                    worker_calls.append(worker_call)
                    
                    # Log worker response
                    conversation_log["conversation"][-1]["response"] = response[0]
                    
                    # Parse response
                    try:
                        summary_json = _extract_json(response[0])
                        summaries.append({
                            "explanation": summary_json.get("explanation", response[0]),
                            "answer": summary_json.get("answer", "relevant"),
                            "sub_task": sub_task
                        })
                    except Exception as e:
                        print(f"    ⚠️  Failed to parse worker response: {e}")
                        summaries.append({
                            "explanation": response[0],
                            "answer": "relevant",
                            "sub_task": sub_task
                        })
                    
                except Exception as e:
                    print(f"    ⚠️  Worker error: {e}")
                    summaries.append({
                        "explanation": f"Error processing content: {str(e)}",
                        "answer": "error",
                        "sub_task": sub_task
                    })
        
        print(f"  ✅ Summarized {len(summaries)} sources")
        return summaries
    
    def _process_single_query(
        self,
        query_data: Dict[str, str],
        query_num: int,
        total_queries: int,
        conversation_log: Dict,
        timing: Dict,
        worker_calls: List[Dict]
    ) -> Dict[str, Any]:
        """
        Process a single query in parallel: search + worker summarization.
        
        Args:
            query_data: Dictionary with 'search_query' and 'sub_task'
            query_num: Current query number (1-indexed for display)
            total_queries: Total number of queries
            conversation_log: Conversation log for this task
            timing: Timing tracking dictionary
            worker_calls: Worker calls tracking list
            
        Returns:
            Dictionary with 'sub_task', 'summaries', and 'metadata'
        """
        search_query = query_data["search_query"]
        sub_task = query_data["sub_task"]
        
        print(f"🔍 Processing Query {query_num}/{total_queries}: {search_query}")
        
        # Search for 10 sources
        search_urls, metadata = self._extract_metadata(
            search_query, conversation_log, timing, max_results=10
        )
        
        # Summarize with awareness of sub-task
        summaries = self._summarize_metadata_with_subtask(
            search_query, sub_task, metadata, conversation_log, timing, worker_calls
        )
        
        # Return structured result
        return {
            "sub_task": sub_task,
            "summaries": summaries,
            "metadata": metadata
        }
    
    def _call_supervisor_with_json(self, prompt: str, conversation_log: Dict, 
                                  timing: Dict, supervisor_calls: List[Dict]) -> Dict:
        """Call supervisor with JSON mode if supported, with fallback"""
        
        # Check if model supports JSON mode (OpenAI models)
        supports_json_mode = (
            self.supervisor_client.__class__.__name__ == "OpenAIClient" or
            (hasattr(self.supervisor_client, 'model_name') and 
             'gpt' in self.supervisor_client.model_name.lower())
        )
        
        start_time = time.time()
        
        if supports_json_mode:
            try:
                # Use structured output
                response, usage = self.supervisor_client.chat(
                    [{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
            except Exception as e:
                # Fallback if response_format not supported
                print(f"[DEBUG] JSON mode failed, using standard mode: {e}")
                response, usage = self.supervisor_client.chat(
                    [{"role": "user", "content": prompt}]
                )
        else:
            # Standard mode for non-OpenAI models
            response, usage = self.supervisor_client.chat(
                [{"role": "user", "content": prompt}]
            )
        
        timing["supervisor_time"] += time.time() - start_time
        
        # Track usage
        supervisor_calls.append({
            "input_tokens": getattr(usage, 'prompt_tokens', 0),
            "output_tokens": getattr(usage, 'completion_tokens', 0)
        })
        
        # Extract and return JSON
        return _extract_json(response[0])
    
    def _synthesize_chunked_multi_query(
        self, 
        task: str, 
        research_plan: str, 
        query_results: Dict[str, Dict],
        synthesis_strategy: str,
        conversation_log: Dict, 
        timing: Dict, 
        supervisor_calls: List[Dict]
    ) -> str:
        """Synthesize findings using chunked approach with dynamic sections"""
        
        # Step 1: Planning Phase - Generate dynamic sections
        print("📝 Planning report structure...")
        planning_prompt = SYNTHESIS_PLANNING_PROMPT.format(
            task=task,
            research_plan=research_plan,
            findings_summary=self._create_findings_summary(query_results),
            num_queries=len(query_results)
        )
        
        # Get section plan from supervisor with JSON mode
        section_plan = self._call_supervisor_with_json(
            planning_prompt, conversation_log, timing, supervisor_calls
        )
        
        # Validate the plan
        section_plan = self._validate_section_plan(section_plan)
        
        sections = section_plan["sections"]
        query_mappings = section_plan["query_mappings"]
        
        print(f"📊 Generated {len(sections)} main sections for the report")
        total_target_words = 0
        for section in sections:
            target_words = section.get('target_words', 0)
            total_target_words += target_words
            word_info = f" (~{target_words} words)" if target_words > 0 else ""
            print(f"   {section['number']}. {section['title']}{word_info}")
            if "subsections" in section:
                for subsection in section["subsections"]:
                    print(f"      - {subsection}")
        
        if total_target_words > 0:
            print(f"📝 Total planned words: ~{total_target_words}")
        
        # Log planning phase
        conversation_log["conversation"].append({
            "role": "supervisor",
            "type": "chunked_synthesis_planning",
            "prompt": planning_prompt,
            "response": section_plan
        })
        
        # Step 2: Write sections in chunks
        all_sections = []
        total_chunks = (len(sections) + self.sections_per_chunk - 1) // self.sections_per_chunk
        
        for chunk_idx, chunk_start in enumerate(range(0, len(sections), self.sections_per_chunk)):
            chunk_sections = sections[chunk_start:chunk_start + self.sections_per_chunk]
            chunk_num = chunk_idx + 1
            
            print(f"✍️  Writing chunk {chunk_num}/{total_chunks}: Sections {chunk_sections[0]['number']}-{chunk_sections[-1]['number']}")
            
            # Get relevant findings for this chunk
            relevant_findings = self._get_relevant_findings_for_chunk(
                chunk_sections, query_results, query_mappings
            )
            
            # Generate chunk content
            chunk_prompt = CHUNKED_SYNTHESIS_PROMPT.format(
                task=task,
                full_outline=self._format_outline(sections),
                sections_to_write=self._format_sections_to_write(chunk_sections),
                relevant_findings=relevant_findings,
                previous_sections=self._summarize_previous_sections(all_sections) if all_sections else "This is the beginning of the report."
            )
            
            # Get chunk response (text, not JSON)
            start_time = time.time()
            chunk_response, usage = self.supervisor_client.chat([
                {"role": "user", "content": chunk_prompt}
            ])
            timing["supervisor_time"] += time.time() - start_time
            
            # Track usage
            supervisor_calls.append({
                "input_tokens": getattr(usage, 'prompt_tokens', 0),
                "output_tokens": getattr(usage, 'completion_tokens', 0)
            })
            
            all_sections.append(chunk_response[0])
            
            # Log chunk synthesis
            conversation_log["conversation"].append({
                "role": "supervisor",
                "type": "chunked_synthesis_write",
                "chunk": chunk_num,
                "sections": [s['number'] for s in chunk_sections],
                "prompt": chunk_prompt[:500] + "...",  # Truncate for logging
                "response": chunk_response[0][:500] + "..."
            })
        
        # Step 3: Combine all sections
        final_answer = "\n\n".join(all_sections)
        
        print("✅ Chunked synthesis completed")
        
        return final_answer
    
    def _validate_section_plan(self, plan_json: Dict) -> Dict:
        """Validate the planning response has required fields"""
        required = ["outline_summary", "sections", "query_mappings"]
        if not all(field in plan_json for field in required):
            raise ValueError(f"Missing required fields in section plan: {plan_json.keys()}")
        
        # Validate sections structure
        for i, section in enumerate(plan_json["sections"]):
            # Check required fields
            required_section_fields = ["number", "title", "focus"]
            if not all(k in section for k in required_section_fields):
                raise ValueError(f"Invalid section structure at index {i}: missing required fields {required_section_fields}")
            
            # Validate subsections if present (new field)
            if "subsections" in section:
                if not isinstance(section["subsections"], list):
                    raise ValueError(f"Section {i} subsections must be a list")
                if len(section["subsections"]) < 2:
                    # Ensure at least 2 subsections per main section
                    print(f"Warning: Section {i} has only {len(section['subsections'])} subsections, should have 2-3")
            
            # Validate target_words if present (optional field)
            if "target_words" in section:
                if not isinstance(section["target_words"], int) or section["target_words"] <= 0:
                    print(f"Warning: Section {i} target_words should be a positive integer, got {section['target_words']}")
        
        return plan_json

    def _create_findings_summary(self, query_results: Dict[str, Dict]) -> str:
        """Create a concise summary of all findings for planning purposes"""
        summary_parts = []
        for i, (query, results) in enumerate(query_results.items(), 1):
            num_findings = len(results.get("summaries", []))
            summary_parts.append(f"Query {i} ({query}): {num_findings} relevant findings")
        return "\n".join(summary_parts)

    def _get_relevant_findings_for_chunk(
        self, 
        chunk_sections: List[Dict], 
        query_results: Dict[str, Dict], 
        query_mappings: Dict[str, List[int]]
    ) -> str:
        """Extract findings relevant to specific sections"""
        relevant_findings = []
        
        # Get all query indices for these sections
        relevant_query_indices = set()
        for section in chunk_sections:
            section_num = str(section['number'])
            if section_num in query_mappings:
                relevant_query_indices.update(query_mappings[section_num])
        
        # Collect findings from relevant queries
        all_queries = list(query_results.keys())
        for idx in relevant_query_indices:
            if 0 <= idx - 1 < len(all_queries):  # Convert 1-based to 0-based
                query = all_queries[idx - 1]
                findings = query_results[query].get("summaries", [])
                if findings:
                    relevant_findings.append(f"\n**From Query: {query}**")
                    for finding in findings:
                        if isinstance(finding, dict) and finding.get("answer") == "relevant":
                            relevant_findings.append(finding.get("explanation", ""))
                        elif isinstance(finding, str):
                            relevant_findings.append(finding)
        
        return "\n\n".join(relevant_findings)

    def _format_outline(self, sections: List[Dict]) -> str:
        """Format the full outline for context"""
        outline_parts = []
        for section in sections:
            outline_parts.append(f"{section['number']}. {section['title']} - {section['focus']}")
            if "subsections" in section:
                for subsection in section["subsections"]:
                    outline_parts.append(f"   {subsection}")
        return "\n".join(outline_parts)

    def _format_sections_to_write(self, sections: List[Dict]) -> str:
        """Format the sections to write for the prompt"""
        parts = []
        for section in sections:
            section_text = f"Section {section['number']}: {section['title']}\nFocus: {section['focus']}"
            if "subsections" in section and section["subsections"]:
                section_text += "\nSubsections to include:"
                for subsection in section["subsections"]:
                    section_text += f"\n  - {subsection}"
            parts.append(section_text)
        return "\n\n".join(parts)

    def _summarize_previous_sections(self, completed_sections: List[str]) -> str:
        """Create brief summary of already-written sections for context"""
        if not completed_sections:
            return "No previous sections."
        
        # Take the last 500 characters of the most recent section for context
        recent_section = completed_sections[-1]
        if len(recent_section) > 500:
            return f"Previous section ended with: ...{recent_section[-500:]}"
        return f"Previous section: {recent_section}"
    
    def _format_multi_query_findings(self, query_results: Dict[str, Dict]) -> str:
        """
        Format multi-query findings for supervisor synthesis.
        """
        findings_text = ""
        
        for i, (search_query, results) in enumerate(query_results.items(), 1):
            sub_task = results["sub_task"]
            summaries = results["summaries"]
            
            findings_text += f"\n**Query {i}: {search_query}**\n"
            findings_text += f"**Sub-task: {sub_task}**\n\n"
            
            if summaries:
                findings_text += "**Findings:**\n"
                for j, summary in enumerate(summaries, 1):
                    if summary.get("answer") == "relevant":
                        findings_text += f"{j}. {summary['explanation']}\n\n"
            else:
                findings_text += "No relevant findings for this query.\n\n"
            
            findings_text += "-" * 80 + "\n"
        
        return findings_text