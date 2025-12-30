import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.special import logsumexp
from typing import List, Dict, Optional, Tuple, Any, Union

from tqdm import tqdm


def _create_sglang_client(client_args: Dict[str, Any]):
    """
    Create SGLang client based on provided arguments.
    
    Args:
        client_args: Dictionary containing client configuration
        
    Returns:
        Instantiated SGLang client (Modal or Regular)
    """
    if (len(list(client_args.keys())) == 1 and "api_base_url" in client_args):
        # Modal client
        from mfactory.clients.sglang_modal import SGLangClient
    else:
        # Regular client
        from mfactory.clients.sglang import SGLangClient
    
    return SGLangClient.Config(**client_args).instantiate()

def prepare_chat(context: str, summary: str, query: Optional[str] = None, predicted_variable: str = "z") -> List[List[Dict[str, str]]]:
    """
    Prepare the chat prompt for the model.

    Args:
        context: The document context.
        summary: The summary text.
        query: The query text (optional).
        predicted_variable: Variable to predict ("z" or "y").

    Returns:
        list: A formatted chat prompt.
    """

    if predicted_variable == "z":
        # Q(Z|X)
        prompt = "Instruction: Please summarize the following text with respect to the question.\n\nHere is the text: {context}\n\nHere is the question: {query}\n\nHere is the queston specific summary: {summary}"
        # prompt = "Here is the text: {context}\n\nHere is the question: {query}\n\nInstruction: Please extract info from the text that is relevant to the question: {summary}"
    elif predicted_variable == "y":
        # Q(Y|Z)
        prompt = "Instruction: Please answer the question based on the summary of the context.\n\nHere is the question: {query}\n\nHere the summary of the context: {context}\n\nHere is the final answer: {summary}"
    format_prompt = prompt.format(summary=summary, context=context, query=query)

    chat = [[{"role": "user", "content": format_prompt}]]
    return chat


def generate_log_prob_q_query(chat: List[List[Dict[str, str]]], sglang_client: Any) -> Any:
    """
    Generate log probabilities from the LLM model.

    Args:
        chat: The formatted chat prompt.
        sglang_client: The LLM client.

    Returns:
        Object: The model response containing tokens and log probabilities.
    """
    # response = client.chat(chat, max_completion_tokens=1, echo=True)
    response = sglang_client.chat(chat, max_completion_tokens=1)
    return response


def compute_q_query(tokens: List[str], log_probs: List[float], predicted_variable: str = "z") -> float:
    """
    Compute the log probability q(x|z) from tokens and their log probabilities.
    This version does not normalize by token count to align with the MI formula.

    Args:
        tokens: List of tokens.
        log_probs: List of log probabilities corresponding to the tokens.
        predicted_variable: Variable to predict ("z" or "y").

    Returns:
        float: The computed log probability (sum, not average).
    """

    if predicted_variable == "y":  # Q(Y|Z)
        span = ["Here", " is", " the", " final", " answer", ":"]
    elif predicted_variable == "z":  # Q(Z|X)
        span = ["Here", " is", " the", " quest", "on", " specific", " summary", ":"]
    start_idx = None
    # Find the starting index where the span occurs.
    for i in range(len(tokens) - len(span) + 1):
        if tokens[i : i + len(span)] == span or tokens[i : i + len(span)] == tuple(span):
            start_idx = i + len(span)
            break

    if start_idx is None:
        # Fallback if the exact span isn't found.
        print("Span not found in tokens")
        print(tokens)
        print(span)
        print(start_idx)
        return sum(log_probs)  # Return sum without normalization

    # Extract the relevant log probabilities after the span.
    relevant_log_probs = log_probs[start_idx:]

    if not relevant_log_probs:
        return 0.0

    return sum(relevant_log_probs) if relevant_log_probs else 0.0

def calculate_log_prob_query(
    context: str, 
    summary: str, 
    query: Optional[str] = None, 
    sglang_client: Optional[Any] = None, 
    predicted_variable: str = "z"
) -> float:
    """
    Calculate the log probability q(x|z) of the document given the summary.

    Args:
        context: The document context.
        summary: The summary.
        query: The query text (optional).
        sglang_client: The LLM client.
        predicted_variable: Variable to predict ("z" or "y").

    Returns:
        float: The computed log probability, returns 0 if calculation fails.
    """
    if summary.strip() == "":
        return 0.0
    
    if sglang_client is None:
        raise ValueError("sglang_client must be provided")
        
    try:
        # Clean query by removing multiple choice options if present
        if query is not None:
            query = query.split("Choose the correct answer from the following options:\n")[0]
        
        # Prepare the chat for the model
        chat = prepare_chat(context, summary, query, predicted_variable)
        
        # Generate log probabilities from the model
        response = generate_log_prob_q_query(chat, sglang_client)
        
        if not response.samples or len(response.samples) == 0:
            print(f"Warning: No samples returned from model for predicted_variable='{predicted_variable}'")
            return 0.0
            
        tokens = response.samples[-1].tokens[:-1]
        log_probs = response.samples[-1].input_log_prob[:-1]
        
        # Compute the log probability over the relevant span
        log_prob = compute_q_query(tokens, log_probs, predicted_variable)
        return log_prob
        
    except AttributeError as e:
        print(f"Error: Invalid response structure from model - {str(e)}")
        return 0.0
    except IndexError as e:
        print(f"Error: Token/probability indexing failed - {str(e)}")
        return 0.0
    except ValueError as e:
        print(f"Error: Invalid input parameters - {str(e)}")
        return 0.0
    except Exception as e:
        print(f"Unexpected error calculating log probability: {str(e)}")
        print(f"Context length: {len(context)}, Summary length: {len(summary)}")
        return 0.0


def process_log_q_query(args: Tuple[str, List[str], List[str], Dict[str, Any], str]) -> List[float]:
    """Compute log q(z_ij | x_k) for all k given fixed z_ij."""
    z_ij, contexts, queries, sglang_client_args, predicted_variable = args
    
    # Use consolidated client creation
    sglang_client = _create_sglang_client(sglang_client_args)

    return [
        calculate_log_prob_query(
            x_k, z_ij, queries[i], sglang_client, predicted_variable
        )
        for i, x_k in enumerate(contexts)
    ]


# Define the worker function for multiprocessing.
# This function will be executed in separate processes.
# It needs sglang_client_args to instantiate its own SGLangClient.
# It also needs access to the globally defined/imported `calculate_log_prob_query` and `logsumexp`.
# Assumes `calculate_log_prob_query` takes `predicted_variable` as an argument.
def _calculate_mi_term_worker(worker_args: Tuple[str, str, Optional[str], Dict[str, Any], List[float], str]) -> float:
    """
    Worker function for multiprocessing MI term calculation.
    
    Args:
        worker_args: Tuple containing (context_val, z_val, query_val, client_args, log_q_values, predicted_variable)
        
    Returns:
        float: The computed MI term (log_num - log_denom)
    """
    context_val, z_val, query_val, local_sglang_client_args, log_q_values_for_z, worker_predicted_variable = worker_args

    # Use consolidated client creation
    local_sglang_client = _create_sglang_client(local_sglang_client_args)
    
    # Calculate log_num: log[q(z_ij | x_i)]
    log_num_val = calculate_log_prob_query(
        context_val, z_val, query_val, local_sglang_client, worker_predicted_variable
    )
    
    # Calculate log_denom: log[∑_k q(z_ij | x_k)]
    log_denom_val = logsumexp(log_q_values_for_z) # log_q_values_for_z is log_q_all[idx]

    del local_sglang_client
    
    return log_num_val - log_denom_val
    
def compute_mi(
    contexts: List[str],
    summaries_dict: Dict[int, List[str]],
    queries: Optional[List[str]] = None,
    sglang_client_args: Optional[Dict[str, Any]] = None,
    num_processes: int = 4,
    predicted_variable: str = "z",
) -> float:
    """
    Compute mutual information according to the formula:
        I(X; Z) ≈ (1/nm) ∑_{i,j} log[ q(z_ij | x_i) / ∑_{k,ℓ} q(z_ij | x_k) ] + log(nm)
    
    Args:
        contexts: List of document contexts.
        summaries_dict: Dictionary mapping context indices to lists of summaries.
        queries: Optional list of queries corresponding to contexts.
        sglang_client_args: Configuration arguments for SGLang client.
        num_processes: Number of processes for multiprocessing.
        predicted_variable: Variable to predict ("z" for Q(Z|X) or "y" for Q(Y|Z)).
        
    Returns:
        float: The computed mutual information value.
    """
    if sglang_client_args is None:
        raise ValueError("sglang_client_args must be provided")

    n = len(contexts)
    m = len(next(iter(summaries_dict.values())))
    nm = n * m

    # Flatten z_ij: list of (i, j, z_ij)
    indexed_z = [(i, j, summaries_dict[i][j]) for i in range(n) for j in range(m)]

    # Precompute log_q(z_ij | x_k) for all i,j,k
    args = [
        (z_ij, contexts, queries, sglang_client_args, predicted_variable)
        for (_, _, z_ij) in indexed_z
    ]
    with Pool(processes=min(num_processes, len(args))) as pool:
        log_q_all = list(
            tqdm(
                pool.imap(process_log_q_query, args),
                total=len(args),
                desc="Computing log probabilities",
            )
        )

    # Prepare arguments for the worker function
    mi_term_args_list = []
    for idx, (i, j, z_ij_val) in enumerate(indexed_z): # Using more descriptive var names
        current_query_for_worker = queries[i] if queries is not None else None
        # Each element in mi_term_args_list is a tuple of arguments for _calculate_mi_term_worker
        mi_term_args_list.append(
            (contexts[i], z_ij_val, current_query_for_worker, sglang_client_args, log_q_all[idx], predicted_variable)
        )

    # Use a Pool for multiprocessing to calculate MI terms
    mi_sum = 0.0
    # num_processes is an argument to the main compute_mi function
    # tqdm and Pool are assumed to be imported at the module level
    with Pool(processes=min(num_processes, len(mi_term_args_list))) as pool:
        # Using imap to allow for progress tracking with tqdm, similar to log_q_all computation
        results = list(
            tqdm(
                pool.imap(_calculate_mi_term_worker, mi_term_args_list),
                total=len(mi_term_args_list),
                desc="Calculating MI terms",
            )
        )
        mi_sum = sum(results)
        
    # The final MI calculation using the sum of terms.
    # np.log is assumed to be available via numpy import.
    return float(
        mi_sum / nm + np.log(n)
    )
