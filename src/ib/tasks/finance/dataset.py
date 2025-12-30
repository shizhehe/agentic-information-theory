import os
import random
from pathlib import Path
import requests

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from pydrantic import ObjectConfig


def extract_text_from_pdf_url(pdf_url):
    import pymupdf

    # Fetch the PDF content from the URL
    response = requests.get(pdf_url)
    response.raise_for_status()  # Ensure the request was successful

    # Open the PDF from the fetched content
    pdf_data = response.content
    doc = pymupdf.open(stream=pdf_data, filetype="pdf")

    all_text = ""
    pages = []
    # Iterate over each page
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load page by index
        text = page.get_text()  # Extract text from page
        all_text += text
        pages.append(text)
        # print(f"Page {page_num + 1}:\n{text}\n")

    # Close the document
    doc.close()
    return all_text, pages


def truncate_dataset_pdfs(df: pd.DataFrame, max_tokens, tokenizer):
    """
    Truncate PDFs for each row in the dataset based on tokenizer and max_tokens using apply.
    """
    if not max_tokens or tokenizer is None:
        print("Skipping truncation: max_tokens or tokenizer not provided.")
        return df

    print(f"Truncating dataset with max_tokens: {max_tokens}")

    def truncate_text(text):
        if pd.isna(text) or not text:
            return text
        # Tokenize the text
        tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
        # Decode back to text (this will be truncated)
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
        return truncated_text

    if "pdf_text" not in df.columns:
        print("'pdf_text' column not found in DataFrame. Skipping truncation.")
        return df

    tqdm.pandas(desc="Truncating PDFs")
    df["pdf_text"] = df["pdf_text"].progress_apply(truncate_text)

    return df


def process_dataset_pdfs(df: pd.DataFrame):
    """
    Process PDFs for each row in the dataset
    Adds a 'pdf_text' column with the extracted text
    """

    # Process each URL in the dataset
    results = []
    for _, row in tqdm(
        df.iterrows(), desc="Processing PDFs", total=len(df)
    ):  # Replace 'pdf_url' with your column name
        url = row["doc_link"]
        try:
            text, pages = extract_text_from_pdf_url(url)
            results.append(
                {
                    **row,
                    "pdf_text": text,
                    "pages": pages,
                }
            )
        except Exception as e:
            print(f"Error processing {url}: {e}")

    return pd.DataFrame(results)


def load_finance(
    force_reprocess=False,
    max_context_length=None,
    tokenizer_model=None,
    evidence_only=False,
):
    from datasets import load_dataset

    tokenizer = None
    if tokenizer_model:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model, max_length=max_context_length
        )

    dataset = load_dataset(
        "PatronusAI/financebench", split="train", trust_remote_code=True
    )

    dataset_dir = Path(dataset.cache_files[0]["filename"]).parent

    # download the pdfs and process them
    # since this takes a while, we cache the result to the huggingface directory
    path = os.path.join(dataset_dir, "bench_with_pdfs.feather")

    # Check if we should use cached file AND that it's not empty
    if os.path.exists(path) and not force_reprocess:
        print("Loading cached dataset")
        df = pd.read_feather(path)
        if len(df) > 0:  # Make sure the DataFrame isn't empty
            # df = tokenize_full_texts(df, tokenizer)
            # truncate pdf around the evidence if evidence_only is True
            if evidence_only:
                df = truncate_dataset_pdfs_around_evidence(df, tokenizer)
            else:
                df = truncate_dataset_pdfs(df, max_context_length, tokenizer)
            print(
                "Number of unique contexts: ",
                len(df.doc_name.unique()),
            )
            return df
        # Otherwise fall through to regenerate the data

    # Either cache doesn't exist, is empty, or we're forcing reprocessing
    df = dataset.to_pandas()
    df = process_dataset_pdfs(df)
    df.to_feather(path)

    # df = tokenize_full_texts(df, tokenizer)

    # truncate pdf around the evidence if evidence_only is True
    if evidence_only:
        df = truncate_dataset_pdfs_around_evidence(df, tokenizer)
    else:
        df = truncate_dataset_pdfs(df, max_context_length, tokenizer)
    return df


def tokenize_full_texts(df: pd.DataFrame, tokenizer):
    from tqdm.auto import tqdm

    tqdm.pandas(desc="Tokenizing PDF texts")
    df["full_pdf_tokens"] = df["pdf_text"].progress_apply(lambda x: tokenizer.encode(x))
    return df


def get_evidence_texts(evidence_list, full_text_tokens, max_tokens, tokenizer):
    pdf_text = ""
    evidence_found = True
    for evidence in evidence_list:
        # tokenize the evidence text
        evidence_text = evidence["evidence_text"]
        evidence_tokens = tokenizer.encode(evidence_text)[1:]
        # print(f"Evidence tokens: len {len(evidence_tokens)}, {evidence_tokens}")
        # print(f"Full text tokens: len {len(full_text_tokens)}, {full_text_tokens}")
        # find start of evidence, full tokens must match
        evidence_start = None
        for i in range(len(full_text_tokens) - len(evidence_tokens) + 1):
            if (
                full_text_tokens[i : i + len(evidence_tokens) - 1]
                == evidence_tokens[:-1]
            ):
                evidence_start = i
                break
        if evidence_start is None:
            # print(f"Evidence not found in full text, defaulting to full text")
            evidence_start = 0
            evidence_end = len(full_text_tokens)
            evidence_found = False
        else:
            # print(f"Evidence found in full text, truncating")
            evidence_start = max(0, evidence_start - max_tokens)
            evidence_end = min(
                len(full_text_tokens),
                evidence_start + len(evidence_tokens) + max_tokens,
            )

        evidence_text = evidence_text[evidence_start:evidence_end]
        # decode the evidence text
        evidence_text = tokenizer.decode(evidence_tokens)
        # print(f"Evidence text: {evidence_text}")
        pdf_text += evidence_text
    return {"pdf_text": pdf_text, "evidence_found": evidence_found}


def truncate_dataset_pdfs_around_evidence_literal(
    df: pd.DataFrame, max_tokens, tokenizer
):
    """
    Extract tokens around the evidence.
    """

    if "full_pdf_tokens" not in df.columns:
        print("'full_pdf_tokens' column not found in DataFrame. Manually tokenizing.")
        tqdm.pandas(desc="Tokenizing PDF texts")
        df["full_pdf_tokens"] = df["pdf_text"].progress_apply(
            lambda x: tokenizer.encode(x)
        )

    if "pdf_text" not in df.columns:
        print("'pdf_text' column not found in DataFrame. Skipping truncation.")
        return df

    # rename "pdf_text" to "full_pdf_text"
    df.rename(columns={"pdf_text": "full_pdf_text"}, inplace=True)

    # add new columns "pdf_text" and "evidence_found" that are the evidence text + max_tokens around the evidence
    # use progress_apply to get the evidence text for each row with a progress bar
    tqdm.pandas(desc="Extracting Evidence Context")
    df[["pdf_text", "evidence_found"]] = df.progress_apply(
        lambda row: get_evidence_texts(
            row["evidence"], row["full_pdf_tokens"], tokenizer
        ),
        axis=1,
        result_type="expand",
    )
    return df


def get_max_context_tokens(tokenizer):
    """
    Leave room for the prompt and the answer.
    """
    return 32000 - 10500


def get_evidence_pages(evidence_list, full_pdf_pages, tokenizer, max_context_tokens):
    """
    Extract the pages around the evidence.
    """
    evidence_pages = {}
    evidence_target_page = evidence_list[0]["evidence_page_num"]

    total_pages = len(full_pdf_pages)
    prev_page_idx = evidence_target_page
    next_page_idx = evidence_target_page
    prev_possible = prev_page_idx > 0
    next_possible = next_page_idx < total_pages

    current_tokens = 0
    while prev_possible or next_possible:
        page = full_pdf_pages[prev_page_idx - 1]
        tokens = tokenizer.encode(page)
        page_added = False
        if prev_possible:
            if current_tokens < max_context_tokens:
                evidence_pages[prev_page_idx - 1] = tokens
                current_tokens += len(tokens)
                prev_page_idx -= 1
                prev_possible = prev_page_idx >= 1
                page_added = True
            else:
                prev_possible = False

        if next_possible:
            page = full_pdf_pages[next_page_idx]
            tokens = tokenizer.encode(page)
            if current_tokens < max_context_tokens:
                evidence_pages[next_page_idx] = tokens
                current_tokens += len(tokens)
                next_page_idx += 1
                next_possible = next_page_idx < total_pages
                page_added = True
            else:
                next_possible = False

        if not page_added:
            break

    sorted_evidence_pages = sorted(evidence_pages.keys())
    evidence_pages_list = []
    for page_idx in sorted_evidence_pages:
        evidence_pages_list.extend(evidence_pages[page_idx])
    evidence_pages_tokens = evidence_pages_list[:max_context_tokens]
    pdf_text = tokenizer.decode(evidence_pages_tokens, skip_special_tokens=True)
    return {
        "pdf_text": pdf_text,
        "evidence_found": 1,
        "evidence_pages": sorted_evidence_pages,
    }


def truncate_dataset_pdfs_around_evidence(df: pd.DataFrame, tokenizer):
    """
    Extract pages around the evidence.
    For our experiments, skip evidence if it's found at multiple places in the document.
    (where len(evidence) > 1)
    """
    df = df[df["evidence"].apply(len) == 1]

    if "pdf_text" not in df.columns:
        print("'pdf_text' column not found in DataFrame. Skipping truncation.")
        return df

    # rename "pdf_text" to "full_pdf_text"
    # df.rename(columns={"pdf_text": "full_pdf_text"}, inplace=True)

    max_context_tokens = get_max_context_tokens(tokenizer)
    print(
        f"Max context tokens around evidence: {max_context_tokens} for tokenizer {tokenizer.name_or_path}"
    )

    # add new columns "pdf_text" and "evidence_found" that are the evidence text + max_tokens around the evidence
    # use progress_apply to get the evidence text for each row with a progress bar
    tqdm.pandas(desc="Extracting Evidence Pages")
    df[["pdf_text", "evidence_found", "evidence_pages"]] = df.progress_apply(
        lambda row: get_evidence_pages(
            row["evidence"], row["pages"], tokenizer, max_context_tokens
        ),
        axis=1,
        result_type="expand",
    )
    return df


class FinanceChunkDataset(Dataset):

    class Config(ObjectConfig):
        _pass_as_config = True
        chunk_size: int = 512
        row_idx: int = 0  # what row in the dataset to use

    def __init__(
        self,
        config: Config,
        tokenizer: AutoTokenizer,
    ):
        self.config = config
        self.df = load_finance()
        self.row = self.df.iloc[self.config.row_idx].to_dict()
        self.pdf_tokens = torch.tensor(tokenizer.encode(self.row["pdf_text"]))
        self.reload()

    def reload(self):
        random_offset = random.randint(0, self.config.chunk_size)
        wrapped_pdf_tokens = self.pdf_tokens.roll(random_offset)
        self.chunks = [
            wrapped_pdf_tokens[i : i + self.config.chunk_size]
            for i in range(0, len(wrapped_pdf_tokens), self.config.chunk_size)
        ]

    def __getitem__(self, idx):
        input_ids = self.chunks[idx]
        return input_ids, input_ids

    def __len__(self):
        return len(self.chunks)
