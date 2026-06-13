"""Official LongBench-V1 prompt and generation settings.

The prompt strings and per-task maximum generation lengths mirror
`THUDM/LongBench/LongBench/config/dataset2prompt.json` and
`dataset2maxlen.json` for the English Table 1 tasks.
"""

from __future__ import annotations

from typing import Any

from .longbench_metrics import normalize_dataset_name


DATASET2PROMPT: dict[str, str] = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story asconcisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        " Answer the question based on the above article as concisely as you can, using a single phrase or "
        "sentence if possible. If the question cannot be answered based on the information in the article, "
        "write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
        "\"unanswerable\". Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and do not "
        "output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
    ),
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output any other words. "
        "The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. "
        "In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. The output format should only contain "
        "the number, such as 1, 2, 3, and so on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n"
        "{context}\n\nThe following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. The answer format must be like "
        "\"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}


DATASET2MAX_NEW_TOKENS: dict[str, int] = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}


NO_CHAT_TEMPLATE_DATASETS = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}


def dataset_name_from_row(row: dict[str, Any], fallback: str | None = None) -> str:
    name = row.get("task") or row.get("dataset") or fallback or ""
    return normalize_dataset_name(str(name))


def _strip_prefix(value: str, prefix: str) -> str:
    if value.lstrip().lower().startswith(prefix.lower()):
        leading = len(value) - len(value.lstrip())
        return value[leading + len(prefix) :].strip()
    return value.strip()


def row_for_official_prompt(row: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    values = dict(row)
    if not values.get("input"):
        input_text = str(values.get("question") or "")
        if dataset_name in {"narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"}:
            input_text = _strip_prefix(input_text, "Question:")
        elif dataset_name == "qmsum":
            input_text = _strip_prefix(input_text, "Query:")
        values["input"] = input_text
    return values


def build_longbench_prompt(row: dict[str, Any], dataset: str | None = None) -> str:
    dataset_name = normalize_dataset_name(dataset or dataset_name_from_row(row))
    template = DATASET2PROMPT.get(dataset_name)
    if template is None:
        raise KeyError(f"No LongBench prompt configured for dataset {dataset_name!r}")
    return template.format(**row_for_official_prompt(row, dataset_name))


def max_new_tokens_for(dataset: str, default: int = 64) -> int:
    return DATASET2MAX_NEW_TOKENS.get(normalize_dataset_name(dataset), default)


def should_apply_chat_template(dataset: str, mode: str) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return True
    if mode != "auto":
        raise ValueError(f"unknown chat template mode: {mode}")
    return normalize_dataset_name(dataset) not in NO_CHAT_TEMPLATE_DATASETS
