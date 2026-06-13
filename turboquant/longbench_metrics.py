"""LongBench-style scoring utilities for reproduction runs."""

from __future__ import annotations

import difflib
import re
import string
from collections import Counter
from typing import Any, Callable

try:
    from fuzzywuzzy import fuzz
except ImportError:  # pragma: no cover
    fuzz = None

try:
    from rouge import Rouge
except ImportError:  # pragma: no cover
    Rouge = None

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover
    rouge_scorer = None


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def token_f1(prediction_tokens: list[str], ground_truth_tokens: list[str]) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    return token_f1(pred_tokens, gt_tokens)


def rouge_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    if Rouge is not None:
        try:
            scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
            return float(scores["rouge-l"]["f"])
        except Exception:
            return 0.0
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        try:
            return float(scorer.score(ground_truth, prediction)["rougeL"].fmeasure)
        except Exception:
            return 0.0
    raise RuntimeError("rouge package is required for official LongBench summarization metrics")


def classification_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    all_classes = kwargs["all_classes"]
    matches = [class_name for class_name in all_classes if class_name in prediction]
    matches = [match for match in matches if not (match in ground_truth and match != ground_truth)]
    if ground_truth in matches:
        return 1.0 / len(matches)
    return 0.0


def retrieval_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1 for number in numbers if str(number) == str(ground_truth_id)) / len(numbers)


def count_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1 for number in numbers if str(number) == str(ground_truth)) / len(numbers)


def code_sim_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    if fuzz is not None:
        return fuzz.ratio(candidate, ground_truth) / 100
    return difflib.SequenceMatcher(None, candidate, ground_truth).ratio()


DATASET_TO_METRIC: dict[str, Callable[..., float]] = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


TABLE1_CATEGORIES: dict[str, list[str]] = {
    "SingleQA": ["narrativeqa", "qasper", "multifieldqa_en"],
    "MultiQA": ["hotpotqa", "2wikimqa", "musique"],
    "Summarization": ["gov_report", "qmsum", "multi_news"],
    "Few shot": ["trec", "triviaqa", "samsum"],
    "Synthetic": ["passage_retrieval_en", "passage_count"],
    "Code": ["lcc", "repobench-p"],
}

TREC_CLASSES = [
    "Description of something",
    "Entity",
    "Expression abbreviated",
    "Individual",
    "Group or organization of person",
    "Title of a person",
    "Description of a person",
    "Reason",
    "Definition of something",
    "Animal",
    "Body",
    "Color",
    "Creative piece",
    "Currency name",
    "Disease and medicine",
    "Event",
    "Food",
    "Musical instrument",
    "Language",
    "Letter",
    "Other entity",
    "Plant",
    "Product",
    "Religion",
    "Sport",
    "Element and substance",
    "Symbol",
    "Techniques and method",
    "Equivalent term",
    "Vehicle",
    "Word with a special property",
    "Numerical value",
    "Code",
    "Number of something",
    "Date",
    "Distance, linear measure",
    "Price",
    "Order, rank",
    "Lasting time of somethin",
    "Percent, fraction",
    "Speed",
    "Temperature",
    "Size, area and volume",
    "Weight",
    "Location",
    "City",
    "Country",
    "Mountain",
    "Other location",
    "State",
]


def normalize_dataset_name(name: str | None) -> str:
    if not name:
        return ""
    if name.endswith("_e"):
        return name[:-2]
    return name


def score_prediction(dataset: str, prediction: str, answers: list[str], all_classes: list[str] | None = None) -> float:
    dataset = normalize_dataset_name(dataset)
    metric = DATASET_TO_METRIC.get(dataset)
    if metric is None:
        return float("nan")
    if dataset == "trec" and all_classes is None:
        all_classes = TREC_CLASSES
    if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        prediction = prediction.lstrip("\n").split("\n")[0]
    best = 0.0
    for answer in answers:
        best = max(best, metric(prediction, answer, all_classes=all_classes))
    return best


def category_for_dataset(dataset: str) -> str | None:
    dataset = normalize_dataset_name(dataset)
    for category, datasets in TABLE1_CATEGORIES.items():
        if dataset in datasets:
            return category
    return None
