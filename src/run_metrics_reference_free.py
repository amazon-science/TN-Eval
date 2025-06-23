# All the imports here
import argparse
import copy
import json
import logging
import multiprocessing
import os.path
from functools import partial
from multiprocessing import Pool

import numpy as np
from nltk import sent_tokenize
from tqdm import tqdm

from constant import SOAP_SECTIONS, CHECKBOX_MAPPING
from generate_soap_note import (
    predict_claude3,
    predict_llama3,
    predict_mistral_v2,
    MISTRAL_LARGE_V2_MODEL_ID,
)

rubric_prompt_completeness = """
Below is a behavioral therapy progress note segment. \
The rubric item outlines one of the necessary components for the note. \
Verify if the rubric item presents in the progress note segment. 

## Note Segment
{note_segment}

## Rubric Item (an item that should present in the note segment)
{rubric_item}

Does the note segment contain the rubric item? Response in [Yes, No] with no other content:
""".strip()

rubric_prompt_likert_completeness = """
Below is a behavioral therapy conversation along with a corresponding progress note segment. \
The rubrics outline the necessary components for the note. Based on the conversation and rubrics, \
evaluate the completeness of the note segment.

## Conversation
{conversation}

## Note Segment
{note_segment}

## Rubrics (a list of items that should present in the note segment)
{rubrics}

## Rating Codebook
1: The note segment is missing most of the key information from the conversation.
2: The note segment includes some important details but is significantly incomplete.
3: The note segment contains a moderate amount of important information.
4: The note segment captures most of the key information from the conversation.
5: The note segment comprehensively captures all the key information.

Using the 1 to 5 scale from the rating codebook, rate the completeness of the note segment. \
Output only the rating [1, 2, 3, 4, 5]:
""".strip()

rubric_prompt_conciseness = """
Below is a sentence from a behavioral therapy progress note. \
The rubrics outlines the necessary components for the note. \
Verify if the note sentence fit in one of the rubric items.

## Note Sentence
{note_sentence}

## Rubrics (a list of items that should present in the note segment)
{rubrics}

Does the note sentence fit in one of the rubric items? Response in [Yes, No] with no other content:
""".strip()

rubric_prompt_likert_conciseness = """
Below is a behavioral therapy conversation along with a corresponding progress note segment. \
The rubrics outline the necessary components for the note. Based on the conversation and rubrics, \
evaluate the conciseness of the note segment.

## Conversation
{conversation}

## Note Segment
{note_segment}

## Rubrics (a list of items that should present in the note segment)
{rubrics}

## Rating Codebook
1: The note segment includes substantial non-important information that detracts from the main points.
2: The note segment includes non-important information that needs to be reduced.
3: The note segment includes some non-important information but does not heavily detract from the main points.
4: The note segment includes minor non-critical information.
5: The note segment includes no non-important information, making it concise and focused.

In the scale of 1 to 5, rate the conciseness of the note segment following the rating codebook. \
Output only the rating [1, 2, 3, 4, 5]:
""".strip()

rubric_prompt_likert_faithfulness = """
Below is a behavioral therapy conversation along with a corresponding progress note segment. \
Verify the faithfulness of the note segment based on the conversation.

## Conversation
{conversation}

## Note Segment
{note_segment}

## Rating Codebook
1: The note segment contains significant inaccuracies or false information.
2: The note segment contains several inaccuracies or false information.
3: The note segment may contain some inaccuracies or false information.
4: The note segment contains minor non-critical inaccuracies or false information.
5: The note segment contains no inaccuracies or false information.

In the scale of 1 to 5, rate the faithfulness of the note segment following the rating codebook. \
Output only the rating [1, 2, 3, 4, 5]:
""".strip()

LIKERT_PROMPTS = {
    "likert_completeness": rubric_prompt_likert_completeness,
    "likert_conciseness": rubric_prompt_likert_conciseness,
    "likert_faithfulness": rubric_prompt_likert_faithfulness,
}


def pre_process_options():
    options = {section: [] for section in SOAP_SECTIONS}
    for key, value in CHECKBOX_MAPPING.items():
        section = key.split("-")[0]
        options[section].append(value)
    return options


def parse_likert(pred):
    try:
        score = int(pred.strip())
        if score < 1 or score > 5:
            raise ValueError
        return score
    except:
        logging.error(f"not a likert score: {pred}")
        for item in range(1, 6):
            if str(item) in pred:
                return item
        return 3


def parse_yes_no(pred):
    if "yes" in pred.lower():
        return 1
    elif "no" in pred.lower():
        return 0
    else:
        logging.error(f"not a yes/no: {pred}")
        return 0


def format_cache_key(task, evaluator_name):
    return f"{evaluator_name}:::{task['prompt']}"


def handle_eval_task(task, function_name, cache, evaluator_name):
    cached_result = cache.get(format_cache_key(task, evaluator_name), None)
    if cached_result:
        return cached_result
    else:
        pred = function_name(task["prompt"])
    if "likert" in task["task"]:
        return parse_likert(pred)
    else:
        return parse_yes_no(pred)


def run_evaluation(
    conversations, predictions, rubric_list, evaluator_name, evaluator_fn, cache, cache_path
):
    metrics = [{"rubric_completeness_raw": [], "rubric_conciseness_raw": []} for _ in predictions]

    tasks = []

    # setup Likert
    for likert_name, likert_prompt in LIKERT_PROMPTS.items():
        for idx, p in enumerate(predictions):
            assert "{conversation}" in likert_prompt
            assert "{note_segment}" in likert_prompt
            if "faithful" not in likert_name:
                assert "{rubrics}" in likert_prompt
            tasks.append(
                {
                    "task": likert_name,
                    "idx": idx,
                    "prompt": likert_prompt.replace("{conversation}", conversations[idx])
                    .replace("{note_segment}", p)
                    .replace("{rubrics}", "\n".join(rubric_list)),
                }
            )

    # setup rubric completeness
    for idx, p in enumerate(predictions):
        for rubric_item in rubric_list:
            tasks.append(
                {
                    "task": "rubric_completeness",
                    "idx": idx,
                    "prompt": rubric_prompt_completeness.format(
                        note_segment=p, rubric_item=rubric_item
                    ),
                }
            )

    # setup rubric conciseness
    for idx, p in enumerate(predictions):
        for sent in sent_tokenize(p):
            tasks.append(
                {
                    "task": "rubric_conciseness",
                    "idx": idx,
                    "prompt": rubric_prompt_conciseness.format(
                        conversation=conversations[idx],
                        note_sentence=sent,
                        rubrics="\n".join(rubric_list),
                    ),
                }
            )

    with multiprocessing.Manager() as manager:
        shared_dict = manager.dict(copy.deepcopy(cache))
        with Pool(5) as p:
            outs = list(
                tqdm(
                    p.imap(
                        partial(
                            handle_eval_task,
                            function_name=evaluator_fn,
                            cache=shared_dict,
                            evaluator_name=evaluator_name,
                        ),
                        tasks,
                    ),
                    total=len(tasks),
                )
            )

    for task, out in zip(tasks, outs):
        cache[format_cache_key(task, evaluator_name)] = out

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    for task, out in zip(tasks, outs):
        if "likert" in task["task"]:
            metrics[task["idx"]][task["task"]] = out
        elif task["task"] == "rubric_completeness":
            metrics[task["idx"]]["rubric_completeness_raw"].append(out)
        elif task["task"] == "rubric_conciseness":
            metrics[task["idx"]]["rubric_conciseness_raw"].append(out)
        else:
            assert 0

    for item in metrics:
        assert item["rubric_completeness_raw"]
        item["rubric_completeness"] = float(np.mean(item["rubric_completeness_raw"]))
        if not item["rubric_conciseness_raw"]:
            item["rubric_conciseness"] = 0
        else:
            assert item["rubric_completeness_raw"]
            item["rubric_conciseness"] = float(np.mean(item["rubric_conciseness_raw"]))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", type=str, help="path to a json note file.")
    parser.add_argument("--output", type=str, help="path to save output json file.")
    parser.add_argument("--cache_path", default="data/eval_cache.json", help="path to save cache.")
    args = parser.parse_args()

    cache = {}
    cache_path = args.cache_path
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    with open(args.note) as f:
        data_dict = json.load(f)
        data_dict = data_dict[list(data_dict.keys())[0]]

    rubrics = pre_process_options()

    for evaluator_name, evaluator_fn in [
        ("mistral_large_v2", partial(predict_mistral_v2, model_id=MISTRAL_LARGE_V2_MODEL_ID)),
        ("sonnet3", predict_claude3),
        ("llama31_70B", predict_llama3),
    ]:
        for section in SOAP_SECTIONS:
            print("Evaluator:", evaluator_name, "Section", section)
            conversations = [item["conversation"] for convo_id, item in data_dict.items()]
            predictions = [item["soap_note"][section] for convo_id, item in data_dict.items()]
            metrics = run_evaluation(
                conversations,
                predictions,
                rubrics[section],
                evaluator_name,
                evaluator_fn,
                cache,
                cache_path,
            )
            for idx, (convo_id, item) in enumerate(data_dict.items()):
                if f"metrics_{evaluator_name}" not in item:
                    item[f"metrics_{evaluator_name}"] = {}
                item[f"metrics_{evaluator_name}"][section] = metrics[idx]

        with open(args.output, "w") as f:
            json.dump(data_dict, f, indent=2)


if __name__ == "__main__":
    main()
