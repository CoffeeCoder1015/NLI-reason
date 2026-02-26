import argparse
import json
import multiprocessing as mp
import os
import time
import torch
from datasets import load_dataset
mp.set_start_method("spawn", force=True)

from generator import pipeline_config

def build_explain_prompt(example):
    premise = example['premise']
    hypothesis = example['hypothesis']
    content = f"""Explain step-by-step how to determine the NLI relationship between:

The three possible NLI relationships are:
- Entailment: the hypothesis is definitely true given the premise
- Neutral: the hypothesis could be true, but is not necessarily true given the premise
- Contradiction: the hypothesis is definitely false given the premise
The answer is guaranteed to be one of the three.

Premise: {premise}
Hypothesis: {hypothesis}
"""
    prompt = [{"role": "user", "content": content}]
    example["prompt"] = prompt
    return example

def build_gold_prompt(example):
    premise = example['premise']
    hypothesis = example['hypothesis']
    label = example['label']
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    label_text = label_map.get(label, "unknown")
    content = f"Explain why the correct NLI relationship is {label_text}.\nPremise: {premise}\nHypothesis: {hypothesis}"
    prompt = [{"role": "user", "content": content}]
    example["prompt"] = prompt
    return example


def shard_data(prompts, labels, metadata, num_shards, rank):
    start = len(prompts) * rank // num_shards
    end = len(prompts) * (rank + 1) // num_shards
    return prompts[start:end], labels[start:end], metadata[start:end]


def run_worker(rank, prompts, labels, metadata, pipeline_name):
    from generator import worker
    worker(rank, prompts, labels, metadata, pipeline_name)


def main():
    parser = argparse.ArgumentParser(description="NLI Evaluation with Multi-GPU Support")
    parser.add_argument(
        "--model",
        type=str,
        default="liquid",
        choices=list(pipeline_config.keys()),
        help="Model pipeline to use"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Number of examples to use (default: full dataset)"
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="gold",
        choices=["gold", "no_hint"],
        help="Prompt style: 'gold' gives correct label, 'no_hint' gives no hint"
    )
    parser.add_argument(
        "--from_tail",
        action="store_true",
        help="Sample from tail end of dataset (last 15k samples)"
    )
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available. This script requires CUDA GPUs.")
    print(f"Detected {num_gpus} GPUs.")

    model_name = pipeline_config[args.model]["model"]
    print(f"Using model: {model_name}")

    print("Loading dataset...")
    dataset = load_dataset("snli", split="train")

    total_samples = len(dataset)
    print(f"Total SNLI train samples: {total_samples}")

    if args.from_tail:
        subset_size = 50_000
        start_idx = max(0, total_samples - subset_size)
        dataset = dataset.select(range(start_idx, total_samples))
        print(f"Using tail end: samples {start_idx} to {total_samples} ({len(dataset)} samples)")
    elif args.subset is not None:
        dataset = dataset.select(range(min(args.subset, len(dataset))))
        print(f"Using subset of {len(dataset)} examples")

    print("Building prompts...")
    if args.prompt_style == "gold":
        dataset = dataset.map(build_gold_prompt)
        output_suffix = "gold"
    else:
        dataset = dataset.map(build_explain_prompt)
        output_suffix = "no_hint"

    classification_map = ["entailment", "neutral", "contradiction"]
    labels_raw = dataset["label"]
    word_labels = [classification_map[i] for i in labels_raw]
    reduced_prompts = dataset["prompt"]

    metadata = [
        {"premise": p, "hypothesis": h, "label": l}
        for p, h, l in zip(dataset["premise"], dataset["hypothesis"], word_labels)
    ]

    print(f"Total samples: {len(reduced_prompts)}")

    processes = []

    print(f"Spawning {num_gpus} workers...")
    for rank in range(num_gpus):
        prompts_shard, labels_shard, metadata_shard = shard_data(
            reduced_prompts, word_labels, metadata, num_gpus, rank
        )

        p = mp.Process(
            target=run_worker,
            args=(rank, prompts_shard, labels_shard, metadata_shard, args.model)
        )
        p.start()
        processes.append(p)

    print("Collecting results...")
    all_responses = []
    all_labels = []
    all_metadata = []

    for expected_rank in range(num_gpus):
        file_path = f"shard-{expected_rank}.json"
        poll_interval = 5
        max_poll_interval = 60
        
        while not os.path.exists(file_path):
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 2, max_poll_interval)
        
        with open(file_path) as f:
            result_data = json.load(f)
        all_responses.extend(result_data["responses"])
        all_labels.extend(result_data["labels"])
        all_metadata.extend(result_data["metadata"])
        print(f"Collected results from shard-{expected_rank}.json")

    reasoning_data = []
    for i in range(len(all_responses)):
        entry = {
            "premise": all_metadata[i]["premise"],
            "hypothesis": all_metadata[i]["hypothesis"],
            "label": all_metadata[i]["label"],
            "prompt": reduced_prompts[i],
            "response": all_responses[i]
        }
        reasoning_data.append(entry)

    output_path = f"snli_reasoning_{len(reasoning_data)}_{output_suffix}.json"
    with open(output_path, "w") as f:
        json.dump(reasoning_data, f, indent=2)
    print(f"\nReasoning traces saved to {output_path}")
    print(f"Total samples: {len(reasoning_data)}")


if __name__ == "__main__":
    main()

