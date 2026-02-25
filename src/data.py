from datasets import load_dataset
from typing import Literal
import random

NLI_PROMPT_VARIATIONS = [
    lambda p, h: f"Is the hypothesis entailed, neutral, or contradictory to the premise? Premise: {p} Hypothesis: {h}",
    lambda p, h: f"What is the relationship between premise and hypothesis? Premise: {p} Hypothesis: {h}",
    lambda p, h: f"Inference the relationship between the Premise: {p} and Hypothesis: {h}",
    lambda p, h: f"Classify as entailment, neutral, or contradiction.\nPremise: {p}\nHypothesis: {h}",
    lambda p, h: f"Premise: {p}\nHypothesis: {h}\nRelationship:",
]

CLASSIFICATION_MAP = ["entailment", "neutral", "contradiction"]

def build_sft_example(example):
    prompt_fn = random.choice(NLI_PROMPT_VARIATIONS)
    prompt = [ {"role":"user","content":prompt_fn(example["premise"], example["hypothesis"])} ]

    label = CLASSIFICATION_MAP[example["label"]]
    completion = [ {"role":"assistant", "content":label}]

    example["prompt"] = prompt
    example["completion"] = completion
    return example


def build_NLI_prompt(example):
    test_example = f"Determine the relationship between the Premise and Hypothesis.\nPremise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves
it. The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. User: {test_example}. Assistant:"""
    example["prompt"] = prompt
    return example

def build_dpo_example(example):
    prompt_fn = random.choice(NLI_PROMPT_VARIATIONS)
    prompt_content = prompt_fn(example["premise"], example["hypothesis"])
    
    correct_label = CLASSIFICATION_MAP[example["label"]]
    all_labels = ["entailment", "neutral", "contradiction"]
    incorrect_labels = [l for l in all_labels if l != correct_label]
    rejected_label = random.choice(incorrect_labels)
    
    example["prompt"] = [{"role": "user", "content": prompt_content}]
    example["chosen"] = [{"role": "assistant", "content": correct_label}]
    example["rejected"] = [{"role": "assistant", "content": rejected_label}]
    example["label"] = correct_label
    
    return example

class Data:
    def __init__(self,dataset_name="snli",split="train"):
        self.dataset = load_dataset(dataset_name, split=split)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def build_prompt(self,template_fn):
        self.dataset = self.dataset.map(template_fn)

    def build_sft(self):
        self.build_prompt(build_sft_example)
        
    def build_grpo(self):
        self.build_prompt(build_NLI_prompt)

    def build_dpo(self):
        self.dataset = self.dataset.map(
            build_dpo_example,
            remove_columns=self.dataset.column_names
        )

    def subsample(self, num_samples, seed=42):
        if isinstance(num_samples, float) and 0 < num_samples <= 1:
            num_samples = int(len(self.dataset) * num_samples)
        
        self.dataset = self.dataset.shuffle(seed=seed).select(range(num_samples))

    def subset(self, num_samples):
        self.dataset = self.dataset.select(range(num_samples))