import random


NLI_PROMPT_VARIATIONS = [
    lambda p, h: f"Is the hypothesis entailed, neutral, or contradictory to the premise? Premise: {p} Hypothesis: {h}",
    lambda p, h: f"Does the premise support the hypothesis? Premise: {p} Hypothesis: {h}",
    lambda p, h: f"What is the relationship between premise and hypothesis? Premise: {p} Hypothesis: {h}",
    lambda p, h: f"Classify as entailment, neutral, or contradiction.\nPremise: {p}\nHypothesis: {h}",
    lambda p, h: f"Premise: {p}\nHypothesis: {h}\nRelationship: ___",
]

CLASSIFICATION_MAP = ["entailment", "neutral", "contradiction"]

def build_sft_example(example):
    prompt_fn = random.choice(NLI_PROMPT_VARIATIONS)
    prompt = prompt_fn(example["premise"], example["hypothesis"])
    label = CLASSIFICATION_MAP[example["label"]]
    example["text"] = f"User: {prompt}\nAssistant: {label}"
    return example



def build_NLI_prompt(example):
    test_example = f"Determine the relationship between the Premise and Hypothesis.\nPremise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves
it. The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think>...</think>
and <answer>...</answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: {test_example}. Assistant:"""
    example["prompt"] = prompt
    return example