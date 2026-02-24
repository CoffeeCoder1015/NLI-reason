from data import Data
from model import Model
import re
from peft import LoraConfig, TaskType
from trl import GRPOTrainer, SFTTrainer
import torch

from train import Train
import wandb

wandb.login()

def reward_func(completions, label, premise, hypothesis, **kwargs):
    classification_map = ["entailment", "neutral", "contradiction"]
    word_labels = [classification_map[i] for i in label]
    
    rewards = []
    for completion, correct_answer, prem, hypo in zip(completions,word_labels,premise,hypothesis):
        reward = 0.0
        # Format compliance
        format_reward = 0.0
        # 1. BRAINSTORMING REWARD (Did it try to think?)
        if "<think>" in completion: format_reward += 0.05
        if "</think>" in completion: format_reward += 0.05
        
        # 2. STRUCTURE format_reward (Did it provide an answer block?)
        if "<answer>" in completion: format_reward += 0.05
        if "</answer>" in completion: format_reward += 0.05

        format_pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        format_match = re.search(format_pattern, completion, re.DOTALL)
        if format_match:
            format_reward += 0.2  # partial reward for compliant format
        
        reward += min(0.2,format_reward)
        
        # Extract <think> and <answer>
        think_text = ""
        answer_text = ""
        if format_match:
            think_text = re.findall(r"<think>(.*?)</think>", completion, re.DOTALL)
            answer_text = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            think_text = think_text[0].strip().lower() if think_text else ""
            answer_text = answer_text[0].strip().lower() if answer_text else ""
        
        # Correct answer
        if correct_answer == answer_text:
            reward += 0.8
            
        # Heuristic magic
        if prem and hypo and think_text:
            premise_words = set(prem.lower().split(" "))
            hypothesis_words = set(hypo.lower().split(" "))
            think_words = set(think_text.split(" "))
            
            check_set = (premise_words | hypothesis_words)
            overlap = check_set & think_words
            copy_rate = len(overlap)/len(check_set) if len(check_set) > 0 else 0

            reward += min(0.1,copy_rate/2)
        
        reward = min(reward,1.0)
        rewards.append(reward)

    return rewards

def GRPO_pipeline():
    m = Model("LiquidAI/LFM2.5-1.2B-Base", attn_implementation="flash_attention_2",dtype=torch.bfloat16)
    
    dataset = Data()
    dataset.build_grpo()

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "fc_in", "fc_out"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = m.attach_lora(config)

    trainer_configs = Train().GRPO_configs
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=trainer_configs,
        train_dataset=dataset.dataset,
    )
    
    wandb.init(project="GRPO",entity="messing_around")
    trainer.train()

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained("./liquid_snli")

def SFT_pipeline():
    m = Model("LiquidAI/LFM2.5-1.2B-Base", attn_implementation="flash_attention_2", dtype=torch.bfloat16)
    
    dataset = Data()
    dataset.subsample(25_000)
    dataset.build_sft()

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "fc_in", "fc_out"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = m.attach_lora(config)
    
    trainer_configs = Train().SFT_configs
    trainer = SFTTrainer(
        model=model,
        args=trainer_configs,
        train_dataset=dataset.dataset,
    )

    wandb.init(project="SFT-NLI",entity="messing_around")
    trainer.train()

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained("./liquid_sft_snli")

# CLI entrypoint
import argparse
import inspect
import sys

def _get_pipelines():
    return {
        name.replace('_pipeline', '').lower(): func
        for name, func in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
        if name.endswith('_pipeline')
    }

def _main():
    pipelines = _get_pipelines()
    parser = argparse.ArgumentParser(description="Run training pipelines")
    parser.add_argument(
        "--pipeline",
        "-p",
        required=True,
        choices=sorted(pipelines.keys()),
        help="Name of the pipeline to run"
    )
    args = parser.parse_args()
    pipeline_func = pipelines[args.pipeline]
    print(f"Running {args.pipeline} pipeline...")
    pipeline_func()

if __name__ == "__main__":
    _main()