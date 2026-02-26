import json
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import autoscale


pipeline_config = {
    "qwen": {
        "model": "Qwen/Qwen3-1.7B",
        "token_limit": 1000,
        "batching_size": 64,
        "autoscale_batch": True
    },
    "liquid": {
        "model": "LiquidAI/LFM2.5-1.2B-Base",
        "token_limit": 1000,
        "batching_size": 128,
        "autoscale_batch": True
    }
}


def worker(rank, prompts, labels, metadata_list, pipeline_name):
    config = pipeline_config[pipeline_name]
    model_id = config["model"]

    device = torch.device(f"cuda:{rank}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        device=device
    )
    print(f"Worker {rank} using device: {device}")

    batch_size = config["batching_size"]
    if config.get("autoscale_batch", False):
        print(f"Autoscaling batch size for worker {rank}, initial size: {batch_size}")
        batch_size = autoscale.get_batch_size(
            pipe, prompts, config["token_limit"], rank,
            memory_buffer_ratio=0.7,
            test_rounds=6
        )
        print(f"Worker {rank} new batch size: {batch_size}")

    print(f"Worker {rank} starting inference on {len(prompts)} samples.")

    responses_raw = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Worker {rank}", disable=(rank != 0)):
            batch = prompts[i:i+batch_size]
            out = pipe(
                batch,
                max_new_tokens=config["token_limit"],
                batch_size=batch_size,
            )
            responses_raw.extend(out)

    print(f"Worker {rank} inference finished!")

    responses = [resp[0]["generated_text"] for resp in responses_raw]

    result_data = {
        "responses": responses,
        "labels": labels,
        "metadata": metadata_list
    }
    with open(f"shard-{rank}.json", "w") as f:
        json.dump(result_data, f)
    print(f"Worker {rank} results written to shard-{rank}.json")

