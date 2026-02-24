from trl import GRPOConfig, SFTConfig
class Train:
    def __init__(self):
        self.GRPO_configs = GRPOConfig(
            output_dir="GRPO",
            learning_rate=1e-5,
            beta=0.01,
            per_device_train_batch_size=8,  # We want to get all generations in one device batch
            gradient_accumulation_steps=2,
            max_completion_length = 1024,
            num_generations=8,  # Number of completions to generate for each prompt
            num_train_epochs=3,
            logging_steps=10,
            report_to=["wandb"],
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.35,
        )
        
        self.SFT_configs = SFTConfig(
            output_dir="SFT",
            learning_rate=1e-4,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            max_seq_length=512,
            num_train_epochs=3,
            logging_steps=10,
            report_to=["wandb"],
            packing=True,
            gradient_checkpointing=True,
            save_strategy="epoch",
            save_total_limit=1,
            use_liger_kernel=True
        )