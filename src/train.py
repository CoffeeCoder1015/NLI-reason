from trl import GRPOConfig, SFTConfig
class Train:
    def __init__(self):
        self.GRPO_configs = GRPOConfig(
            output_dir="GRPO",
            learning_rate=1e-5,
            beta=0.008,
            per_device_train_batch_size=8,  # We want to get all generations in one device batch
            gradient_accumulation_steps=2,
            max_completion_length=256,
            num_generations=8,  # Number of completions to generate for each prompt
            num_train_epochs=3,
            logging_steps=10,
            report_to=["wandb"],
            use_vllm=True,  # Speed up generation
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.3,  # Reserve 30% GPU for inference, rest for training
            vllm_max_model_length=512,  # Ensure this covers prompt + completion
            # 🔽 ADD THESE
            save_strategy="steps",  # save by step count
            save_steps=500,  # save every 500 optimizer steps
            save_total_limit=3,  # keep only last 3 checkpoints
            save_safetensors=True,  # recommended
        )

        self.SFT_configs = SFTConfig(
            output_dir="SFT",
            learning_rate=0.5e-4,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            logging_steps=10,
            report_to=["wandb"],
            packing=True,
            use_liger_kernel=True,
            # 🔽 ADD THESE
            save_strategy="steps",  # save by step count
            save_steps=500,  # save every 500 optimizer steps
            save_total_limit=3,  # keep only last 3 checkpoints
            save_safetensors=True,  # recommended
        )