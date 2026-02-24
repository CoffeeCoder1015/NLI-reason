from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, PeftModel

class Model:
    def __init__(self,model_id,**kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"

    def attach_lora(self,lora_config: LoraConfig):
        return get_peft_model(self.model,lora_config)
    
    def load_with_lora(self,lora_path):
        return PeftModel.from_pretrained(self.model,lora_path)
