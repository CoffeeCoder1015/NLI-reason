from datasets import load_dataset

class Data:
    def __init__(self,dataset_name="snli",split="train"):
        self.dataset = load_dataset(dataset_name, split=split)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def build_prompt(self,template_fn,mode: Literal["multi-turn","text-completion"] = "text-completion",tokenizer=None):
        if mode == "multi-turn":
            # User provides full chat template, requires tokenizer to format
            self.dataset = self.dataset.map(lambda x:tokenizer.apply_chat_template(template_fn(x),tokenize=False))
        else:
            # User provides prompt 
            self.dataset = self.dataset.map(template_fn)
        