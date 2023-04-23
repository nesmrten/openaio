import torch.utils.data as data
import json


class ChatDataset(Dataset):
    def __init__(self, conversations):
        self.conversations = conversations
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, index):
        input_text = self.conversations[index]["input"]
        target_text = self.conversations[index]["output"]

        input_ids = torch.tensor(self.tokenizer.encode(input_text, add_special_tokens=True))
        target_ids = torch.tensor(self.tokenizer.encode(target_text, add_special_tokens=True))

        return input_ids, target_ids
