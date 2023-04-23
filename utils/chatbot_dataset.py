import torch
from torch.utils.data import Dataset

class ChatbotDataset(Dataset):
    def __init__(self, conversations):
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        input_text = conversation["input_text"]
        target_text = conversation["target_text"]
        input_tensor = torch.Tensor(input_text)
        target_tensor = torch.Tensor(target_text)
        return input_tensor, target_tensor