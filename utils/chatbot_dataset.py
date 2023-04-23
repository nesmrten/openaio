import torch.utils.data as data
import json


class ChatDataset(data.Dataset):
    def __init__(self, conversations):
        self.conversations = conversations
        self.input_size = None
        self.output_size = None
        self.build_vocab()

    def __getitem__(self, index):
        conversation = self.conversations[index]
        input_text = conversation["input_text"]
        target_text = conversation["target_text"]
        input_tensor = torch.Long
