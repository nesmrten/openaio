import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import numpy as np

import json

import logging


logging.basicConfig(level=logging.DEBUG)



class ChatDataset(Dataset):

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



class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(Encoder, self).__init__()

        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)


    def forward(self, input_tensor, hidden_tensor):
        output_tensor, hidden_tensor = self.gru(input_tensor, hidden_tensor)
        return output_tensor, hidden_tensor


    def init_hidden(self):

        return torch.zeros(1, 1, self.hidden_size)



class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size

        self.gru = nn.GRU(output_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input_tensor, hidden_tensor):
        output_tensor, hidden_tensor = self.gru(input_tensor, hidden_tensor)

        output_tensor = self.softmax(self.out(output_tensor[0]))
        return output_tensor, hidden_tensor



def train(encoder, decoder, dataloader, criterion, optimizer, device, n_epochs=10):

    for epoch in range(n_epochs):

        running_loss = 0.0

        for i, (inputs, targets) in enumerate(dataloader):

            inputs = inputs.to(device)

            targets = targets.to(device)

            batch_size = inputs.size(0)

            encoder_hidden = encoder.init_hidden().to(device)

            optimizer.zero_grad()

            loss = 0.0

            for j in range(batch_size):
                encoder_output, encoder_hidden = encoder(

                    inputs[j], encoder_hidden)
                decoder_hidden = encoder_hidden

                decoder_input = torch.Tensor([[0.0]]).to(device)

                for k in range(targets.size(1)):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)

                    loss += criterion(decoder_output, targets[j][k])

                    decoder_input = targets[j][k]

            loss /= targets.size(1)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            step = epoch * len(dataloader) + i

            logging.debug(f"Loss at step {step}: {loss}")

        logging.info(f"Epoch {epoch} loss: {running_loss/len(dataloader)}")



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load conversations from JSON file

    with open("conversations.json", "r") as f:

        conversations = json.load(f)


    # Create dataset and dataloader

    dataset = ChatDataset(conversations)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


    # Initialize encoder and decoder models

    encoder = Encoder(1, 256)

    decoder = Decoder(256, 1)


    # Define loss and optimizer functions

    criterion = criterion = nn.MSELoss()

    nn.NLLLoss()



# Train the model
train(encoder, decoder, dataloader, criterion, optim.Adam(list(encoder.parameters()) + list(decoder.parameters())), device)

