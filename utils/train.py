import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import logging
import argparse

logging.basicConfig(level=logging.DEBUG)


class ChatDataset(Dataset):
    def __init__(self, conversations, max_len):
        self.conversations = conversations
        self.max_len = max_len

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(
            self.conversations[idx]["input_text"]).unsqueeze(1).float()
        target_tensor = torch.tensor(
            self.conversations[idx]["target_text"]).unsqueeze(1).float()

        # pad sequences with 0s if length is less than max_len
        if input_tensor.shape[0] < self.max_len:
            padding = torch.zeros(
                self.max_len - input_tensor.shape[0], input_tensor.shape[1])
            input_tensor = torch.cat((input_tensor, padding))
        if target_tensor.shape[0] < self.max_len:
            padding = torch.zeros(
                self.max_len - target_tensor.shape[0], target_tensor.shape[1])
            target_tensor = torch.cat((target_tensor, padding))

        return input_tensor, target_tensor


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)

    def forward(self, input_tensor, hidden_tensor):
        # compute lengths of non-zero elements in input_tensor
        input_lengths = torch.count_nonzero(input_tensor, dim=-1)
        sorted_lengths, sorted_idx = input_lengths.sort(0, descending=True)
        input_tensor = input_tensor[sorted_idx]
        _, reverse_idx = sorted_idx.sort(0)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            input_tensor, sorted_lengths, batch_first=True, enforce_sorted=False)
        output_tensor, hidden_tensor = self.gru(packed_input, hidden_tensor)
        output_tensor, _ = nn.utils.rnn.pad_packed_sequence(
            output_tensor, batch_first=True)
        output_tensor = output_tensor[reverse_idx]
        hidden_tensor = hidden_tensor[:, reverse_idx]
        return output_tensor, hidden_tensor


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_tensor, hidden_tensor):
        output_tensor, hidden_tensor = self.gru(input_tensor, hidden_tensor)
        output_tensor = self.out(output_tensor)
        return output_tensor,


logging.basicConfig(level=logging.DEBUG)


class ChatDataset(Dataset):
    def __init__(self, conversations, max_len):
        self.conversations = conversations
        self.max_len = max_len

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(
            self.conversations[idx]["input_text"]).unsqueeze(1).float()
        target_tensor = torch.tensor(
            self.conversations[idx]["target_text"]).unsqueeze(1).float()

        # pad sequences with 0s if length is less than max_len
        if input_tensor.shape[0] < self.max_len:
            padding = torch.zeros(
                self.max_len - input_tensor.shape[0], input_tensor.shape[1])
            input_tensor = torch.cat((input_tensor, padding))
        if target_tensor.shape[0] < self.max_len:
            padding = torch.zeros(
                self.max_len - target_tensor.shape[0], target_tensor.shape[1])
            target_tensor = torch.cat((target_tensor, padding))

        return input_tensor, target_tensor


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)

    def forward(self, input_tensor, hidden_tensor, input_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            input_tensor, input_lengths, batch_first=True, enforce_sorted=False)
        output_tensor, hidden_tensor = self.gru(packed_input, hidden_tensor)
        output_tensor, _ = nn.utils.rnn.pad_packed_sequence(
            output_tensor, batch_first=True)
        return output_tensor, hidden_tensor


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_tensor, hidden_tensor):
        input_tensor = torch.tensor(input_seq, dtype=torch.long).view(-1, 1, 1)
        output_tensor, hidden_tensor = self.gru(input_tensor, hidden_tensor)
        output_tensor = self.out(output_tensor)
        return output_tensor, hidden_tensor


class SeqDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(
            self.input_data[idx], dtype=torch.long).unsqueeze(1)
        target_tensor = torch.tensor(
            self.target_data[idx], dtype=torch.long).unsqueeze(1)
        return input_tensor, target_tensor


class Chatbot:
    def __init__(self, encoder, decoder, device):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        self.encoder.eval()
        self.decoder.eval()

# Regenerate response


def train(encoder, decoder, dataloader, criterion, optimizer, device, n_epochs=1):
    encoder.train()
    decoder.train()
    for epoch in range(n_epochs):
        for i, data in enumerate(dataloader, 0):
            input_tensor, target_tensor = data
            input_tensor, target_tensor = input_tensor.to(
                device), target_tensor.to(device)

            # Reshape input_tensor
            input_tensor = input_tensor.view(1, 1, -1).float()
            batch_size = input_tensor.size(0)
            encoder_hidden = encoder.init_hidden().to(device)
            optimizer.zero_grad()
            loss = 0.0
            for j in range(batch_size):
                encoder_input = input_tensor[j].unsqueeze(
                    0)  # Add the seq_len dimension
                encoder_output, encoder_hidden = encoder(
                    encoder_input, encoder_hidden)
                decoder_hidden = encoder_hidden
                decoder_input = torch.Tensor([[0.0]]).to(device)
                for k in range(target_tensor.size(1)):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    loss += criterion(decoder_output, target_tensor[j][k])
                    decoder_input = target_tensor[j][k]
            loss /= target_tensor.size(1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step = epoch * len(dataloader) + i
            logging.debug(f"Loss at step {step}: {loss}")
        logging.info(f"Epoch {epoch} loss: {running_loss/len(dataloader)}")

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            batch_size = input_tensor.size(0)
            encoder_hidden = encoder.init_hidden().to(device)
            optimizer.zero_grad()
            loss = 0.0
            for j in range(batch_size):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[j], encoder_hidden)
                decoder_hidden = encoder_hidden
                decoder_input = torch.Tensor([[0.0]]).to(device)
                for k in range(target_tensor.size(1)):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    loss += criterion(decoder_output, target_tensor[j][k])
                    decoder_input = target_tensor[j][k]
            loss /= target_tensor.size(1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step = epoch * len(dataloader) + i
            logging.debug(f"Loss at step {step}: {loss}")
        logging.info(f"Epoch {epoch} loss: {running_loss/len(dataloader)}")

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            batch_size = input_tensor.size(0)
            encoder_hidden = encoder.init_hidden().to(device)
            optimizer.zero_grad()
            loss = 0.0
            for j in range(batch_size):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[j].unsqueeze(0), encoder_hidden)  # Add unsqueeze here
                decoder_hidden = encoder_hidden
                decoder_input = torch.Tensor([[0.0]]).to(device)
                for k in range(target_tensor.size(1)):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    loss += criterion(decoder_output, target_tensor[j][k])
                    decoder_input = target_tensor[j][k]
            loss /= target_tensor.size(1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step = epoch * len(dataloader) + i
            logging.debug(f"Loss at step {step}: {loss}")
        logging.info(f"Epoch {epoch} loss: {running_loss/len(dataloader)}")

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            batch_size = input_tensor.size(0)
            encoder_hidden = encoder.init_hidden().to(device)
            optimizer.zero_grad()
            loss = 0.0
            for j in range(batch_size):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[j], encoder_hidden)
                decoder_hidden = encoder_hidden
                decoder_input = torch.Tensor([[0.0]]).to(device)
                for k in range(target_tensor.size(1)):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    loss += criterion(decoder_output, target_tensor[j][k])
                    decoder_input = target_tensor[j][k]
            loss /= target_tensor.size(1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step = epoch * len(dataloader) + i
            logging.debug(f"Loss at step {step}: {loss}")
        logging.info(f"Epoch {epoch} loss: {running_loss/len(dataloader)}")

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            batch_size = input_tensor.size(0)
            encoder_hidden = encoder.init_hidden().to(device)
            optimizer.zero_grad()
            loss = 0.0
            for j in range(batch_size):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[j], encoder_hidden)
                decoder_hidden = encoder_hidden
                decoder_input = torch.Tensor([[0.0]]).to(device)
                for k in range(target_tensor.size(1)):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    loss += criterion(decoder_output, target_tensor[j][k])
                    decoder_input = target_tensor[j][k]
            loss /= target_tensor.size(1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step = epoch * len(dataloader) + i
            logging.debug(f"Loss at step {step}: {loss}")
        logging.info(f"Epoch {epoch} loss: {running_loss/len(dataloader)}")

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            input_tensor = input_tensor.unsqueeze(1)  # Add the batch dimension
            target_tensor = target_tensor.unsqueeze(
                1)  # Add the batch dimension
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            batch_size = input_tensor.size(0)
            encoder_hidden = encoder.init_hidden().to(device)
            optimizer.zero_grad()
            loss = 0.0

            # Updated: pass the entire input_tensor
            encoder_output, encoder_hidden = encoder(
                input_tensor, encoder_hidden)

            for j in range(batch_size):
                decoder_hidden = encoder_hidden
                decoder_input = torch.Tensor([[0.0]]).to(device)
                for k in range(target_tensor.size(1)):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    loss += criterion(decoder_output, target_tensor[j][k])
                    decoder_input = target_tensor[j][k]
            loss /= target_tensor.size(1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step = epoch * len(dataloader) + i
            logging.debug(f"Loss at step {step}: {loss}")
        logging.info(f"Epoch {epoch} loss: {running_loss/len(dataloader)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=768,
                        help='size of input tensor')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='size of hidden layer tensor')
    parser.add_argument('--output_size', type=int, default=1,
                        help='size of output tensor')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for optimizer')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='number of training epochs')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load conversations from JSON file
    with open("utils/chat_data.json", "r") as f:
        conversations = json.load(f)

    # Create dataset and dataloader
    dataset = ChatDataset(conversations)
    train_input_data, train_target_data = load_data()
    train_dataset = SeqDataset(train_input_data, train_target_data)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Initialize encoder and decoder models
    encoder = Encoder(args.input_size, args.hidden_size)
    decoder = Decoder(args.hidden_size, args.output_size)

    # Define loss and optimizer functions
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) +
                           list(decoder.parameters()), lr=args.learning_rate)

    # Train the model
    train(encoder, decoder, dataloader, criterion,
          optimizer, device, n_epochs=args.n_epochs)
