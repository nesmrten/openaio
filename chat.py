import torch
import json
import argparse
from train import Encoder, DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained models
encoder = Encoder(input_size=768, hidden_size=256).to(device)
decoder = DecoderRNN(hidden_size=256, output_size=1).to(device)
encoder.load_state_dict(torch.load("encoder.pth"))
decoder.load_state_dict(torch.load("decoder.pth"))

# Load word to index and index to word mappings
with open("word2idx.json", "r") as f:
    word2idx = json.load(f)
idx2word = {i: w for w, i in word2idx.items()}

# Function to generate response
def generate_response(encoder, decoder, sentence, device, max_length=50):
    with torch.no_grad():
        # Encode input sentence
        input_tensor = torch.tensor(sentence, dtype=torch.long, device=device).unsqueeze(1).float()
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.init_hidden().to(device)
        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        # Initialize decoder input with SOS_token
        decoder_input = torch.zeros((1, 1, 1), dtype=torch.float, device=device)

        # Decode encoded input
        decoded_words = []
        decoder_hidden = encoder_hidden
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            if topi == 0:
                break
            else:
                decoded_words.append(idx2word[str(topi.item())])
            decoder_input = torch.tensor([[topi]], dtype=torch.float, device=device)
        return " ".join(decoded_words)

# Start chatting with the bot
while True:
    # Get user input
    user_input = input("You: ")
    # Generate bot response
    bot_response = generate_response(encoder, decoder, user_input, device)
    print("Bot:", bot_response)
