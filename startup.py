import os
import argparse
from chatbot_ai import ChatbotAI

def main():
    # Create instance of ChatbotAI class
    chatbot = ChatbotAI()

    # Start chatbot
    chatbot.start()

if __name__ == '__main__':
    main()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='models/chatbot.pth', help='path to trained model')
parser.add_argument('--conversations_path', type=str, default='conversations.json', help='path to conversations JSON file')
parser.add_argument('--use_gpu', action='store_true', help='use GPU for inference')
args = parser.parse_args()

# Initialize ChatbotAI object
chatbot = ChatbotAI(args.model_path, args.conversations_path, args.use_gpu)

# Run chatbot
while True:
    user_input = input("You: ")
    response = chatbot.generate_response(user_input)
    print(f"Chatbot: {response}")
