from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import os
import json
import random
import torch
import openai
import pyttsx3
import speech_recognition as sr
import wikipediaapi
import webbrowser


from dotenv import load_dotenv
from datetime import datetime
from pytz import timezone

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatbotAI:
    def __init__(self, model_path='models/chatbot.pth', conversations_path='conversations.json', use_gpu=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.conversations_path = conversations_path
        self.conversations = self.load_conversations()

    def load_conversations(self):
        with open(self.conversations_path, 'r') as f:
            conversations = json.load(f)
        return conversations

    def generate_response(self, user_input):
        prompt = f'{user_input.strip()}'
        input_ids = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors='pt').to(self.device)
        chatbot_output = self.model.generate(input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(chatbot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

    def start(self):
        print('Welcome to the ChatbotAI. Start talking to the bot by typing a message or enter "quit" to exit.')
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
            response = self.generate_response(user_input)
            print(f"Chatbot: {response}")

# Set up Wikipedia API
wiki = wikipediaapi.Wikipedia('en')

# Initialize Text-to-Speech (TTS) engine
engine = pyttsx3.init()

# Function to get current time in a specified timezone


def get_current_time(timezone_name):
    tz = timezone(timezone_name)
    current_time = datetime.now(tz)
    return current_time.strftime("%I:%M %p")

# Function to generate response


def generate_response(prompt):
    # Send prompt to OpenAI API for completion
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# Function to convert text to speech


def text_to_speech(text):
    # Set speech rate and volume
    rate = engine.getProperty('rate')
    volume = engine.getProperty('volume')
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)

    # Convert text to speech
    engine.say(text)
    engine.runAndWait()

# Function to convert speech to text


def speech_to_text():
    # Initialize recognizer
    r = sr.Recognizer()

    # Take input from microphone
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    # Convert speech to text
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language="en-US")
        print(f"You: {query}\n")
    except Exception as e:
        print("Sorry, I did not understand.")
        query = ""
    return query

# Function to handle user queries


def handle_query(query):
    response = ""

    # Check for keywords in user query
    if "time" in query:
        current_time = get_current_time("US/Pacific")
        response = f"The current time is {current_time}."
    elif "search" in query:
        query = query.replace("search", "")
        search_url = f"https://www.google.com/search?q={query}"
        webbrowser.get().open(search_url)
        response = f"Here are some search results for '{query}'."
    elif "who is" in query:
        query = query.replace("who is", "")
        page = wiki.page(query)
        if page.exists():
            summary = page.summary[0:200]
            response = f"{summary}..."
        else:
            response = f"I'm sorry, I could not find any information on '{query}'."
    elif "what is" in query:
        query = query.replace("what is", "")
        page = wiki.page(query)
        if page.exists():
            summary = page.summary[0:200]
            response = f"{summary}..."
        else:
            response = f"I'm sorry, I could not find any information on '{query}'."
    elif "define" in query:
        query = query.replace("define", "")
        page = wiki.page(query)
        if page.exists():
            summary = page.summary[0:200]
            response = f"{summary}..."
        else:
            response = f"I'm sorry, I could not find a definition for '{query}'."
    else:
        # Generate response using OpenAI API
        prompt = f"User: {query}\nAI:"
        response = generate_response(prompt)

    return response
