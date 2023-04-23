import os
import pyttsx3
import speech_recognition as sr
#import wikipediaapi
import openai
import requests
from bs4 import BeautifulSoup

# Set up OpenAI API key
openai.api_key = "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"

# Set up search engine API key and search query
SEARCH_ENGINE_ID = "davinci"
API_KEY = "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"

# Set up Wikipedia API
#wiki = wikipediaapi.Wikipedia('en')

# Set up text-to-speech engine
engine = pyttsx3.init()

#Function to get current time in a specified timezone#
def get_current_time(timezone_name):
    tz = timezone(timezone_name)
    current_time = datetime.now(tz)
    return current_time.strftime("%I:%M %p")


# Function to generate speech from text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to generate text from speech
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        user_input = r.recognize_google(audio)
        print("You said: ", user_input)
        return user_input
    except:
        print("Sorry, I could not understand what you said. Please try again.")
        return ""

# Function to generate code
def generate_code(prompt):
    model_engine = "curie"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

# Function to search for solutions
def search_solutions(query):
    url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    result_links = [r.get('href') for r in soup.find_all('a')]
    return result_links

# Main function to interact with the bot
def main():
    speak("Welcome to the code writing bot! How can I help you today?")
    print("Welcome to the code writing bot! Speak or type 'exit' to end the conversation.")
    while True:
        user_input = listen()
        if user_input.lower() == "exit":
            speak("Goodbye!")
            break
        
        # Generate code
        code = generate_code(user_input)
        speak("Here's the code I generated:")
        print("Here's the code I generated:")
        print(code)
        # speak(code)
        
        # Search for solutions
        solutions = search_solutions(user_input)
        if len(solutions) > 0:
            speak("Here are some relevant solutions I found online:")
            print("Here are some relevant solutions I found online:")
            for i, solution in enumerate(solutions[:3]):
                print(f"{i+1}. {solution}")
                speak(f"{i+1}. {solution}")
        else:
            speak("I'm sorry, I couldn't find any relevant solutions online.")
            print("I'm sorry, I couldn't find any relevant solutions online.")

if __name__ == "__main__":
    main()
