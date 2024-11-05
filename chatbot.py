import whisper
import sounddevice as sd
import numpy as np
import requests
import random
import logging
import time
import json
import pyttsx3  # Using pyttsx3 as an alternative for text-to-speech

# Set up logging to a file for debugging
logging.basicConfig(filename='chatbot.log', level=logging.INFO)

# Initialize the components
model = whisper.load_model("base")  # Load the Whisper model

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 180)  # Increased speed of speech
engine.setProperty('volume', 0.9)  # Set volume of speech

# Load the Horizon club info JSON data
with open('horizon_club_info.json', 'r') as file:
    horizon_data = json.load(file)

# Convert Horizon club info into a text block to add to prompts
horizon_info_text = f"""
Club Name: {horizon_data['club_name']}
Tagline: {horizon_data['tagline']}
About Us: {horizon_data['about_us']}
Vision: {horizon_data['vision']}
Focus Areas: {', '.join(horizon_data['focus_areas'].keys())}
Event Spotlight: {horizon_data['event_spotlight']['name']} - {horizon_data['event_spotlight']['description']}
"""

# Club FAQ Dictionary
faq_dict = {
    "What is your name?": [
        "I'm your friendly chatbot! created by Mustafa Idris,the advisor of Horizon",
        "You can call me Chatbot.",
        "I'm known as the Horizon Event Chatbot."
    ],
    "How can I help you?": [
        "I'm here to assist you with your queries.",
        "Feel free to ask me anything!",
        "How may I assist you today?"
    ],
    # Add more FAQs and responses as needed
}

# Function to listen and recognize speech using Whisper
def listen():
    print("Listening...")
    try:
        duration = 3  # Reduced seconds to listen for quicker response
        sample_rate = 16000  # Sample rate for recording
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        audio = np.squeeze(audio)
        result = model.transcribe(audio, fp16=False)  # Disable fp16 for better compatibility
        text = result['text'].strip()
        print(f"You said: {text}")
        return text
    except Exception as e:
        print(f"Error during listening: {e}")
        return None

# Function to get response from the Ollama model
def get_ollama_response(query):
    api_endpoint = r"http://localhost:11434/api/generate"  # Replace with the actual Ollama2 API endpoint

    headers = {
        'Content-Type': 'application/json'
    }

    # Append Horizon club information to the prompt for context
    complete_prompt = f"{horizon_info_text}\nUser Question: {query}\n"

    data = {
        "model": "llama2",  # Replace with the correct model name
        "prompt": complete_prompt,
        "max_tokens": 100,  # Reduced max_tokens for faster response
        "temperature": 0.7
    }

    # Retry mechanism to handle network issues
    retries = 2  # Reduced retries to make process quicker
    for attempt in range(retries):
        try:
            # Debugging: Print the input to Ollama2 and log it
            print(f"Sending to Ollama2: {query}")
            logging.info(f"Sending to Ollama2: {query}")

            response = requests.post(api_endpoint, headers=headers, json=data, timeout=5)  # Reduced timeout for quicker responses

            # Debugging: Print the response status and log it
            print(f"Response Status Code: {response.status_code}")
            logging.info(f"Response Status Code: {response.status_code}")

            if response.status_code == 200:
                # Debugging: Print the raw response text from Ollama2
                print(f"Raw Response Text: {response.text}")
                logging.info(f"Raw Response Text: {response.text}")

                try:
                    response_lines = response.text.strip().splitlines()
                    complete_response = "".join([json.loads(line).get("response", "") for line in response_lines if line.strip()])
                    return complete_response
                except json.JSONDecodeError as e:
                    error_message = f"JSON decode error: {e}. Response text: {response.text}"
                    print(error_message)
                    logging.error(error_message)
                    return "Error: Could not understand the response from Ollama2."
            else:
                # Print and log error details if the request fails
                error_message = f"Error: Could not connect to Ollama2. Status Code: {response.status_code}, Response: {response.text}"
                print(error_message)
                logging.error(error_message)
                return "Error: Could not connect to Ollama2."
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1)  # Wait for 1 second before retrying

    return "Error: Unable to connect to Ollama2 after multiple attempts."

# Function to get response from the chatbot model
def get_response(user_input):
    # Check if the user input is related to club or event information
    if any(keyword in user_input.lower() for keyword in ["club", "event", "join", "horizon"]):
        # Use Ollama2 model for club-related questions
        reply = get_ollama_response(user_input)
    elif user_input.lower() in [q.lower() for q in faq_dict]:
        # Use predefined responses for common questions (case-insensitive match)
        key = next(q for q in faq_dict if q.lower() == user_input.lower())
        reply = random.choice(faq_dict[key])
    else:
        # Use Ollama2 model for general conversation
        reply = get_ollama_response(user_input)
    
    print(f"Chatbot: {reply}")
    logging.info(f"Chatbot: {reply}")
    return reply

# Function to speak the response using pyttsx3
def speak(text):
    if text and text.strip():  # Ensure there's a valid response to speak
        try:
            engine.say(text)  # Use pyttsx3 to generate speech
            engine.runAndWait()
        except Exception as e:
            print(f"Error during speaking: {e}")
            logging.error(f"Error during speaking: {e}")

# Main loop to keep the chatbot running
def main():
    print("Horizon Event Chatbot is ready to chat!")
    logging.info("Horizon Event Chatbot is ready to chat!")
    last_response = ""
    while True:
        try:
            user_text = listen()  # Capture the user's voice input
            if user_text:
                response = get_response(user_text)  # Get a response from the chatbot model
                # Avoid repeating the same response
                if response != last_response:
                    speak(response)  # Speak out the response to the user
                    last_response = response
                else:
                    print("(Avoided repeating the same response)")
                    logging.info("(Avoided repeating the same response)")
        except KeyboardInterrupt:
            print("Chatbot shutting down.")
            logging.info("Chatbot shutting down.")
            break
        except Exception as e:
            error_message = f"An error occurred: {e}"
            print(error_message)
            logging.error(error_message)
            speak("An error occurred. Please try again later.")

# Run the chatbot
if __name__ == "__main__":
    main()

