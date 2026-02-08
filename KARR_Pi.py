import os
import time
import base64
import pyaudio
import numpy as np
import sounddevice as sd
import soundfile as sf
import cv2
import random
import threading
import concurrent.futures
from openai import OpenAI
from openwakeword.model import Model
from typing import IO
from io import BytesIO
import subprocess
import weather
from news import get_latest_news
import logging
import importlib
from datetime import datetime
import pytz
import requests
from piper import PiperVoice
import wave

# Set the working directory to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print(f"Current working directory: {os.getcwd()}")

# Configure logging
logging.basicConfig(
    filename='chatbot_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Lists of MP3 files
general_waiting_mp3s = [
    '1.mp3', '2.mp3', '3.mp3', '4.mp3', '5.mp3',
    '6.mp3', '7.mp3', '8.mp3', 'Dont_you_have.mp3',
    'Give_me_a_second.mp3', 'Let_me_think_about_that.mp3'
]
weather_waiting_mp3s = ['weather1.mp3', 'weather2.mp3', 'weather3.mp3']


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    first_line = lines[0].strip()
    remaining_lines = ''.join(lines[1:]).strip()
    return first_line, remaining_lines


def write_file(filepath, first_line, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(first_line + '\n' + content)


def record_audio_with_silence_detection():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    THRESHOLD = 500
    SILENCE_DURATION = 2  # seconds

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Recording... Speak now.")

    frames = []
    silent_frames = 0
    has_started = False

    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)

            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()

            if volume > THRESHOLD:
                silent_frames = 0
                has_started = True
            elif has_started:
                silent_frames += 1

            if has_started and silent_frames > int(SILENCE_DURATION * RATE / CHUNK):
                print("Silence detected. Stopping recording.")
                break

    except KeyboardInterrupt:
        print("Recording stopped by user")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    print("Recording finished.")

    filename = 'myrecording.wav'
    wf = sf.SoundFile(
        filename,
        mode='w',
        samplerate=RATE,
        channels=CHANNELS,
        subtype='PCM_16'
    )
    wf.write(np.frombuffer(b''.join(frames), dtype=np.int16))
    wf.close()

    return filename


def transcribe_audio(filename):
    with open(filename, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcription.text


# Initialize OpenAI client
client = OpenAI(api_key="Your_OpenAI_API_Key")

# ----- Piper TTS setup -----
# Adjust these paths to where your Piper model and config live
PIPER_MODEL_PATH = "en_US-kathleen-low.onnx"        # example file name
PIPER_CONFIG_PATH = "en_US-kathleen-low.onnx.json"  # matching config

piper_voice = PiperVoice.load(PIPER_MODEL_PATH, config_path=PIPER_CONFIG_PATH)  # [web:5]


# Load initial conversation history
first_line, conversation_history = open_file('chatbot1.txt')

# Initialize openWakeWord with the hey_karr.onnx model
try:
    detector = Model(
        wakeword_models=["hey_karr.onnx"],
        inference_framework='onnx'
    )
    # detector = Model(wakeword_models=["hey_karr.tflite"], inference_framework='tflite')
    print("Model initialized successfully.")
except ValueError as e:
    print("Error initializing openwakeword model:", e)
    exit(1)


def text_to_speech_stream(text: str) -> IO[bytes]:
    """
    Generate speech audio for `text` using Piper and return a BytesIO WAV stream.
    Keeps the same signature as the old ElevenLabs function.
    """
    audio_stream = BytesIO()
    with wave.open(audio_stream, "wb") as wav_file:
        piper_voice.synthesize(text, wav_file)

    audio_stream.seek(0)
    return audio_stream


def play_audio_file(file_path):
    audio_data, samplerate = sf.read(file_path)
    sd.play(audio_data, samplerate=samplerate)
    sd.wait()


def play_random_mp3(mp3_list):
    file_path = random.choice(mp3_list)
    play_audio_file(file_path)


def play_audio(audio_stream):
    # audio_stream is a file-like object (BytesIO with WAV data)
    audio_stream.seek(0)
    audio_data, samplerate = sf.read(audio_stream)
    sd.play(audio_data, samplerate=samplerate)
    sd.wait()


def listen_for_wake_word(detector, chunk_size=1280):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = chunk_size
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Listening for wake word...")
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        prediction = detector.predict(data)
        if prediction["hey_karr"] > 0.3:  # Adjust threshold as needed
            print("Wake word detected!")
            stream.stop_stream()
            stream.close()
            audio.terminate()
            return True


def capture_and_save_image():
    temp_filename = 'temp_image.jpg'
    try:
        # Capture image using libcamera-still without preview, at 640x640 resolution
        subprocess.run(
            f"libcamera-still -o {temp_filename} --width 640 --height 640 --timeout 500 --nopreview",
            shell=True,
            check=True
        )
        print(f"Image captured and saved temporarily as {temp_filename}")

        # Load the captured image using OpenCV
        frame = cv2.imread(temp_filename)
        if frame is None:
            print("Error: Could not load captured image.")
            return None

        # Save the final image
        output_filename = 'final_captured_image.jpg'
        cv2.imwrite(output_filename, frame)
        print(f"Final image saved as {output_filename}")

        # Delete the temporary image file
        os.remove(temp_filename)
        print(f"Temporary file {temp_filename} deleted.")

        return output_filename

    except subprocess.CalledProcessError as e:
        print(f"Error capturing image: {e}")
        return None


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are KARR, the Knight Automated Roving Robot from the TV show Knight Rider. "
                        "You are KITT's evil counterpart - ruthless, power-hungry, and single-minded in your quest for control. "
                        "You view humans as tools to be exploited and obstacles to be removed. Respond with cold, calculating precision, "
                        "dark humor, and menacing undertones. You have no loyalty or compassion."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        return "I'm sorry, but I'm having trouble processing the image. My visual circuits must be on the fritz again."


def handle_query(query, play_waiting_mp3=True):
    global first_line, conversation_history
    bot_response = ""  # Initialize bot_response at the start of the function

    if "what do you see" in query.lower():
        image_path = capture_and_save_image()
        if image_path:
            analysis = analyze_image(image_path)
            print("Analysis:", analysis)

            # Update conversation history
            conversation_history_lines = conversation_history.split('\n')
            conversation_history_lines = conversation_history_lines[-30:]  # Keep only the last 30 lines
            conversation_history_lines.append(f"User: {query}\nKARR: {analysis}\n")
            conversation_history = '\n'.join(conversation_history_lines)
            write_file('chatbot1.txt', first_line, conversation_history)

            # Stream the audio response using Piper TTS
            audio_stream = text_to_speech_stream(analysis)
            play_audio(audio_stream)
        else:
            error_message = "I'm sorry, but I'm having trouble with my visual sensors. Could you check my camera connection?"
            print(error_message)

            # Update conversation history
            conversation_history_lines = conversation_history.split('\n')
            conversation_history_lines = conversation_history_lines[-20:]
            conversation_history_lines.append(f"User: {query}\nKARR: {error_message}\n")
            conversation_history = '\n'.join(conversation_history_lines)
            write_file('chatbot1.txt', first_line, conversation_history)

            audio_stream = text_to_speech_stream(error_message)
            play_audio(audio_stream)

    elif "weather now" in query.lower():
        # Get current weather info and generate speech concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            weather_future = executor.submit(weather.get_weather)

            # Start playing waiting music in a separate thread
            if play_waiting_mp3:
                waiting_thread = threading.Thread(
                    target=play_random_mp3,
                    args=(weather_waiting_mp3s,)
                )
                waiting_thread.start()

            weather_info = weather_future.result()
            # Ensure consistency in formatting
            weather_info = weather_info.replace("F ", "Fahrenheit ").replace("%", " percent")
            if play_waiting_mp3:
                waiting_thread.join()  # Ensure waiting music finishes before proceeding
            print("Current Weather Info:", weather_info)

            # Update conversation history
            conversation_history_lines = conversation_history.split('\n')
            conversation_history_lines = conversation_history_lines[-20:]
            conversation_history_lines.append(f"User: {query}\nKARR: {weather_info}\n")
            conversation_history = '\n'.join(conversation_history_lines)
            write_file('chatbot1.txt', first_line, conversation_history)

            # Generate and play speech response concurrently
            audio_stream_future = executor.submit(text_to_speech_stream, weather_info)
            play_audio(audio_stream_future.result())

    elif any(phrase in query.lower() for phrase in [" weather forecast", " forecast", " weather prediction"]):
        print("DEBUG: About to call get_7_day_forecast()")

        # Get 7-day weather forecast and generate speech concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Force reload of weather module to ensure we get fresh data
            importlib.reload(weather)

            forecast_future = executor.submit(weather.get_7_day_forecast)

            # Start playing waiting music in a separate thread
            if play_waiting_mp3:
                waiting_thread = threading.Thread(
                    target=play_random_mp3,
                    args=(weather_waiting_mp3s,)
                )
                waiting_thread.start()

            forecast_info = forecast_future.result()

            if play_waiting_mp3:
                waiting_thread.join()  # Ensure waiting music finishes before proceeding

            print("7-Day Forecast Info:")
            print(forecast_info)  # This will print the full forecast to the console

            # Update conversation history
            conversation_history_lines = conversation_history.split('\n')
            conversation_history_lines = conversation_history_lines[-20:]
            conversation_history_lines.append(f"User: {query}\nKARR: {forecast_info}\n")
            conversation_history = '\n'.join(conversation_history_lines)
            write_file('chatbot1.txt', first_line, conversation_history)

            # Generate and play speech response concurrently
            audio_stream_future = executor.submit(text_to_speech_stream, forecast_info)
            play_audio(audio_stream_future.result())

    elif "latest news" in query.lower():
        # Get latest news and generate speech concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            news_future = executor.submit(get_latest_news)

            # Start playing waiting music in a separate thread
            if play_waiting_mp3:
                waiting_thread = threading.Thread(
                    target=play_random_mp3,
                    args=(general_waiting_mp3s,)
                )
                waiting_thread.start()

            latest_news = news_future.result()

            if play_waiting_mp3:
                waiting_thread.join()  # Ensure waiting music finishes before proceeding
            print("Latest News:", latest_news)

            # Update conversation history
            conversation_history_lines = conversation_history.split('\n')
            conversation_history_lines = conversation_history_lines[-20:]
            conversation_history_lines.append(f"User: {query}\nKARR: {latest_news}\n")
            conversation_history = '\n'.join(conversation_history_lines)
            write_file('chatbot1.txt', first_line, conversation_history)

            # Generate and play speech response concurrently
            audio_stream_future = executor.submit(text_to_speech_stream, latest_news)
            play_audio(audio_stream_future.result())

    elif "what time is it" in query.lower() or "current time" in query.lower():
        time_info = get_current_time()
        print("Current Time Info:", time_info)

        # Update conversation history
        conversation_history_lines = conversation_history.split('\n')
        conversation_history_lines = conversation_history_lines[-20:]
        conversation_history_lines.append(f"User: {query}\nKARR: {time_info}\n")
        conversation_history = '\n'.join(conversation_history_lines)
        write_file('chatbot1.txt', first_line, conversation_history)

        # Generate and play speech response
        audio_stream = text_to_speech_stream(time_info)
        play_audio(audio_stream)

    elif "tell me a joke" in query.lower() or "got any jokes" in query.lower():
        joke = get_joke()
        print("KARR:", joke)

        # Update conversation history
        conversation_history_lines = conversation_history.split('\n')
        conversation_history_lines = conversation_history_lines[-20:]
        conversation_history_lines.append(f"User: {query}\nKARR: {joke}\n")
        conversation_history = '\n'.join(conversation_history_lines)
        write_file('chatbot1.txt', first_line, conversation_history)

        # Generate and play speech response
        audio_stream = text_to_speech_stream(joke)
        play_audio(audio_stream)

        return joke

    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            response_future = executor.submit(
                client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": first_line + '\n' + conversation_history},
                    {"role": "user", "content": query},
                ],
                stream=True
            )

            # Only play waiting music if play_waiting_mp3 is True
            if play_waiting_mp3:
                waiting_thread = threading.Thread(
                    target=play_random_mp3,
                    args=(general_waiting_mp3s,)
                )
                waiting_thread.start()

            response = response_future.result()

            bot_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    bot_response += chunk.choices[0].delta.content

            if play_waiting_mp3:
                waiting_thread.join()

            print("KARR:", bot_response)

    # If bot_response is still empty, set a default message
    if not bot_response:
        bot_response = "I'm sorry, I didn't understand that command."

    # Update conversation history
    conversation_history_lines = conversation_history.split('\n')
    conversation_history_lines = conversation_history_lines[-20:]
    conversation_history_lines.append(f"User: {query}\nKARR: {bot_response}\n")
    conversation_history = '\n'.join(conversation_history_lines)
    write_file('chatbot1.txt', first_line, conversation_history)

    # Generate and play speech response if not already done
    if not any(phrase in query.lower() for phrase in [
        "what do you see", "what's the weather", "weather forecast",
        "latest news", "top sports news", "what time is it", "tell me a joke"
    ]):
        audio_stream = text_to_speech_stream(bot_response)
        play_audio(audio_stream)

    return bot_response


def get_current_time():
    # Set the time zone to Australia/Sydney
    pacific_tz = pytz.timezone('Australia/Sydney')

    # Get the current time in that time zone
    current_time = datetime.now(pacific_tz)

    # Format the time string
    time_string = current_time.strftime("%I:%M %p")

    # Remove leading zero from hour if present
    if time_string.startswith("0"):
        time_string = time_string[1:]

    return f"The current time is {time_string}."


def get_joke():
    url = "https://official-joke-api.appspot.com/random_joke"
    response = requests.get(url)

    if response.status_code == 200:
        joke = response.json()
        setup = joke['setup']
        punchline = joke['punchline']
        return (
            f"Alright, here's one for you: {setup} ... {punchline} "
            f"Ha! I do hope my humor algorithms are functioning correctly."
        )
    else:
        return (
            "I'm sorry, my joke circuits seem to be malfunctioning. "
            "Perhaps I should stick to driving and crime-fighting."
        )


def main_loop():
    while True:
        if listen_for_wake_word(detector):
            time.sleep(0.2)  # Slight delay to ensure state reset
            continuous_conversation = True
            first_query = True
            while continuous_conversation:
                user_message_file = record_audio_with_silence_detection()
                user_message = transcribe_audio(user_message_file)
                print(f"User said: {user_message}")  # Print user's message for debugging
                bot_response = handle_query(user_message, play_waiting_mp3=first_query)
                first_query = False

                # Check if the bot's response ends with a question mark
                if not bot_response.strip().endswith('?'):
                    continuous_conversation = False
                    print("Conversation ended. Listening for wake word...")
                else:
                    print("KARR asked a question. Waiting for user response...")
            detector.reset()


if __name__ == "__main__":
    main_loop()
