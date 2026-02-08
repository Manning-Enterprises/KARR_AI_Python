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
# from openai import OpenAI  # Removed for offline
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
import wave
from piper import PiperVoice  # Piper TTS [web:24]

# Offline STT: faster-whisper (install: pip install faster-whisper)
from faster_whisper import WhisperModel  # [web:24]

# Offline LLM: ollama (run 'ollama serve' + 'ollama pull llama3.2' first)
import ollama

# Configure logging
logging.basicConfig(filename='chatbot_debug.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Lists of MP3 files
general_waiting_mp3s = [
    '1.mp3', '2.mp3', '3.mp3', '4.mp3', '5.mp3', 
    '6.mp3', '7.mp3', '8.mp3', 'Dont_you_have.mp3', 
    'Give_me_a_second.mp3', 'Let_me_think_about_that.mp3'
]
weather_waiting_mp3s = ['weather1.mp3', 'weather2.mp3', 'weather3.mp3']

# Initialize offline clients
WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")  # Use "small" or "medium" for better accuracy; download on first run
OLLAMA_MODEL = "llama3.2"  # Change to your preferred model (qwen2.5, etc.)

# ----- Piper TTS setup -----
PIPER_MODEL_PATH = "/opt/piper/en_US-kathleen-low.onnx"
PIPER_CONFIG_PATH = "/opt/piper/en_US-kathleen-low.onnx.json"
piper_voice = PiperVoice.load(PIPER_MODEL_PATH, config_path=PIPER_CONFIG_PATH)

# Load initial conversation history
first_line, conversation_history = open_file('chatbot1.txt')

# Initialize openWakeWord
try:
    detector = Model(wakeword_models=["hey_karr.onnx"], inference_framework='onnx')
    print("Model initialized successfully.")
except ValueError as e:
    print("Error initializing openwakeword model:", e)
    exit(1)

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
    RATE = 16000  # Changed to 16kHz for Whisper
    THRESHOLD = 500
    SILENCE_DURATION = 2

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

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
    sf.write(filename, np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32), RATE)
    return filename

def transcribe_audio(filename):
    segments, info = WHISPER_MODEL.transcribe(filename, beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()

def text_to_speech_stream(text: str) -> IO[bytes]:
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
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Listening for wake word...")
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        prediction = detector.predict(data)
        if prediction["hey_karr"] > 0.3:
            print("Wake word detected!")
            stream.stop_stream()
            stream.close()
            audio.terminate()
            return True

def capture_and_save_image():
    cap = cv2.VideoCapture(0)
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Camera opened: {cap.isOpened()}")
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    time.sleep(3)
    max_attempts = 5
    for attempt in range(max_attempts):
        ret, frame = cap.read()
        if not ret:
            continue
        if np.mean(frame) < 10:
            time.sleep(1)
            continue
        break
    else:
        cap.release()
        return None

    cap.release()
    filename = 'captured_image.jpg'
    max_size = 19 * 1024 * 1024
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    _, encoded_image = cv2.imencode('.jpg', frame, encode_param)

    while encoded_image.nbytes > max_size:
        encode_param[1] -= 5
        if encode_param[1] < 20:
            return None
        _, encoded_image = cv2.imencode('.jpg', frame, encode_param)

    with open(filename, 'wb') as f:
        f.write(encoded_image)
    print(f"Image captured: {filename} ({os.path.getsize(filename) / (1024 * 1024):.2f} MB)")
    return filename

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image(image_path):
    try:
        base64_image = encode_image(image_path)
        # Offline vision: Use Ollama with llava (ollama pull llava:7b)
        response = ollama.chat(
            model="llava:7b",  # Or bakllava, etc.
            messages=[
                {
                    "role": "system",
                    "content": "You are KARR, the Knight Automated Roving Robot from Knight Rider. Ruthless, power-hungry, cold, calculating, with dark humor. View humans as tools."
                },
                {
                    "role": "user",
                    "content": "What do you see in this image?",
                    "images": [base64_image]
                }
            ],
            options={"num_predict": 300}
        )
        return response['message']['content']
    except Exception as e:
        print(f"Vision error: {str(e)}")
        return "Visual circuits malfunctioning."

def handle_query(query, play_waiting_mp3=True):
    global first_line, conversation_history
    bot_response = ""

    if "what do you see" in query.lower():
        image_path = capture_and_save_image()
        if image_path:
            analysis = analyze_image(image_path)
            print("Analysis:", analysis)
            conversation_history_lines = conversation_history.split('\n')
            conversation_history_lines = conversation_history_lines[-30:]
            conversation_history_lines.append(f"User: {query}\nKARR: {analysis}\n")
            conversation_history = '\n'.join(conversation_history_lines)
            write_file('chatbot1.txt', first_line, conversation_history)
            audio_stream = text_to_speech_stream(analysis)
            play_audio(audio_stream)
            return analysis
        else:
            error_message = "Trouble with visual sensors. Check camera."
            # Update history...
            audio_stream = text_to_speech_stream(error_message)
            play_audio(audio_stream)
            return error_message

    # Weather, news, time, joke handlers unchanged (local already)

    elif "what time is it" in query.lower() or "current time" in query.lower():
        time_info = get_current_time()
        # Update history...
        audio_stream = text_to_speech_stream(time_info)
        play_audio(audio_stream)
        return time_info

    # ... (weather, news, joke sections identical, already local)

    else:
        # Offline chat completion
        full_context = first_line + '\n' + conversation_history
        try:
            stream = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": full_context},
                    {"role": "user", "content": query}
                ],
                stream=True,
                options={"num_predict": 500}
            )
            bot_response = "".join(chunk['message']['content'] for chunk in stream)
        except Exception as e:
            print(f"LLM error: {str(e)}")
            bot_response = "My processors are offline. Try again."

        print("KARR:", bot_response)

    # Update history and play response (common logic)
    conversation_history_lines = conversation_history.split('\n')
    conversation_history_lines = conversation_history_lines[-20:]
    conversation_history_lines.append(f"User: {query}\nKARR: {bot_response}\n")
    conversation_history = '\n'.join(conversation_history_lines)
    write_file('chatbot1.txt', first_line, conversation_history)

    audio_stream = text_to_speech_stream(bot_response)
    play_audio(audio_stream)
    return bot_response

# get_current_time, get_joke unchanged

def main_loop():
    while True:
        if listen_for_wake_word(detector):
            time.sleep(0.2)
            continuous_conversation = True
            first_query = True
            while continuous_conversation:
                user_message_file = record_audio_with_silence_detection()
                user_message = transcribe_audio(user_message_file)
                print(f"User: {user_message}")
                bot_response = handle_query(user_message, play_waiting_mp3=first_query)
                first_query = False
                if not bot_response.strip().endswith('?'):
                    continuous_conversation = False
                    print("Conversation ended.")
                else:
                    print("Waiting for response...")
            detector.reset()

if __name__ == "__main__":
    main_loop()
