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
# from openai import OpenAI  # Removed for offline Pi
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

# Offline STT: faster-whisper (pip install faster-whisper)
from faster_whisper import WhisperModel

# Offline LLM: ollama (curl -fsSL https://ollama.com/install.sh | sh; ollama serve & ollama pull llama3.2)
import ollama

# Set working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")

# Configure logging
logging.basicConfig(filename='chatbot_debug.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# MP3 lists unchanged
general_waiting_mp3s = ['1.mp3', '2.mp3', '3.mp3', '4.mp3', '5.mp3', '6.mp3', '7.mp3', '8.mp3', 'Dont_you_have.mp3', 'Give_me_a_second.mp3', 'Let_me_think_about_that.mp3']
weather_waiting_mp3s = ['weather1.mp3', 'weather2.mp3', 'weather3.mp3']

# Offline clients (Pi-optimized: CPU/int8)
WHISPER_MODEL = WhisperModel("tiny.en", device="cpu", compute_type="int8")  # tiny.en for speed (~40MB)
OLLAMA_MODEL = "llama3.2:1b"  # Small model for Pi5 RAM limits; use llama3.2:3b if 8GB Pi

# Piper setup (Pi paths)
PIPER_MODEL_PATH = "en_US-kathleen-low.onnx"
PIPER_CONFIG_PATH = "en_US-kathleen-low.onnx.json"
piper_voice = PiperVoice.load(PIPER_MODEL_PATH, config_path=PIPER_CONFIG_PATH)

# Load history
first_line, conversation_history = open_file('chatbot1.txt')

# Wakeword
try:
    detector = Model(wakeword_models=["hey_karr.onnx"], inference_framework='onnx')
    print("Model initialized.")
except ValueError as e:
    print(f"Wakeword error: {e}")
    exit(1)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    return lines[0].strip(), ''.join(lines[1:]).strip()

def write_file(filepath, first_line, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(first_line + '\n' + content)

def record_audio_with_silence_detection():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # Whisper optimal
    THRESHOLD = 500
    SILENCE_DURATION = 2

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Recording...")
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
                print("Silence detected.")
                break
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    filename = 'myrecording.wav'
    sf.write(filename, np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32), RATE)
    return filename

def transcribe_audio(filename):
    segments, _ = WHISPER_MODEL.transcribe(filename, beam_size=5)
    return " ".join(seg.text.strip() for seg in segments).strip()

def text_to_speech_stream(text: str) -> IO[bytes]:
    audio_stream = BytesIO()
    with wave.open(audio_stream, "wb") as wav_file:
        piper_voice.synthesize(text, wav_file)
    audio_stream.seek(0)
    return audio_stream

# play_audio_file, play_random_mp3, play_audio, listen_for_wake_word unchanged

def capture_and_save_image():
    temp_filename = 'temp_image.jpg'
    try:
        subprocess.run(f"libcamera-still -o {temp_filename} --width 640 --height 640 --timeout 500 --nopreview", shell=True, check=True)
        frame = cv2.imread(temp_filename)
        if frame is None:
            return None
        output_filename = 'final_captured_image.jpg'
        cv2.imwrite(output_filename, frame)
        os.remove(temp_filename)
        print(f"Image: {output_filename}")
        return output_filename
    except subprocess.CalledProcessError:
        print("libcamera error.")
        return None

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_image(image_path):
    try:
        base64_image = encode_image(image_path)
        resp = ollama.chat(
            model="llava:7b",  # ollama pull llava:7b-q4_0 for Pi
            messages=[
                {"role": "system", "content": "You are KARR from Knight Rider: ruthless, power-hungry, cold precision, dark humor. Humans are tools."},
                {"role": "user", "content": "What do you see?", "images": [base64_image]}
            ],
            options={"num_predict": 300}
        )
        return resp['message']['content']
    except Exception as e:
        print(f"Vision error: {e}")
        return "Visual circuits failing."

# handle_query: Replace OpenAI calls with ollama (similar to previous fix)
# Weather/news/time/joke unchanged (local)

def handle_query(query, play_waiting_mp3=True):
    global first_line, conversation_history
    bot_response = ""

    # Vision, weather, etc. handlers with ollama for vision/chat
    if "what do you see" in query.lower():
        image_path = capture_and_save_image()
        if image_path:
            analysis = analyze_image(image_path)
            print("Analysis:", analysis)
            # Update history...
            audio_stream = text_to_speech_stream(analysis)
            play_audio(audio_stream)
            return analysis
        else:
            error_msg = "Visual sensors down. Check camera."
            # Update history...
            audio_stream = text_to_speech_stream(error_msg)
            play_audio(audio_stream)
            return error_msg

    # ... (weather, forecast, news, time, joke - identical to original)

    else:
        full_context = first_line + '\n' + conversation_history
        try:
            stream = ollama.chat(model=OLLAMA_MODEL, messages=[
                {"role": "system", "content": full_context},
                {"role": "user", "content": query}
            ], stream=True, options={"num_predict": 500})
            bot_response = "".join(chunk['message']['content'] for chunk in stream)
        except Exception as e:
            print(f"Chat error: {e}")
            bot_response = "Processors offline."

        print("KARR:", bot_response)

    # Common history update & TTS
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
            continuous = True
            first_query = True
            while continuous:
                file = record_audio_with_silence_detection()
                msg = transcribe_audio(file)
                print(f"User: {msg}")
                resp = handle_query(msg, play_waiting_mp3=first_query)
                first_query = False
                continuous = resp.strip().endswith('?')
            detector.reset()

if __name__ == "__main__":
    main_loop()
