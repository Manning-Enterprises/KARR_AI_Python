# KARR AI – Knight Rider Assistant

> **Note**: This fork is still under construction. I will remove this statement when it has been completed and fully tested.

KARR AI brings the infamous KARR from *Knight Rider* into a modern voice assistant. It uses Python to provide wake-word activation, voice interaction, image processing, and information retrieval, with KARR responding in-character.

The **main branch** uses OpenAI + ElevenLabs (cloud TTS).  
The **piper-tts branch** provides a **fully local/offline** version using Piper TTS, faster-whisper, and Ollama-compatible LLMs.

## Features

- **Voice interaction**  
  - Main: GPT-4o + ElevenLabs voice.  
  - Piper: local LLM (via Ollama) + Piper TTS (no API limits).
- **Wake word detection** – “Hey KARR” using openWakeWord.  
- **Weather & forecasts** – Local Python modules using Open-Meteo API.  
- **News updates** – Latest headlines via a simple news module.  
- **Image processing**  
  - Main: GPT-4o vision.  
  - Piper: local vision models (e.g., LLaVA via Ollama) when configured.

## Repository Structure

> **Note**: All scripts, models, and MP3 files must remain in the project folder structure for KARR to work correctly.

- `KARR.py` – Main Windows-focused script (cloud ElevenLabs + OpenAI).  
- `pi.py` / `KARR_Pi.py` – Raspberry Pi–optimized scripts (Piper + local stack).  
- `modules/` – Helper modules (`weather.py`, `news.py`, etc.).  
- `sounds/` – MP3 files for waiting responses and SFX.  
- `chatbot1.txt` – Conversation history and system prompt.  
- `models.py` – Downloads wakeword and related models (e.g., `hey_karr.onnx`).  

Branches:

- `main` – OpenAI + ElevenLabs (internet required).  
- `piper-tts` – Fully local Piper TTS, faster-whisper, Ollama LLM/vision.

## Requirements

### Common

- Python 3.10+  
- A decent microphone and speakers (USB recommended)  
- Git

### Main (cloud) branch

- `openai` – Chat & vision (GPT-4o).  
- `elevenlabs` – Text-to-speech.  
- `openwakeword`, `pyaudio`, `numpy`, `sounddevice`, `soundfile`, `opencv-python`, `requests`, `pytz`, `openmeteo-requests`, `requests-cache`, `retry-requests`.

### Piper-TTS (offline) branch

- `piper-tts` – Local neural TTS.  
- `faster-whisper` – Local speech-to-text.  
- `ollama` – Local LLM and optional vision (LLaVA etc.).  
- Same audio, wakeword, and utility libs as above.

## Installation – Main (ElevenLabs/OpenAI)

1. **Clone**

   ```bash
   git clone https://github.com/Manning_Enterprises/KARR_AI_python.git
   cd KARR_AI_python
   ```

2. **Install dependencies**

   ```bash
   pip install openai elevenlabs pyaudio numpy sounddevice soundfile opencv-python \
               requests openwakeword pytz openmeteo-requests requests-cache retry-requests
   ```

3. **Set up keys**

   - `OPENAI_API_KEY` – Environment variable or inside your config.  
   - `ELEVENLABS_API_KEY` – Environment variable or config.

4. **Create your KARR voice (main branch)**

   - Collect KARR audio samples from *Knight Rider*.  
   - Use ElevenLabs Instant Voice Clone to create a similar voice.  
   - Keep this voice private to avoid copyright issues.

5. **Run**

   ```bash
   python KARR.py
   ```

## Installation – Piper-TTS (Fully Local)

For detailed, step-by-step guides see the wiki pages:

- **Windows 11 – Piper-TTS Setup**  
- **Raspberry Pi 5 – Piper-TTS Setup**

High-level:

1. **Checkout piper-tts branch**

   ```bash
   git clone https://github.com/Manning_Enterprises/KARR_AI_python.git
   cd KARR_AI_python
   git checkout piper-tts
   ```

2. **Create venv & install**

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate # Linux / Pi

   pip install faster-whisper piper-tts pyaudio numpy sounddevice soundfile \
               opencv-python requests openwakeword pytz openmeteo-requests \
               requests-cache retry-requests ollama
   ```

3. **Install/configure Piper voice**

   - Download a Piper `.onnx` + `.onnx.json` into a `piper_voices` folder.  
   - Set `PIPER_MODEL_PATH` and `PIPER_CONFIG_PATH` in `KARR.py` / `pi.py`.

4. **Set up Ollama and models**

   - Install Ollama (Windows or Pi).  
   - Pull models such as `llama3.2` (chat) and `llava` (vision).  
   - Configure `OLLAMA_MODEL` in the Piper scripts.

5. **Wakeword models**

   ```bash
   python models.py
   ```

6. **Run (offline)**

   ```bash
   python KARR.py   # Windows piper-tts
   # or
   python pi.py     # Raspberry Pi 5 piper-tts
   ```

Once started, KARR listens for “Hey KARR” and handles chat, weather, time, jokes, and (optionally) local image description completely offline after models are downloaded.

## Usage

Examples after saying **“Hey KARR”**:

- “What’s the weather right now?”  
- “Tell me a joke.”  
- “What time is it?”  
- “What’s the latest news?”  
- “What do you see?” (with camera, main = GPT-4o vision, piper = Ollama vision model).

The main branch uses GPT-4o, while the piper-tts branch uses a local LLM but both respond in-character as KARR.

## Compatibility

- **Windows 11** – Primary development/test platform (`KARR.py`).  
- **Raspberry Pi 5** – Supported via `KARR_Pi.py` / `pi.py` plus dedicated Piper/Ollama/faster-whisper setup guides in the wiki.

You may need to adjust:

- Audio input/output devices (mic and speakers).  
- Camera configuration (`libcamera` on Pi vs USB cams on Windows).  
- Model sizes depending on CPU/RAM (smaller Ollama models on Pi).
## Troubleshooting

- **Wake word not detected**: Check mic in OS settings, ensure `hey_karr.onnx` exists (run `models.py` once), and confirm openWakeWord loads without errors.  
- **No audio**: Verify device selection (`arecord -l / aplay -l` on Pi, Windows Sound settings), and confirm PyAudio/sounddevice see the devices.  
- **Model errors**:  
  - Main: Confirm `OPENAI_API_KEY` and ElevenLabs key.  
  - Piper: Check Piper paths, that Ollama is running, and that models are pulled.

## Contributing

Issues and pull requests are welcome—especially around:

- Additional KARR behaviors and “in-character” responses.  
- More hardware configs (other SBCs, different mics/cams).  
- Performance tuning for Pi/low-power hardware.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
