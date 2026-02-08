# KARR AI - Knight Rider Assistant
> **Note**: This folk is still under construction, I will remove this statement when it has been completed and fully tested.
Welcome to the KARR AI project! This project brings the infamous KARR AI from the "Knight Rider" series to life. It uses various Python libraries to provide voice interaction, image processing, and information retrieval, emulating the functionality of the original KARR.

## Features

- **Voice Interaction**: Talk to KARR using voice commands powered by GPT-4o.
- **Wake Word Detection**: KARR can be activated using the wake word "Hey KARR".
- **Weather Updates**: Get current weather information and forecasts.
- **News Updates**: Receive the latest news.
- **Image Processing**: Capture and analyze images with KARR's witty commentary.

## Repository Structure

> **Note**: All scripts and MP3 files must be in the same folder for the project to work correctly.

KARR.py: The main script to run KARR AI.
modules/: Various helper modules such as weather.py and news.py.
sounds/: Various MP3 files used for waiting responses and sound effects.
chatbot1.txt: Stores conversation history for the AI.

## Requirements

To run this project, you'll need the following Python libraries:

Python 3.8 or higher
openai: For interacting with OpenAI's API.
elevenlabs: For text-to-speech functionality.
openwakeword: For wake word detection.
pyaudio: For audio input/output.
numpy: For numerical operations.
sounddevice: For real-time audio playback.
soundfile: For reading and writing sound files.
opencv-python: For image processing.
requests: For making HTTP requests.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/tusabez/KARR_AI_python.git
   cd KARR_AI_python
2. **Install Dependencies**

   ```bash
   pip install openai elevenlabs pyaudio numpy sounddevice soundfile opencv-python requests openwakeword pytz openmeteo-requests request-cache retry-requests
3. **Set Up Keys**

   OpenAI: Store your OpenAI API key in your environment variables or add it directly to the script.
   ElevenLabs: Store your ElevenLabs API key in your environment variables or add it directly to the script.

4. **Create Your KARR Voice**

   You will need to create your own KARR voice since the original voice cannot be legally distributed. You can do this by:

   Obtaining voice samples from the "Knight Rider" series.

   Using ElevenLabs' Instant Voice Clone feature to create a voice that mimics KARR.

   Note: Please do not publicly share the voice you create to avoid any potential copyright issues.

5. **Run the script**

   Start the main loop:

   ```bash
   python KARR.py
   
## Usage

Once the script is running, KARR will listen for the wake word "Hey KARR". After activation, you can interact with KARR using natural language commands such as:

"What's the weather right now?"
"Tell me a joke."
"What time is it?"
"What is the latest news?"
Since KARR is powered by GPT-4o, you can ask him about anything, and he'll respond in character as KARR from "Knight Rider."

## Compatibility

This project was tested on a Windows 11 computer, but it can be adapted to work on a Raspberry Pi 5 with some modifications to the code. I used a USB microphone and Raspberry Pi camera module 3. If you're planning to run this on a Raspberry Pi, you might need to adjust the setup for audio input/output, camera access, and performance optimizations. You'll also need to set up a virtual environment before you install the libraries. Update: (use KARR_Pi.py instead of KARR.py for the Raspberry Pi 5 as it has been optimized for it).

## Troubleshooting

Wake Word Detection Issues: Ensure your microphone is working correctly and that the wake word model is properly initialized.
API Key Errors: Double-check that your API keys are correctly stored and valid.
Run models.py to download needed models to have the wakeword work. This is only done once.

## Contributing

Feel free to submit issues or pull requests if you find bugs or want to add new features. Contributions are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for details.
