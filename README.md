# MOMO — Alzheimer's Companion Chatbot

A local, offline companion chatbot for older adults living with
dementia. Talks through a desktop UI, keeps conversation warm and
patient, and never sends data to the cloud.

## What it does

- Chat UI built with CustomTkinter (dark theme, large fonts)
- Local LLM (`qwen2.5:3b` via Ollama) customised with a behavioral
  prompt (`companion.md`) to act as a patient, warm companion
- Speech-to-text via faster-whisper (`small.en`)
- Text-to-speech via Piper (`en_US-amy-medium`)
- Zero cloud dependencies at runtime

## Requirements

- Windows 11 (or Linux/macOS with minor path tweaks)
- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- A working microphone and speakers

## Setup

1. Install Python dependencies:
pip install customtkinter faster-whisper sounddevice numpy scipy piper-tts
pip install playsound==1.2.2

text

2. Pull the base model:
ollama pull qwen2.5:3b

text

3. Build the custom companion model:
build_companion.bat

text

4. Download the Piper voice (into `voices/`):
mkdir voices
cd voices
curl.exe -L -o en_US-amy-medium.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
curl.exe -L -o en_US-amy-medium.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"
cd ..

text

## Run
python companion_ui.py

text

Type a message and press Enter. Or click the 🎤 button, speak,
and click it again to stop — the message is transcribed and sent.

## Project structure

| File | Purpose |
|---|---|
| `companion.md` | Behavioral guidelines — the system prompt |
| `build_companion.bat` | Builds the `momo` Ollama model from `companion.md` |
| `companion_ui.py` | The desktop app |
| `voices/` | Piper voice files (not committed) |

## Status

Work in progress. See project docs for details.

## Not a medical device

MOMO is a companionship tool. It is not a diagnostic system and
does not provide medical advice.