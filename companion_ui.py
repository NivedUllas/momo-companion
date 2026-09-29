"""
companion_ui.py — CustomTkinter chat UI for the Momo Alzheimer's companion.

Adds: push-to-talk STT (faster-whisper) and TTS (Piper).
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import urllib.request
from datetime import datetime

import numpy as np
import sounddevice as sd
import tkinter as tk
import customtkinter as ctk


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "momo"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

MAX_HISTORY_TURNS = 10

COMPANION_NAME = "Momo"
PATIENT_NAME = ""
CAREGIVER_NAME = ""

# --- Audio ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

WHISPER_MODEL_SIZE = "small.en"

PIPER_MODEL = os.path.join("voices", "en_US-amy-medium.onnx")
PIPER_CONFIG = os.path.join("voices", "en_US-amy-medium.onnx.json")

ENABLE_STT = True
ENABLE_TTS = True


# ============================================================
# APPEARANCE
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG_DARK       = "#0a0a0a"
PANEL_DARK    = "#141414"
PANEL_BORDER  = "#1f1f1f"
CHAT_BG       = "#0f0f0f"
INPUT_BG      = "#1a1a1a"

TEXT_PRIMARY  = "#f5f5f5"
TEXT_MUTED    = "#8a8a8a"

USER_NAME     = "#5aa9ff"
MOMO_NAME     = "#66d07a"
ERROR_COLOR   = "#d97777"
TYPING_COLOR  = "#7a7a7a"
REC_COLOR     = "#e07777"

FONT_BODY     = ("Segoe UI", 16)
FONT_NAME     = ("Segoe UI", 15, "bold")
FONT_SYSTEM   = ("Segoe UI", 13, "italic")
FONT_TYPING   = ("Segoe UI", 14, "italic")
FONT_INPUT    = ("Segoe UI", 16)
FONT_BUTTON   = ("Segoe UI", 16, "bold")


# ============================================================
# CONTEXT
# ============================================================

def build_context_block():
    now = datetime.now()
    lines = ["[CONTEXT]"]
    lines.append(f"Time: {now.strftime('%I:%M %p').lstrip('0')}")
    lines.append(f"Day: {now.strftime('%A, %B %d, %Y')}")
    lines.append(f"Companion name: {COMPANION_NAME}")
    if PATIENT_NAME:
        lines.append(f"Patient name: {PATIENT_NAME}")
    if CAREGIVER_NAME:
        lines.append(f"Caregiver name: {CAREGIVER_NAME}")
    return "\n".join(lines)


def wrap_user_message(user_text):
    return f"{build_context_block()}\n\n{user_text}"


# ============================================================
# OLLAMA
# ============================================================

def stream_reply(messages, on_chunk):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    full = ""
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            for line in response:
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    full += piece
                    on_chunk(piece)
                if chunk.get("done"):
                    break
    except Exception as e:
        return f"[error] {e}"
    return full


# ============================================================
# WHISPER (STT)
# ============================================================

print("Loading Whisper model...")
from faster_whisper import WhisperModel
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
print("Whisper loaded.")


def transcribe(audio_int16):
    audio_float = audio_int16.astype(np.float32) / 32768.0
    segments, _ = whisper_model.transcribe(
        audio_float,
        language="en",
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    return " ".join(s.text for s in segments).strip()


# ============================================================
# PIPER (TTS)
# ============================================================

def synthesize_to_wav(text, out_path):
    if not os.path.isfile(PIPER_MODEL):
        raise FileNotFoundError(f"Piper model not found: {PIPER_MODEL}")

    cmd = [
        sys.executable, "-m", "piper",
        "--model", PIPER_MODEL,
        "--config", PIPER_CONFIG,
        "--output_file", out_path,
    ]
    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Piper failed: {proc.stderr.decode(errors='replace')}")


def play_wav(path):
    try:
        import simpleaudio as sa
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        return
    except ImportError:
        pass

    try:
        subprocess.run(
            ["powershell", "-c",
             f'(New-Object Media.SoundPlayer "{path}").PlaySync()'],
            timeout=60,
        )
    except Exception as e:
        print(f"[tts] playback failed: {e}")


# ============================================================
# UI
# ============================================================

class CompanionUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(f"{COMPANION_NAME} — Companion")
        self.geometry("900x760")
        self.minsize(700, 600)
        self.configure(fg_color=BG_DARK)

        self.history = []
        self.busy = False
        self.typing_dots = 0
        self.typing_after_id = None
        self._status_prefix = ""

        self.recording = False
        self.recorded_chunks = []
        self.mic_stream = None

        self._build_widgets()
        self._greet()

    def _build_widgets(self):
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

        self.entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Type your message...",
            placeholder_text_color=TEXT_MUTED,
            font=FONT_INPUT,
            height=56,
            corner_radius=14,
            border_width=1,
            border_color=PANEL_BORDER,
            fg_color=INPUT_BG,
            text_color=TEXT_PRIMARY,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 12))
        self.entry.bind("<Return>", lambda e: self.on_send())

        self.mic_btn = ctk.CTkButton(
            input_frame,
            text="🎤",
            font=("Segoe UI", 20),
            width=64,
            height=56,
            corner_radius=14,
            fg_color="#2a2a2a",
            hover_color="#3a3a3a",
            text_color="#ffffff",
            command=self.on_mic_toggle,
        )
        self.mic_btn.pack(side=tk.RIGHT, padx=(0, 8))

        self.send_btn = ctk.CTkButton(
            input_frame,
            text="Send",
            font=FONT_BUTTON,
            width=100,
            height=56,
            corner_radius=14,
            fg_color="#2f7ddf",
            hover_color="#3a8ef0",
            text_color="#ffffff",
            command=self.on_send,
        )
        self.send_btn.pack(side=tk.RIGHT)

        self.typing_label = ctk.CTkLabel(
            self,
            text="",
            font=FONT_TYPING,
            text_color=TYPING_COLOR,
            anchor="w",
        )
        self.typing_label.pack(side=tk.BOTTOM, fill=tk.X, padx=32, pady=(0, 4))

        chat_wrapper = ctk.CTkFrame(
            self,
            fg_color=PANEL_DARK,
            corner_radius=18,
            border_width=1,
            border_color=PANEL_BORDER,
        )
        chat_wrapper.pack(
            side=tk.TOP, fill=tk.BOTH, expand=True,
            padx=20, pady=(20, 8),
        )

        self.chat = tk.Text(
            chat_wrapper,
            wrap=tk.WORD,
            bg=CHAT_BG,
            fg=TEXT_PRIMARY,
            font=FONT_BODY,
            borderwidth=0,
            highlightthickness=0,
            padx=24,
            pady=20,
            state=tk.DISABLED,
            cursor="arrow",
            insertbackground=TEXT_PRIMARY,
            spacing1=2,
            spacing3=8,
            takefocus=0,
        )
        self.chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ctk.CTkScrollbar(
            chat_wrapper,
            command=self.chat.yview,
            width=14,
            corner_radius=8,
            fg_color="transparent",
            button_color="#2a2a2a",
            button_hover_color="#3a3a3a",
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        self.chat.configure(yscrollcommand=scrollbar.set)

        self.chat.tag_config("user_name", foreground=USER_NAME, font=FONT_NAME, spacing1=14)
        self.chat.tag_config("momo_name", foreground=MOMO_NAME, font=FONT_NAME, spacing1=14)
        self.chat.tag_config("body", foreground=TEXT_PRIMARY, font=FONT_BODY)
        self.chat.tag_config("system", foreground=TEXT_MUTED, font=FONT_SYSTEM, spacing1=10)
        self.chat.tag_config("error", foreground=ERROR_COLOR, font=FONT_SYSTEM, spacing1=10)

        self.after(100, self.entry.focus_set)
        self.bind("<FocusIn>", lambda e: self.entry.focus_set())

    def _append(self, name, text, name_tag):
        self.chat.config(state=tk.NORMAL)
        if name:
            self.chat.insert(tk.END, f"{name}\n", name_tag)
        self.chat.insert(tk.END, f"{text}\n", "body")
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def _append_system(self, text, tag="system"):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{text}\n", tag)
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def _start_momo_line(self):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{COMPANION_NAME}\n", "momo_name")
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def _stream_chunk(self, piece):
        if self.typing_after_id is not None:
            self._stop_status()
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, piece, "body")
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def _end_momo_line(self):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, "\n", "body")
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def _start_status(self, prefix):
        self._status_prefix = prefix
        self.typing_dots = 0
        self._animate_status()

    def _animate_status(self):
        self.typing_dots = (self.typing_dots + 1) % 4
        dots = "." * self.typing_dots
        try:
            self.typing_label.configure(text=f"{self._status_prefix}{dots}")
        except Exception:
            return
        self.typing_after_id = self.after(400, self._animate_status)

    def _stop_status(self):
        if self.typing_after_id is not None:
            try:
                self.after_cancel(self.typing_after_id)
            except Exception:
                pass
            self.typing_after_id = None
        try:
            self.typing_label.configure(text="")
        except Exception:
            pass

    def _greet(self):
        hour = datetime.now().hour
        if hour < 12:
            greeting = "Good morning."
        elif hour < 17:
            greeting = "Good afternoon."
        else:
            greeting = "Good evening."
        self._append(
            COMPANION_NAME,
            f"{greeting} I'm here whenever you'd like to talk.",
            "momo_name",
        )

    def on_mic_toggle(self):
        if not ENABLE_STT:
            self._append_system("[STT disabled]")
            return
        if self.recording:
            self._stop_recording_and_transcribe()
        else:
            self._start_recording()

    def _start_recording(self):
        if self.busy:
            return
        self.recording = True
        self.recorded_chunks = []

        self.mic_btn.configure(fg_color=REC_COLOR, hover_color="#f08a8a")
        self._start_status("Listening")

        def callback(indata, frames, time_info, status):
            self.recorded_chunks.append(indata.copy())

        self.mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=callback,
        )
        self.mic_stream.start()

    def _stop_recording_and_transcribe(self):
        self.recording = False

        if self.mic_stream is not None:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass
            self.mic_stream = None

        self.mic_btn.configure(fg_color="#2a2a2a", hover_color="#3a3a3a")
        self._stop_status()

        if not self.recorded_chunks:
            return

        audio = np.concatenate(self.recorded_chunks, axis=0).flatten()

        if len(audio) < SAMPLE_RATE * 0.3:
            return

        self._start_status("Transcribing")
        threading.Thread(
            target=self._transcribe_and_send,
            args=(audio,),
            daemon=True,
        ).start()

    def _transcribe_and_send(self, audio):
        try:
            text = transcribe(audio)
        except Exception as e:
            self.after(0, self._stop_status)
            self.after(0, lambda: self._append_system(f"[stt error] {e}", "error"))
            return

        self.after(0, self._stop_status)

        if not text:
            self.after(0, lambda: self._append_system("[heard nothing]"))
            return

        self.after(0, lambda: self._send_text(text))

    def _send_text(self, text):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, text)
        self.on_send()

    def on_send(self):
        if self.busy:
            return

        user_text = self.entry.get().strip()
        if not user_text:
            return

        self.entry.delete(0, tk.END)

        if user_text.lower() in ("/exit", "/quit"):
            self.destroy()
            return
        if user_text.lower() == "/clear":
            self.history = []
            self.chat.config(state=tk.NORMAL)
            self.chat.delete("1.0", tk.END)
            self.chat.config(state=tk.DISABLED)
            self._append_system("[conversation cleared]")
            self._greet()
            return

        self._append("You", user_text, "user_name")

        wrapped = wrap_user_message(user_text)
        self.history.append({"role": "user", "content": wrapped})

        self.busy = True
        self.send_btn.configure(state=tk.DISABLED)
        self.entry.configure(state=tk.DISABLED)

        self._start_momo_line()
        self._start_status(f"{COMPANION_NAME} is thinking")

        threading.Thread(target=self._do_reply, daemon=True).start()

    def _do_reply(self):
        reply = stream_reply(self.history, on_chunk=self._stream_chunk_ui)
        self.after(0, self._finish_reply, reply)

    def _stream_chunk_ui(self, piece):
        self.after(0, self._stream_chunk, piece)

    def _finish_reply(self, reply):
        if self.typing_after_id is not None:
            self._stop_status()

        self._end_momo_line()

        if reply and not reply.startswith("[error]"):
            self.history.append({"role": "assistant", "content": reply})
            self.history = self.history[-MAX_HISTORY_TURNS * 2:]

            if ENABLE_TTS:
                threading.Thread(
                    target=self._speak,
                    args=(reply,),
                    daemon=True,
                ).start()
            else:
                self._release_input()
        else:
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            if reply:
                self._append_system(reply, tag="error")
            self._release_input()

    def _release_input(self):
        self.busy = False
        self.send_btn.configure(state=tk.NORMAL)
        self.entry.configure(state=tk.NORMAL)
        self.after(50, self.entry.focus_set)

    def _speak(self, text):
        self.after(0, lambda: self._start_status(f"{COMPANION_NAME} is speaking"))
        try:
            tmp = os.path.join(tempfile.gettempdir(), "momo_reply.wav")
            synthesize_to_wav(text, tmp)
            play_wav(tmp)
        except Exception as e:
            self.after(0, lambda: self._append_system(f"[tts error] {e}", "error"))
        finally:
            self.after(0, self._stop_status)
            self.after(0, self._release_input)


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    app = CompanionUI()
    app.mainloop()


if __name__ == "__main__":
    main()