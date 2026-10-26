"""
companion_ui.py - MOMO companion (redesigned UI, same backend)
Big mic button, readable chat bubbles, live date/time, auto-stop on silence.
"""
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
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

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
WHISPER_MODEL_SIZE = "small.en"

PIPER_MODEL = os.path.join("voices", "en_US-lessac-medium.onnx")
PIPER_CONFIG = os.path.join("voices", "en_US-lessac-medium.onnx.json")

ENABLE_STT = True
ENABLE_TTS = True

SILENCE_END_SECONDS = 1.8     # stop listening after this much silence
NO_SPEECH_TIMEOUT = 8         # give up if nobody speaks
MAX_RECORD_SECONDS = 25
MIN_VOICE_RMS = 300           # minimum loudness counted as speech

# ============================================================
# THEME - calm, high contrast, large type
# ============================================================
ctk.set_appearance_mode("light")

BG_MAIN = "#EEF3F5"
BG_PANEL = "#FFFFFF"
BORDER = "#D5DEE3"
TEXT_MAIN = "#1B2A33"
TEXT_MUTED = "#5B6B75"

ACCENT = "#1F6F8B"
ACCENT_HOVER = "#185A72"
MOMO_BUBBLE = "#FFF0C2"
USER_BUBBLE = "#27496D"
USER_TEXT = "#FFFFFF"

C_LISTEN = "#C8443B"
C_LISTEN_ALT = "#A93027"
C_THINK = "#D99A1E"
C_SPEAK = "#3E8E5A"
C_OFF = "#AEB9BF"

FONT_BODY = ("Segoe UI", 21)
FONT_STATUS = ("Segoe UI", 20, "bold")

# mode -> (status text, mic colour, mic icon)
MODES = {
    "loading":      ("Getting ready...", C_OFF, "🎙"),
    "idle":         ("Tap the microphone and talk to me", ACCENT, "🎙"),
    "listening":    ("I'm listening... tap when you're done", C_LISTEN, "⏹"),
    "transcribing": ("One moment...", C_THINK, "⏳"),
    "thinking":     ("Thinking...", C_THINK, "💭"),
    "speaking":     ("Momo is speaking...", C_SPEAK, "🔊"),
}

# ============================================================
# BACKEND HELPERS
# ============================================================
def build_context_block():
    now = datetime.now()
    return (f"[CONTEXT]\nDate: {now.strftime('%A, %d %B %Y')}\n"
            f"Time: {now.strftime('%I:%M %p').lstrip('0')}\nCompanion: {COMPANION_NAME}")


def stream_reply(messages, on_chunk):
    """Streams the reply from Ollama. Raises on connection errors."""
    payload = {"model": MODEL_NAME, "messages": messages, "stream": True}
    request = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    full = ""
    with urllib.request.urlopen(request, timeout=300) as response:
        for line in response:
            if not line.strip():
                continue
            try:
                chunk = json.loads(line.decode())
            except ValueError:
                continue
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                full += piece
                on_chunk(piece)
            if chunk.get("done"):
                break
    return full


whisper_model = None


def load_whisper():
    global whisper_model
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


def trim_silence(audio, threshold=300):
    loud = np.where(np.abs(audio) > threshold)[0]
    if len(loud) == 0:
        return audio
    pad = int(SAMPLE_RATE * 0.3)
    return audio[max(0, loud[0] - pad): min(len(audio), loud[-1] + pad)]


def transcribe(audio_i16):
    audio = trim_silence(audio_i16)
    audio_f = audio.astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(audio_f))) if audio_f.size else 0.0
    if 0 < peak < 0.5:                       # boost quiet speakers
        audio_f = audio_f * min(0.8 / peak, 8.0)
    segments, _ = whisper_model.transcribe(
        audio_f, language="en", beam_size=5, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=600),
        condition_on_previous_text=False)
    return " ".join(s.text for s in segments).strip()


def clean_for_speech(text):
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[*_#`~>]", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def synthesize_and_play(text):
    text = clean_for_speech(text)
    if not text:
        return
    tmp = os.path.join(tempfile.gettempdir(), f"momo_{os.getpid()}.wav")
    cmd = [sys.executable, "-m", "piper", "--model", PIPER_MODEL,
           "--config", PIPER_CONFIG, "--output_file", tmp]
    flags = 0x08000000 if sys.platform == "win32" else 0   # no console flash
    result = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True,
                            creationflags=flags)
    if result.returncode != 0 or not os.path.exists(tmp):
        raise RuntimeError(result.stderr.decode(errors="ignore")[:300])
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(tmp, winsound.SND_FILENAME)
    else:
        import simpleaudio as sa
        sa.WaveObject.from_wave_file(tmp).play().wait_done()


# ============================================================
# UI
# ============================================================
class CompanionUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Momo")
        self.geometry("1000x840")
        self.minsize(640, 660)
        self.configure(fg_color=BG_MAIN)

        self.history = []
        self.mode = "loading"
        self.stt_ready = False
        self.ui_queue = queue.Queue()      # worker threads -> UI thread
        self.msg_labels = []
        self.mic_stream = None
        self.recorded_chunks = []
        self.noise_blocks = []
        self.voice_threshold = MIN_VOICE_RMS
        self.has_spoken = False
        self.rec_start = 0.0
        self.last_voice = 0.0
        self._pulse_on = None
        self._wrap_job = None
        self._last_clock = ""

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._build_header()
        self._build_chat()
        self._build_controls()
        self.bind("<Configure>", self._on_resize)

        self._set_mode("loading")
        self._greet()
        self._tick()
        threading.Thread(target=self._load_models, daemon=True).start()

    # ---------------- layout ----------------
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=96)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(header, text="M", font=("Segoe UI", 26, "bold"), text_color="white",
                     fg_color=ACCENT, width=60, height=60, corner_radius=30
                     ).grid(row=0, column=0, rowspan=2, padx=(28, 16), pady=18)
        ctk.CTkLabel(header, text="Momo", font=("Segoe UI", 28, "bold"),
                     text_color=TEXT_MAIN, anchor="w").grid(row=0, column=1, sticky="sw", pady=(22, 0))
        ctk.CTkLabel(header, text="Your friendly companion", font=("Segoe UI", 15),
                     text_color=TEXT_MUTED, anchor="w").grid(row=1, column=1, sticky="nw")

        self.clock_label = ctk.CTkLabel(header, text="", font=("Segoe UI", 19), text_color=TEXT_MAIN,
                                        justify="right")
        self.clock_label.grid(row=0, column=3, rowspan=2, padx=(10, 20))
        ctk.CTkButton(header, text="New chat", font=("Segoe UI", 16), width=110, height=42,
                      corner_radius=21, fg_color="transparent", border_width=2, border_color=BORDER,
                      text_color=TEXT_MAIN, hover_color=BG_MAIN, command=self._clear_chat
                      ).grid(row=0, column=4, rowspan=2, padx=(0, 28))

        ctk.CTkFrame(self, fg_color=BORDER, height=1, corner_radius=0).grid(row=1, column=0, sticky="ew")

    def _build_chat(self):
        self.chat_frame = ctk.CTkScrollableFrame(
            self, fg_color=BG_MAIN, corner_radius=0,
            scrollbar_button_color="#C3CDD3", scrollbar_button_hover_color="#A9B6BE")
        self.chat_frame.grid(row=2, column=0, sticky="nsew")

    def _build_controls(self):
        panel = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0)
        panel.grid(row=3, column=0, sticky="ew")

        self.mic_btn = ctk.CTkButton(
            panel, text="🎙", font=("Segoe UI Emoji", 38), width=108, height=108, corner_radius=54,
            fg_color=C_OFF, hover_color=C_OFF, text_color="white", command=self.on_mic_toggle)
        self.mic_btn.pack(pady=(20, 8))

        self.status_label = ctk.CTkLabel(panel, text="", font=FONT_STATUS, text_color=TEXT_MAIN)
        self.status_label.pack(pady=(0, 12))

        row = ctk.CTkFrame(panel, fg_color="transparent")
        row.pack(fill="x", padx=40, pady=(0, 24))
        row.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(
            row, placeholder_text="Or type a message here...", font=("Segoe UI", 18), height=54,
            corner_radius=27, border_width=2, border_color=BORDER, fg_color="#F7FAFB",
            text_color=TEXT_MAIN, placeholder_text_color=TEXT_MUTED)
        self.entry.grid(row=0, column=0, sticky="ew")
        self.entry.bind("<Return>", lambda e: self.on_send())

        self.send_btn = ctk.CTkButton(
            row, text="Send", font=("Segoe UI", 18, "bold"), width=104, height=54, corner_radius=27,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.on_send)
        self.send_btn.grid(row=0, column=1, padx=(12, 0))

    # ---------------- thread-safe UI plumbing ----------------
    def _post(self, fn, *args):
        self.ui_queue.put((fn, args))

    def _tick(self):
        while True:
            try:
                fn, args = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn(*args)
            except Exception as e:
                print("UI error:", e)
        self._update_clock()
        if self.mode == "listening":
            self._check_listening()
        self.after(60, self._tick)

    def _update_clock(self):
        now = datetime.now()
        text = f"{now.strftime('%A')}, {now.day} {now.strftime('%B')}\n{now.strftime('%I:%M %p').lstrip('0')}"
        if text != self._last_clock:
            self._last_clock = text
            self.clock_label.configure(text=text)

    # ---------------- state ----------------
    def _set_mode(self, mode):
        self.mode = mode
        _, color, icon = MODES[mode]
        self._pulse_on = None
        self.mic_btn.configure(fg_color=color, hover_color=ACCENT_HOVER if mode == "idle" else color, text=icon)
        if mode == "idle" and not self.stt_ready:
            self.mic_btn.configure(fg_color=C_OFF, hover_color=C_OFF)
        self._refresh_status()
        st = "normal" if mode == "idle" else "disabled"
        self.entry.configure(state=st)
        self.send_btn.configure(state=st)
        if mode == "idle":
            self.after(50, self.entry.focus_set)

    def _refresh_status(self):
        text = MODES[self.mode][0]
        if self.mode == "idle" and not self.stt_ready:
            text = "Type your message below"
        self.status_label.configure(text=text)

    def _flash(self, text, ms=3500):
        self.status_label.configure(text=text)
        self.after(ms, self._refresh_status)

    def _load_models(self):
        ok = False
        if ENABLE_STT:
            try:
                load_whisper()
                ok = True
            except Exception as e:
                print("Whisper error:", e)
        self.stt_ready = ok
        self._post(self._set_mode, "idle")
        if ENABLE_STT and not ok:
            self._post(self._flash, "Voice input isn't working. You can still type to me.", 6000)

    # ---------------- chat rendering ----------------
    def _wrap(self):
        return max(280, int(self.winfo_width() * 0.62))

    def _on_resize(self, event):
        if event.widget is not self:
            return
        if self._wrap_job:
            self.after_cancel(self._wrap_job)
        self._wrap_job = self.after(150, self._apply_wrap)

    def _apply_wrap(self):
        wrap = self._wrap()
        for lbl in self.msg_labels:
            try:
                lbl.configure(wraplength=wrap)
            except Exception:
                pass

    def _add_message(self, sender, text):
        is_user = sender == "You"
        row = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        row.pack(fill="x", pady=8, padx=24)

        if not is_user:
            ctk.CTkLabel(row, text="M", font=("Segoe UI", 18, "bold"), text_color="white",
                         fg_color=ACCENT, width=44, height=44, corner_radius=22
                         ).pack(side="left", anchor="n", padx=(0, 12))

        bubble = ctk.CTkFrame(row, fg_color=USER_BUBBLE if is_user else MOMO_BUBBLE, corner_radius=22)
        bubble.pack(side="right" if is_user else "left")
        label = ctk.CTkLabel(bubble, text=text, font=FONT_BODY,
                             text_color=USER_TEXT if is_user else TEXT_MAIN,
                             wraplength=self._wrap(), justify="left", anchor="w")
        label.pack(padx=22, pady=14)
        self.msg_labels.append(label)
        self.after(30, self._scroll_bottom)
        return label

    def _scroll_bottom(self):
        try:
            self.chat_frame._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def _update_bubble(self, label, text):
        label.configure(text=text)
        self.after(30, self._scroll_bottom)

    def _clear_chat(self):
        if self.mode != "idle":
            return
        self.history = []
        self.msg_labels = []
        for w in self.chat_frame.winfo_children():
            w.destroy()
        self._greet()

    def _greet(self):
        h = datetime.now().hour
        g = "Good morning." if h < 12 else ("Good afternoon." if h < 17 else "Good evening.")
        self._add_message(COMPANION_NAME, f"{g} I'm here whenever you'd like to talk.")

    # ---------------- microphone ----------------
    def on_mic_toggle(self):
        if self.mode == "idle" and self.stt_ready:
            self._start_recording()
        elif self.mode == "listening":
            self._stop_recording()

    def _start_recording(self):
        self.recorded_chunks = []
        self.noise_blocks = []
        self.has_spoken = False
        self.voice_threshold = MIN_VOICE_RMS
        self.rec_start = self.last_voice = time.monotonic()

        def callback(indata, frames, t, status):
            self.recorded_chunks.append(indata.copy())
            rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
            if len(self.noise_blocks) < 4:          # learn room noise from first 0.4s
                self.noise_blocks.append(rms)
                if len(self.noise_blocks) == 4:
                    self.voice_threshold = max(MIN_VOICE_RMS, min(1500.0, min(self.noise_blocks) * 2.5))
            if rms > self.voice_threshold:
                self.has_spoken = True
                self.last_voice = time.monotonic()

        try:
            self.mic_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
                                             blocksize=1600, callback=callback)
            self.mic_stream.start()
        except Exception as e:
            print("Mic error:", e)
            self.mic_stream = None
            self._flash("I can't reach the microphone.")
            return
        self._set_mode("listening")

    def _check_listening(self):
        now = time.monotonic()
        elapsed = now - self.rec_start
        if self.has_spoken and now - self.last_voice > SILENCE_END_SECONDS:
            self._stop_recording()
            return
        if (not self.has_spoken and elapsed > NO_SPEECH_TIMEOUT) or elapsed > MAX_RECORD_SECONDS:
            self._stop_recording()
            return
        on = int(now * 2) % 2 == 0                  # gentle pulse while listening
        if on != self._pulse_on:
            self._pulse_on = on
            c = C_LISTEN if on else C_LISTEN_ALT
            self.mic_btn.configure(fg_color=c, hover_color=c)

    def _stop_recording(self):
        if self.mode != "listening":
            return
        if self.mic_stream:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass
            self.mic_stream = None
        if not self.recorded_chunks or not self.has_spoken:
            self._set_mode("idle")
            self._flash("I didn't hear anything. Tap the microphone and try again.")
            return
        audio = np.concatenate(self.recorded_chunks, axis=0).flatten()
        self._set_mode("transcribing")
        threading.Thread(target=self._transcribe_thread, args=(audio,), daemon=True).start()

    def _transcribe_thread(self, audio):
        try:
            text = transcribe(audio)
        except Exception as e:
            print("STT error:", e)
            text = ""
        self._post(self._on_transcribed, text)

    def _on_transcribed(self, text):
        if not text:
            self._set_mode("idle")
            self._flash("I didn't catch that. Could you say it again?")
            return
        self._send_text(text)

    # ---------------- chat ----------------
    def on_send(self):
        if self.mode != "idle":
            return
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self._send_text(text)

    def _send_text(self, text):
        self._add_message("You", text)
        self.history.append({"role": "user", "content": text})
        label = self._add_message(COMPANION_NAME, "...")
        self._set_mode("thinking")
        threading.Thread(target=self._do_reply, args=(label,), daemon=True).start()

    def _do_reply(self, label):
        messages = [dict(m) for m in self.history[-MAX_HISTORY_TURNS * 2:]]
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] = f"{build_context_block()}\n\n{messages[-1]['content']}"

        buf = []
        pending = {"flag": False}

        def flush():
            pending["flag"] = False
            self._update_bubble(label, "".join(buf) + " ▌")

        def on_chunk(piece):
            buf.append(piece)
            if not pending["flag"]:
                pending["flag"] = True
                self._post(flush)

        try:
            reply = stream_reply(messages, on_chunk).strip()
        except Exception as e:
            print("LLM error:", e)
            reply = ""

        if not reply:
            self._post(self._on_reply_failed, label)
            return

        self.history.append({"role": "assistant", "content": reply})
        self._post(self._update_bubble, label, reply)

        if ENABLE_TTS:
            self._post(self._set_mode, "speaking")
            try:
                synthesize_and_play(reply)
            except Exception as e:
                print("TTS error:", e)
        self._post(self._set_mode, "idle")

    def _on_reply_failed(self, label):
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()
        label.configure(text="I'm having trouble thinking right now. "
                             "Please check that Ollama is running, then try again.")
        self._set_mode("idle")


if __name__ == "__main__":
    app = CompanionUI()
    app.mainloop()