# app.py
# AI Legal Assistant (Streamlit) with voice recording/transcription
# Designed for requirements:
#   streamlit, pdfplumber, faiss-cpu, sentence-transformers,
#   google-generativeai, numpy, requests, gTTS, SpeechRecognition,
#   streamlit-lottie, pyttsx3 (Windows), audio-recorder-streamlit,
#   soundfile, soundfile>=0.12.1, SpeechRecognition>=3.8.1
#
# If you later add pydub + ffmpeg, audio conversion becomes more robust for webm/mp3.

import streamlit as st
import os
import main
import uuid
import threading
import requests
from gtts import gTTS
import io
import speech_recognition as sr
import time
import traceback
import base64
import streamlit.components.v1 as components
from contextlib import contextmanager
import tempfile
import re
import soundfile as sf
import numpy as np

# optional lottie import
try:
    from streamlit_lottie import st_lottie
    LOTTIE = True
except Exception:
    LOTTIE = False

# ---------------------------
# Page Config & Styling
# ---------------------------
st.set_page_config(page_title="🤖 AI Legal Assistant", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1620937125831-7c1e70e2eede?auto=format&fit=crop&w=1950&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }
    .stTextInput>div>input, .stTextArea>div>textarea {
        color: black;
    }
    .disclaimer { color: red; font-weight: bold; }
    .center-msg { text-align: center; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Indian Law RAG Assistant: Gemini-Powered Legal Query System")
st.markdown("This tool provides **informational summaries** based on uploaded laws.")
st.markdown('<p class="disclaimer">❗ Disclaimer: This is informational only. Not legal advice.</p>', unsafe_allow_html=True)

# ---------------------------
# Load Gemini API Key (optional)
# ---------------------------
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    gemini_key = gemini_key.strip().strip('"').strip("'")
    os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ["GOOGLE_API_KEY"] = gemini_key
    try:
        if hasattr(main, "set_api_key"):
            main.set_api_key(gemini_key)
        else:
            st.warning("main.set_api_key not found; continuing without setting key from env")
    except Exception as e:
        st.warning(f"Could not set Gemini key: {e}")
else:
    st.warning("No GEMINI_API_KEY found. Gemini answers will be disabled.")

# ---------------------------
# Sidebar Settings
# ---------------------------
st.sidebar.header("Settings")
K = st.sidebar.slider("Top K chunks to retrieve", 1, 10, 5)
USE_GEMINI = st.sidebar.checkbox("Use Gemini (requires API key)", True if gemini_key else False)
AUTO_READ = st.sidebar.checkbox("Auto-read answer after submission", value=False)
ENABLE_VOICE = st.sidebar.checkbox("Enable voice features (experimental)", value=False)
# TTS options
st.sidebar.markdown("---")
TTS_CHARS_LIMIT = st.sidebar.slider("Max characters to TTS (helps speed)", 200, 4000, 1200)
USE_LOCAL_TTS = st.sidebar.checkbox("Use local TTS (pyttsx3) if available", value=False)
READ_FULL_TTS = st.sidebar.checkbox("Read full TTS (may be slow for long answers)", value=False)

# ---------------------------
# Lottie Loader
# ---------------------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

LOTTIE_URL = "https://assets6.lottiefiles.com/packages/lf20_j1adxtyb.json"
lottie_json = load_lottie_url(LOTTIE_URL) if LOTTIE else None

# ---------------------------
# Session State Initialization
# ---------------------------
st.session_state.setdefault("query_text", "")
st.session_state.setdefault("voice_text", "")
st.session_state.setdefault("voice_audio_bytes", None)  # can hold raw bytes or raw recorder payload
st.session_state.setdefault("last_rec_payload", None)   # raw recorder payload for debugging
st.session_state.setdefault("spinner_counter", 0)
st.session_state.setdefault("play_audio", False)
st.session_state.setdefault("audio_bytes", None)
st.session_state.setdefault("tts_thread", None)
st.session_state.setdefault("last_answer", "")
st.session_state.setdefault("status_message", "")
st.session_state.setdefault("last_error_trace", "")

# ---------------------------
# Voice helpers (no pydub) — use soundfile where possible
# ---------------------------

def can_use_microphone() -> bool:
    """Server-side microphone availability check (only relevant for local dev)."""
    if not ENABLE_VOICE:
        return False
    try:
        with sr.Microphone() as source:
            return True
    except Exception:
        return False

def _debug_log(msg, print_console=True):
    try:
        st.session_state.setdefault("status_message", "")
        st.session_state["status_message"] = msg
    except Exception:
        pass
    if print_console:
        print("[VOICE DEBUG]", msg)

def _try_base64_decode(s: str):
    s = s.strip()
    m = re.match(r"data:audio/([^;]+);base64,(.*)$", s, flags=re.I | re.S)
    if m:
        mime = "audio/" + m.group(1).lower()
        b64_payload = m.group(2)
        try:
            return base64.b64decode(b64_payload), mime
        except Exception:
            return None, mime
    try:
        return base64.b64decode(s), None
    except Exception:
        return None, None

def _extract_bytes_from_rec(rec):
    """Try to find raw bytes and optional mime from recorder return."""
    if isinstance(rec, (bytes, bytearray)):
        return bytes(rec), None
    if isinstance(rec, str):
        decoded, mime = _try_base64_decode(rec)
        if decoded:
            return decoded, mime
        return None, None
    if isinstance(rec, dict):
        for k in ("audio", "data", "wav", "blob", "audio_base64", "base64", "file", "recording"):
            if k in rec and rec[k]:
                val = rec[k]
                if isinstance(val, (bytes, bytearray)):
                    return bytes(val), rec.get("mimeType") or rec.get("type") or None
                if isinstance(val, str):
                    decoded, mime = _try_base64_decode(val)
                    if decoded:
                        return decoded, mime
                if isinstance(val, dict) and "content" in val:
                    c = val["content"]
                    if isinstance(c, (bytes, bytearray)):
                        return bytes(c), val.get("mimeType") or None
                    if isinstance(c, str):
                        decoded, mime = _try_base64_decode(c)
                        if decoded:
                            return decoded, mime
        # deep search for long base64 strings
        def _search_for_b64(v):
            if isinstance(v, dict):
                for vv in v.values():
                    out = _search_for_b64(vv)
                    if out:
                        return out
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    out = _search_for_b64(vv)
                    if out:
                        return out
            elif isinstance(v, str):
                if len(v) > 100:
                    decoded, mime = _try_base64_decode(v)
                    if decoded:
                        return decoded, mime
            return None
        res = _search_for_b64(rec)
        if res:
            return res
        return None, None
    if isinstance(rec, (list, tuple)):
        for item in rec:
            if isinstance(item, (bytes, bytearray)):
                return bytes(item), None
            if isinstance(item, str):
                decoded, mime = _try_base64_decode(item)
                if decoded:
                    return decoded, mime
            if isinstance(item, dict):
                out = _extract_bytes_from_rec(item)
                if out and out[0]:
                    return out
        return None, None
    return None, None

def _convert_to_wav_bytes_using_soundfile(raw_bytes: bytes):
    """Try to read the bytes with soundfile and re-export as WAV bytes.
       soundfile supports WAV/FLAC/OGG (depending on libsndfile build).
       MP3 and WebM/Opus may not be supported — in that case this will fail."""
    try:
        bio = io.BytesIO(raw_bytes)
        # soundfile can accept file-like objects in recent versions
        with sf.SoundFile(bio) as sf_file:
            data = sf_file.read(dtype='float32')
            samplerate = sf_file.samplerate
            # write WAV to buffer
            out = io.BytesIO()
            # soundfile.write expects array-like, samplerate, and file info
            sf.write(out, data, samplerate, format="WAV", subtype="PCM_16")
            out.seek(0)
            return out.read()
    except Exception as e:
        # might be unsupported format (mp3/webm), return None so caller can try fallbacks
        _debug_log(f"soundfile conversion failed: {e}", print_console=False)
        return None

def transcribe_recording_and_set_text():
    """Transcribe recorder payload stored in session (either voice_audio_bytes or last_rec_payload)."""
    raw = st.session_state.get("voice_audio_bytes")
    if not raw:
        raw = st.session_state.get("last_rec_payload")
    if not raw:
        st.error("No recording found to transcribe.")
        return False

    _debug_log("Attempting to extract audio bytes from recorder return...")
    raw_bytes, mime = _extract_bytes_from_rec(raw)
    if raw_bytes is None and isinstance(raw, (bytes, bytearray)):
        raw_bytes = bytes(raw)

    if raw_bytes is None:
        st.error("Could not extract audio bytes from the recorder return. See diagnostics for payload preview.")
        try:
            st.write("Recorder preview:")
            if isinstance(raw, dict):
                st.json(raw)
            else:
                st.write(str(raw)[:2000])
        except Exception:
            pass
        return False

    _debug_log(f"Extracted bytes length={len(raw_bytes)} mime={mime}")

    # Try soundfile conversion -> wav bytes
    wav_bytes = _convert_to_wav_bytes_using_soundfile(raw_bytes)
    if not wav_bytes:
        # fallback: try writing raw bytes to temp with common suffixes and let SpeechRecognition try
        try:
            for ext in (".wav", ".ogg", ".flac", ".mp3", ".webm"):
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                    tmp.write(raw_bytes)
                    tmp.flush()
                    tmp.close()
                    r = sr.Recognizer()
                    with sr.AudioFile(tmp.name) as source:
                        audio_data = r.record(source)
                    text = r.recognize_google(audio_data)
                    st.session_state["voice_text"] = text
                    st.session_state["query_text"] = text
                    st.success(f"Transcription: {text}")
                    return True
                except Exception:
                    continue
        except Exception:
            st.session_state["last_error_trace"] = traceback.format_exc()
            st.error("Fallback transcription attempts failed (see trace).")
            return False

    # Transcribe using speech_recognition from in-memory WAV bytes
    try:
        audio_file = io.BytesIO(wav_bytes)
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        st.session_state["voice_text"] = text
        st.session_state["query_text"] = text
        st.success(f"Transcription: {text}")
        return True
    except sr.UnknownValueError:
        st.error("Could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Speech recognition service error: {e}")
    except Exception as e:
        st.session_state["last_error_trace"] = traceback.format_exc()
        st.error(f"Transcription failed: {e}")
    return False

# ---------------------------
# Recorder UI + extraction wrapper
# ---------------------------
def listen_to_voice():
    """Show browser recorder (if installed) and attempt transcription immediately."""
    audio_recorder = None
    _rec_import_ok = None

    # try several recorder packages (you have audio-recorder-streamlit in your reqs)
    try:
        from audio_recorder_streamlit import audio_recorder
        _rec_import_ok = "audio_recorder_streamlit"
    except Exception:
        try:
            from st_audiorec import st_audiorec
            def audio_recorder():
                return st_audiorec()
            _rec_import_ok = "st_audiorec (streamlit-audiorec)"
        except Exception:
            try:
                from streamlit_audio_recorder import audio_recorder
                _rec_import_ok = "streamlit_audio_recorder"
            except Exception:
                try:
                    from streamlit_audiorecorder import audio_recorder
                    _rec_import_ok = "streamlit_audiorecorder"
                except Exception:
                    audio_recorder = None
                    _rec_import_ok = None

    _debug_log(f"Recorder import: {_rec_import_ok}")

    if audio_recorder is None:
        st.error("No supported audio recorder component found. Install one (e.g. audio-recorder-streamlit) and add to requirements.")
        return

    st.markdown("<div class='center-msg'>🎤 Click the record button, speak, then click Stop. Then click 'Submit' to transcribe.</div>", unsafe_allow_html=True)
    rec = audio_recorder()
    # keep raw payload for debugging
    st.session_state["last_rec_payload"] = rec

    # show quick preview of what recorder returned
    try:
        if isinstance(rec, (bytes, bytearray)):
            st.write(f"Recorder returned bytes (len={len(rec)})")
        elif isinstance(rec, str):
            st.write(f"Recorder returned string (len={len(rec)}): {rec[:160]}...")
        elif isinstance(rec, dict):
            st.write("Recorder returned dict keys: " + ", ".join(list(rec.keys())))
            try:
                st.json({k: (str(v)[:300] + "..." if isinstance(v, (str, bytes)) and len(str(v))>300 else v) for k,v in list(rec.items())[:8]})
            except Exception:
                pass
        else:
            st.write("Recorder returned object of type:", type(rec))
    except Exception:
        pass

    # Try to extract bytes & transcribe immediately
    ok = transcribe_recording_and_set_text()
    if not ok:
        st.warning("Recording captured but transcription failed. Check Diagnostics / Notes for last_error_trace.")
    else:
        # optionally auto-submit
        AUTO_SUBMIT_AFTER_TRANSCRIBE = False
        if AUTO_SUBMIT_AFTER_TRANSCRIBE:
            submit_query_internal()

# ---------------------------
# Text-to-Speech (gTTS)
# ---------------------------
def generate_audio_bytes(text: str, lang: str = "en") -> bytes:
    try:
        tts = gTTS(text=text, lang=lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.session_state["last_error_trace"] = traceback.format_exc()
        st.error(f"gTTS generation failed: {e}")
        return None

def read_aloud_callback():
    text = st.session_state.get("last_answer", "")
    if not text:
        st.warning("No answer to read. Submit a query first.")
        return
    text = str(text)
    tts_text = text if READ_FULL_TTS else text[:TTS_CHARS_LIMIT]

    if USE_LOCAL_TTS:
        try:
            import pyttsx3
            def _speak_local(tt):
                try:
                    engine = pyttsx3.init()
                    engine.say(tt)
                    engine.runAndWait()
                except Exception as e:
                    st.session_state["last_error_trace"] = traceback.format_exc()
                    print("Local TTS error:", e)
            threading.Thread(target=_speak_local, args=(tts_text,), daemon=True).start()
            st.session_state["status_message"] = "Playing (local TTS)..."
            return
        except Exception as e:
            st.session_state["last_error_trace"] = traceback.format_exc()
            print("pyttsx3 not available, falling back to gTTS", e)

    st.session_state["status_message"] = "Generating audio..."
    audio_bytes = generate_audio_bytes(tts_text)
    if audio_bytes:
        st.session_state["audio_bytes"] = audio_bytes
        st.session_state["play_audio"] = True
        st.session_state["status_message"] = "Playing..."

def stop_callback():
    st.session_state["play_audio"] = False
    st.session_state["audio_bytes"] = None
    st.session_state["status_message"] = "Stopped."

def clear_callback():
    st.session_state["query_text"] = ""
    st.session_state["voice_text"] = ""
    st.session_state["voice_audio_bytes"] = None
    st.session_state["last_rec_payload"] = None
    st.session_state["last_answer"] = ""
    st.session_state["play_audio"] = False
    st.session_state["audio_bytes"] = None
    st.session_state["status_message"] = ""
    st.session_state["last_error_trace"] = ""

# ---------------------------
# Spinner with Unique Key (context manager)
# ---------------------------
@contextmanager
def show_spinner_placeholder():
    placeholder = st.empty()
    st.session_state["spinner_counter"] += 1
    key = f"spinner-{st.session_state['spinner_counter']}-{uuid.uuid4().hex}"
    try:
        if LOTTIE and lottie_json:
            with placeholder.container():
                try:
                    st_lottie(lottie_json, height=140, key=key)
                    yield placeholder
                except Exception:
                    with st.spinner("Searching...", key=key):
                        yield placeholder
        else:
            with placeholder.container():
                with st.spinner("Searching...", key=key):
                    yield placeholder
    finally:
        try:
            placeholder.empty()
        except Exception:
            pass

# ---------------------------
# Submit Query
# ---------------------------

def submit_query_internal():
    # If user recorded audio via browser, transcribe it first so voice_text/query_text are set
    if st.session_state.get("voice_audio_bytes") or st.session_state.get("last_rec_payload"):
        try:
            transcribe_recording_and_set_text()
        except Exception:
            st.session_state["last_error_trace"] = traceback.format_exc()
            print("Transcription error:", st.session_state["last_error_trace"])

    typed = st.session_state.get("query_text", "").strip()
    voice = st.session_state.get("voice_text", "").strip()
    query = typed or voice
    if not query:
        st.session_state["status_message"] = "Please type a question or use the Speak button first."
        return

    st.session_state["status_message"] = "DEBUG: calling main.answer_query(...) — starting"
    st.session_state["last_error_trace"] = ""
    print("DEBUG: submit_query_internal - starting. query:", repr(query))

    t0 = time.time()
    try:
        with show_spinner_placeholder():
            try:
                answer = main.answer_query(query, k=K, use_gemini=USE_GEMINI)
            except Exception:
                st.session_state["last_answer"] = ""
                st.session_state["status_message"] = "DEBUG: main.answer_query raised exception (see terminal/UI trace)."
                st.session_state["last_error_trace"] = traceback.format_exc()
                print("DEBUG: main.answer_query EXCEPTION:", st.session_state["last_error_trace"])
                return

        t1 = time.time()
        dur = t1 - t0
        if answer is None or (isinstance(answer, (str, list, dict)) and len(answer) == 0):
            st.session_state["last_answer"] = ""
            st.session_state["status_message"] = f"DEBUG: main.answer_query returned empty/None (runtime {dur:.2f}s)."
            print(f"DEBUG: main.answer_query returned empty/None in {dur:.2f}s. type={type(answer)}")
        else:
            st.session_state["last_answer"] = answer
            sample = (answer[:300] + "...") if isinstance(answer, str) and len(answer) > 300 else str(answer)
            st.session_state["status_message"] = f"DEBUG: Answer retrieved in {dur:.2f}s. See debug area."
            print(f"DEBUG: main.answer_query finished in {dur:.2f}s. type={type(answer)}, sample={sample!r}")

            # Auto-read if user enabled it
            try:
                if AUTO_READ:
                    tts_text_auto = str(answer) if READ_FULL_TTS else str(answer)[:TTS_CHARS_LIMIT]
                    if USE_LOCAL_TTS:
                        try:
                            import pyttsx3
                            def _speak_local_auto(tt):
                                try:
                                    engine = pyttsx3.init()
                                    engine.say(tt)
                                    engine.runAndWait()
                                except Exception as e:
                                    st.session_state["last_error_trace"] = traceback.format_exc()
                                    print("Local TTS error during auto-read:", e)
                            threading.Thread(target=_speak_local_auto, args=(tts_text_auto,), daemon=True).start()
                            st.session_state["status_message"] = "Auto-reading (local TTS)..."
                        except Exception:
                            st.session_state["last_error_trace"] = traceback.format_exc()
                            print("pyttsx3 not available for auto-read, falling back to gTTS")
                            audio_bytes = generate_audio_bytes(tts_text_auto)
                            if audio_bytes:
                                st.session_state["audio_bytes"] = audio_bytes
                                st.session_state["play_audio"] = True
                                st.session_state["status_message"] = "Auto-playing..."
                    else:
                        audio_bytes = generate_audio_bytes(tts_text_auto)
                        if audio_bytes:
                            st.session_state["audio_bytes"] = audio_bytes
                            st.session_state["play_audio"] = True
                            st.session_state["status_message"] = "Auto-playing..."
            except Exception:
                st.session_state["last_error_trace"] = traceback.format_exc()
                print("Auto-read failed:", st.session_state["last_error_trace"])
    except Exception:
        st.session_state["last_answer"] = ""
        st.session_state["last_error_trace"] = traceback.format_exc()
        st.session_state["status_message"] = "DEBUG: Unexpected exception in submit flow (see trace)."
        print("DEBUG: submit_query_internal UNEXPECTED exception:", st.session_state["last_error_trace"])
        return

def submit_query():
    submit_query_internal()

# ---------------------------
# Show answer and chunks
# ---------------------------

def show_answer_and_chunks():
    status = st.session_state.get("status_message", "")
    if status:
        st.markdown(f"<div style='text-align:center; font-weight:bold'>{status}</div>", unsafe_allow_html=True)

    last_err = st.session_state.get("last_error_trace", "")
    if last_err:
        with st.expander("⚠️ Last error trace (click to expand)", expanded=False):
            st.code(last_err)

    with st.expander("🔧 Debug Output (click to show)", expanded=False):
        st.markdown("**Status message:**")
        st.write(st.session_state.get("status_message", ""))
        last_err = st.session_state.get("last_error_trace", "")
        if last_err:
            st.markdown("**Last error trace:**")
            st.code(last_err)
        ans = st.session_state.get("last_answer")
        if ans is not None and ans != "":
            st.markdown("**DEBUG: last_answer type & length**")
            try:
                st.write("type:", type(ans))
                st.write("length:", len(ans))
            except Exception:
                st.write("Cannot compute length for this type")
            st.text_area("DEBUG: Last answer (raw)", ans, height=220)

    if st.session_state.get("last_answer"):
        with st.container():
            with st.expander("📜 Detailed Answer", expanded=True):
                ans = st.session_state.get("last_answer")
                if isinstance(ans, str):
                    st.markdown(ans)
                else:
                    st.write(ans)
                st.markdown(
                    "<p style='color:red; font-weight:bold;'>⚠️ Disclaimer: This is for informational purposes only — not legal advice.</p>",
                    unsafe_allow_html=True,
                )

        if st.session_state.get("play_audio") and st.session_state.get("audio_bytes"):
            try:
                b64 = base64.b64encode(st.session_state.get("audio_bytes")).decode()
                audio_html = f"""
                <div style="text-align:center">
                  <audio id="player" autoplay controls controlsList="nodownload" oncontextmenu="return false" style="width:100%; max-width:720px;">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    Your browser does not support the audio element.
                  </audio>
                </div>
                <script>
                  const player = document.getElementById('player');
                  if (player) {{
                    player.play().catch(e => console.log('autoplay prevented', e));
                  }}
                </script>
                """
                components.html(audio_html, height=140)
            except Exception as e:
                st.error(f"Audio render error: {e}")
                st.session_state["play_audio"] = False
                st.session_state["audio_bytes"] = None

        st.subheader("🔍 Retrieved Chunks (for transparency)")
        try:
            docs = []
            try:
                docs = main.search(st.session_state.get("query_text", ""), k=K)
            except Exception:
                st.session_state["last_error_trace"] = traceback.format_exc()
                st.error("Error retrieving chunks; see last error trace.")
                docs = []

            if not docs:
                st.info("No chunks retrieved (index may be empty).")
            for i, doc in enumerate(docs):
                if isinstance(doc, tuple) and len(doc) >= 2:
                    meta, dist = doc[0], doc[1]
                else:
                    meta, dist = ({"source": "-", "text": str(doc)}, 0.0)
                source = meta.get("source", meta.get("id", "-")) if isinstance(meta, dict) else "-"
                text = meta.get("text", str(meta)) if isinstance(meta, dict) else str(meta)
                with st.expander(f"Source {i+1}: {source} (distance {dist:.4f})"):
                    st.write(text[:2000] + ("..." if len(text) > 2000 else ""))
        except Exception:
            st.session_state["last_error_trace"] = traceback.format_exc()
            st.error("An unexpected error occurred while showing chunks; see last error trace.")

# ---------------------------
# UI Controls
# ---------------------------
col1, col2 = st.columns([4, 1])
with col1:
    st.text_input("Ask a legal question about India:", key="query_text")

with col2:
    st.button("🎤 Speak", on_click=listen_to_voice)
    st.button("📨 Submit", on_click=submit_query)

# After the main controls, show answer (so it persists across reruns)
show_answer_and_chunks()

# ---------------------------
# Audio Control Buttons
# ---------------------------
if st.session_state.get("last_answer"):
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button("▶ Read Aloud", on_click=read_aloud_callback)
    with col2:
        st.button("⏹ Stop", on_click=stop_callback)
    with col3:
        st.button("🗑 Clear", on_click=clear_callback)

# ---------------------------
# Diagnostics / Notes
# ---------------------------
with st.expander("⚙️ Diagnostics / Notes", expanded=False):
    st.write("Voice enabled:", ENABLE_VOICE)
    st.write("Microphone available:", can_use_microphone() if ENABLE_VOICE else "N/A")
    st.write("Gemini key present:", bool(gemini_key))
    st.write("Use Gemini:", USE_GEMINI)
    st.write("Status message:", st.session_state.get("status_message", ""))
    st.write("soundfile available:", True)
    if st.session_state.get("last_rec_payload") is not None:
        st.write("Last recorder payload preview:")
        try:
            rec = st.session_state.get("last_rec_payload")
            if isinstance(rec, dict):
                st.json({k: (str(v)[:300] + "..." if isinstance(v, (str, bytes)) and len(str(v))>300 else v) for k,v in list(rec.items())[:12]})
            else:
                st.write(str(rec)[:1000])
        except Exception:
            st.write("Could not preview recorder payload.")
    if st.session_state.get("last_error_trace"):
        st.write("Last error trace (short):")
        st.code(st.session_state.get("last_error_trace")[:10000])
    st.write("If audio format is webm/mp3/opus and transcription fails, install pydub and ensure ffmpeg is on PATH for robust conversion.")
    st.write("Check the terminal where `streamlit run app.py` is running for the full stack trace and more info.")
