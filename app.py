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
# Load Gemini API Key
# ---------------------------
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    # sanitize accidental surrounding quotes
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
st.session_state.setdefault("voice_audio_bytes", None)
st.session_state.setdefault("spinner_counter", 0)
st.session_state.setdefault("play_audio", False)
st.session_state.setdefault("audio_bytes", None)
st.session_state.setdefault("tts_thread", None)
st.session_state.setdefault("last_answer", "")
st.session_state.setdefault("status_message", "")
st.session_state.setdefault("last_error_trace", "")

# ---------------------------
# Voice Recognition (browser-first)
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


def listen_to_voice():
    """Browser recorder first (works on Streamlit Cloud). Falls back to server mic only for local dev.
    Stores WAV bytes in st.session_state['voice_audio_bytes'] when recording completes.
    """
    # Try browser recorder component
        # Try browser recorder component (support multiple package names / forks)
    audio_recorder = None
    _rec_import_ok = None

    # 1) audio-recorder-streamlit -> module: audio_recorder_streamlit, function: audio_recorder
    try:
        from audio_recorder_streamlit import audio_recorder  # pip name: audio-recorder-streamlit
        _rec_import_ok = "audio_recorder_streamlit"
    except Exception:
        # 2) streamlit-audiorec -> module: st_audiorec, function: st_audiorec
        try:
            from st_audiorec import st_audiorec  # pip name: streamlit-audiorec
            def audio_recorder():
                return st_audiorec()
            _rec_import_ok = "st_audiorec (streamlit-audiorec)"
        except Exception:
            # 3) streamlit_audio_recorder (other forks)
            try:
                from streamlit_audio_recorder import audio_recorder
                _rec_import_ok = "streamlit_audio_recorder"
            except Exception:
                # 4) streamlit_audiorecorder or other common names
                try:
                    from streamlit_audiorecorder import audio_recorder
                    _rec_import_ok = "streamlit_audiorecorder"
                except Exception:
                    audio_recorder = None
                    _rec_import_ok = None

    # Small debug print so you can see in logs what succeeded (deployment logs / terminal)
    try:
        if _rec_import_ok:
            print(f"[DEBUG] Recorder import succeeded: {_rec_import_ok}")
        else:
            print("[DEBUG] No browser recorder import succeeded; audio_recorder is None")
    except Exception:
        pass


    if audio_recorder is not None:
        st.markdown("<div class='center-msg'>🎤 Click the record button, speak, then click Stop. Then click 'Submit' to transcribe.</div>", unsafe_allow_html=True)
        rec = audio_recorder()  # returns WAV bytes or None
        if rec is None:
            return
        st.session_state["voice_audio_bytes"] = rec
        st.success("✅ Recording captured. Click Submit to transcribe and run the query.")
        return

    # Fallback to server-side mic (only works locally if pyaudio is installed)
    if not can_use_microphone():
        st.error("Microphone not available or voice features are disabled. For cloud deployments use the browser recorder (enable streamlit-audio-recorder in requirements).")
        return

    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.markdown("<div class='center-msg'>🎤 Listening... Speak now! (will stop after 8s)</div>", unsafe_allow_html=True)
            audio = r.listen(source, phrase_time_limit=8)
    except Exception as e:
        st.error(f"Microphone error: {e}")
        return

    try:
        text = r.recognize_google(audio)
        st.session_state["voice_text"] = text
        st.session_state["query_text"] = text
        st.markdown("<div class='center-msg'>✅ Voice captured. Text was placed into the question box.</div>", unsafe_allow_html=True)
    except sr.UnknownValueError:
        st.error("❌ Could not understand audio.")
    except sr.RequestError:
        st.error("❌ Speech recognition service unavailable.")


def transcribe_recording_and_set_text():
    """Transcribe WAV bytes from st.session_state['voice_audio_bytes'] using SpeechRecognition.
    Sets voice_text and query_text on success. Returns True if transcription succeeded.
    """
    wav_bytes = st.session_state.get("voice_audio_bytes")
    if not wav_bytes:
        return False

    r = sr.Recognizer()
    try:
        audio_file = io.BytesIO(wav_bytes)
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
        st.error(f"Speech service error: {e}")
    except Exception as e:
        st.error(f"Transcription failed: {e}")
    return False

# ---------------------------
# Text-to-Speech (gTTS + in-memory MP3)
# ---------------------------

def generate_audio_bytes(text: str, lang: str = "en") -> bytes:
    """Generate MP3 bytes using gTTS. Requires internet access."""
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
    # If user opted to use local TTS and pyttsx3 is available, speak locally (faster on local machines)
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
            threading.Thread(target=_speak_local, args=(text[:TTS_CHARS_LIMIT],), daemon=True).start()
            st.session_state["status_message"] = "Playing (local TTS)..."
            return
        except Exception as e:
            st.session_state["last_error_trace"] = traceback.format_exc()
            print("pyttsx3 not available, falling back to gTTS", e)

    # Otherwise use gTTS -> in-memory mp3. Limit characters to speed up.
    st.session_state["status_message"] = "Generating audio..."
    audio_bytes = generate_audio_bytes(text[:TTS_CHARS_LIMIT])
    if audio_bytes:
        st.session_state["audio_bytes"] = audio_bytes
        st.session_state["play_audio"] = True
        st.session_state["status_message"] = "Playing..."


def stop_callback():
    # Stop playback by clearing audio bytes and stopping render (removes embedded player)
    st.session_state["play_audio"] = False
    st.session_state["audio_bytes"] = None
    st.session_state["status_message"] = "Stopped."


def clear_callback():
    st.session_state["query_text"] = ""
    st.session_state["voice_text"] = ""
    st.session_state["voice_audio_bytes"] = None
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
# Submit Query (kept separate so it can be used in callbacks)
# ---------------------------

def submit_query_internal():
    # If user recorded audio via browser, transcribe it first so voice_text/query_text are set
    if st.session_state.get("voice_audio_bytes"):
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

            # Auto-read if user enabled it: generate/play audio for a shortened chunk to speed up
            try:
                if AUTO_READ:
                    # If local TTS is preferred and available, use it (faster locally)
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
                            threading.Thread(target=_speak_local_auto, args=(answer[:TTS_CHARS_LIMIT],), daemon=True).start()
                            st.session_state["status_message"] = "Auto-reading (local TTS)..."
                        except Exception:
                            st.session_state["last_error_trace"] = traceback.format_exc()
                            print("pyttsx3 not available for auto-read, falling back to gTTS")
                            audio_bytes = generate_audio_bytes(answer[:TTS_CHARS_LIMIT])
                            if audio_bytes:
                                st.session_state["audio_bytes"] = audio_bytes
                                st.session_state["play_audio"] = True
                                st.session_state["status_message"] = "Auto-playing..."
                    else:
                        audio_bytes = generate_audio_bytes(answer[:TTS_CHARS_LIMIT])
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
# Show answer and retrieved chunks
# ---------------------------

def show_answer_and_chunks():
    # show status message
    status = st.session_state.get("status_message", "")
    if status:
        st.markdown(f"<div style='text-align:center; font-weight:bold'>{status}</div>", unsafe_allow_html=True)

    # show last error trace if present (collapsible)
    last_err = st.session_state.get("last_error_trace", "")
    if last_err:
        with st.expander("⚠️ Last error trace (click to expand)", expanded=False):
            st.code(last_err)

    # Debug area hidden in an expander so user can toggle visibility
    with st.expander("🔧 Debug Output (click to show)", expanded=False):
        st.markdown("**Status message:**")
        st.write(st.session_state.get("status_message", ""))
        last_err = st.session_state.get("last_error_trace", "")
        if last_err:
            st.markdown("**Last error trace:**")
            st.code(last_err)
        # show raw last_answer type and content for troubleshooting
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

        # Audio playback area: will appear only when play_audio is True
        if st.session_state.get("play_audio") and st.session_state.get("audio_bytes"):
            try:
                # Render an HTML5 audio player via components.html to reduce download UI.
                # controlsList="nodownload" and oncontextmenu="return false" hide the browser's usual download buttons.
                b64 = base64.b64encode(st.session_state.get("audio_bytes")).decode()
                audio_html = f"""
                <div style="text-align:center">
                  <audio id="player" autoplay controls controlsList="nodownload" oncontextmenu="return false" style="width:100%; max-width:720px;">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    Your browser does not support the audio element.
                  </audio>
                </div>
                <script>
                  // try to auto-play (may be blocked by browser policies)
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

        # Show retrieved chunks
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
    # Always call browser/server listen function; it handles which method is available
    st.button("🎤 Speak", on_click=listen_to_voice)
    st.button("📨 Submit", on_click=submit_query)

# After the main controls, show answer (so it persists across reruns)
show_answer_and_chunks()

# ---------------------------
# Audio Control Buttons (use callbacks to modify state inside callbacks)
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

    if st.session_state.get("last_error_trace"):
        st.write("Last error trace (short):")
        st.code(st.session_state.get("last_error_trace")[:10000])  # truncate in UI
    st.write("Check the terminal where `streamlit run app.py` is running for the full stack trace and more info.")

# End of file
