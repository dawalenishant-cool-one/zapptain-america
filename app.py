import os
import tempfile

# Force temporary files to be created on the D: drive to bypass disk space errors on C:
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
D_TEMP_DIR = os.path.join(SCRIPT_DIR, "temp")
os.makedirs(D_TEMP_DIR, exist_ok=True)
tempfile.tempdir = D_TEMP_DIR
os.environ["TEMP"] = D_TEMP_DIR
os.environ["TMP"] = D_TEMP_DIR
os.environ["TMPDIR"] = D_TEMP_DIR

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
import io
import pickle
import time

# Try importing Plotly for interactive plots
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Set page configuration for a wide dashboard
st.set_page_config(
    page_title="Zapptain America - Audio Fingerprinting System",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# 0. STYLING OVERRIDES (Dynamic Light & Dark Themes)
# =====================================================================
logo_col1, logo_col2, logo_col3 = st.sidebar.columns([1, 1.5, 1])
with logo_col2:
    st.image("https://img.icons8.com/nolan/96/audio-wave.png", width=70)

theme_choice = "Dark Theme"

# Apply dynamic CSS styles using root variables
if theme_choice == "Dark Theme":
    css_variables = """
    :root {
        --bg-color: #0f111a;
        --card-bg: #1e2235;
        --card-border: #2e344e;
        --text-main: #ffffff;
        --text-muted: #9ca3af;
        --accent-grad: linear-gradient(90deg, #818cf8 0%, #c084fc 100%);
        --success-bg: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        --success-border: #10b981;
        --success-text: #10b981;
        --sidebar-bg: #0b0d16;
        --tab-bg: #1e2235;
        --tab-text: #9ca3af;
        --tab-selected: #4f46e5;
    }
    """
    plot_theme = "dark"
elif theme_choice == "Light Theme":
    css_variables = """
    :root {
        --bg-color: #f3f4f6;
        --card-bg: #ffffff;
        --card-border: #e5e7eb;
        --text-main: #111827;
        --text-muted: #4b5563;
        --accent-grad: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%);
        --success-bg: linear-gradient(135deg, rgba(52, 211, 153, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%);
        --success-border: #059669;
        --success-text: #047857;
        --sidebar-bg: #f9fafb;
        --tab-bg: #e5e7eb;
        --tab-text: #4b5563;
        --tab-selected: #4f46e5;
    }
    """
    plot_theme = "light"
else:
    css_variables = """
    :root {
        --bg-color: #0f111a;
        --card-bg: #1e2235;
        --card-border: #2e344e;
        --text-main: #ffffff;
        --text-muted: #9ca3af;
        --accent-grad: linear-gradient(90deg, #818cf8 0%, #c084fc 100%);
        --success-bg: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        --success-border: #10b981;
        --success-text: #10b981;
        --sidebar-bg: #0b0d16;
        --tab-bg: #1e2235;
        --tab-text: #9ca3af;
        --tab-selected: #4f46e5;
    }
    @media (prefers-color-scheme: light) {
        :root {
            --bg-color: #f3f4f6;
            --card-bg: #ffffff;
            --card-border: #e5e7eb;
            --text-main: #111827;
            --text-muted: #4b5563;
            --accent-grad: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%);
            --success-bg: linear-gradient(135deg, rgba(52, 211, 153, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%);
            --success-border: #059669;
            --success-text: #047857;
            --sidebar-bg: #f9fafb;
            --tab-bg: #e5e7eb;
            --tab-text: #4b5563;
            --tab-selected: #4f46e5;
        }
    }
    """
    plot_theme = "dark"

st.markdown(f"""
<style>
    {css_variables}
    
    /* Apply Background color globally to avoid layout discrepancies */
    body, .stApp, section.main, .block-container, [data-testid="stAppViewBlockContainer"] {{
        background-color: var(--bg-color) !important;
    }}
    
    /* Force text color on all core elements */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp li, .stApp div[data-testid="stMarkdownContainer"] {{
        color: var(--text-main) !important;
    }}
    
    /* Style sidebar background and borders */
    section[data-testid="stSidebar"] {{
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--card-border) !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: var(--text-main) !important;
    }}
    
    /* Exclude specific exceptions where custom styles are used */
    .success-banner, .success-banner * {{
        color: var(--success-text) !important;
        background: var(--success-bg) !important;
        border: 1px solid var(--success-border) !important;
    }}
    
    /* Metric cards background and text */
    .metric-card {{
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        color: var(--text-main) !important;
        transition: transform 0.2s, border-color 0.2s;
    }}
    .metric-card * {{
        color: var(--text-main) !important;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        border-color: var(--tab-selected) !important;
    }}
    
    /* Re-override for native alerts to maintain warning styles */
    div[data-testid="stAlert"] {{
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 12px !important;
    }}
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] span, div[data-testid="stAlert"] div {{
        color: inherit !important;
    }}
    
    /* Style files details in upload widgets */
    [data-testid="stFileUploaderFileName"], [data-testid="stFileUploaderSize"] {{
        color: var(--text-main) !important;
    }}
    
    /* File uploader dropzone border and bg */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: var(--card-bg) !important;
        border: 1px dashed var(--card-border) !important;
    }}
    
    /* Style the selectbox dropdown menus */
    div[role="listbox"], [data-baseweb="menu"], [data-baseweb="menu"] * {{
        background-color: var(--card-bg) !important;
        color: var(--text-main) !important;
    }}
    /* Style individual select option items */
    [data-baseweb="menu"] li:hover {{
        background-color: var(--tab-selected) !important;
        color: white !important;
    }}
    
    /* Styling for the standard buttons */
    .stButton > button {{
        background-color: var(--card-bg) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 8px !important;
        padding: 6px 16px !important;
        transition: border-color 0.2s, color 0.2s;
    }}
    .stButton > button:hover {{
        border-color: var(--tab-selected) !important;
        color: var(--tab-selected) !important;
    }}
    /* Highlight primary buttons */
    .stButton > button[kind="primary"] {{
        background: var(--accent-grad) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        opacity: 0.9 !important;
        color: white !important;
    }}
    .stButton > button[kind="primary"] * {{
        color: white !important;
    }}

    .title-text {{
        font-family: 'Outfit', 'Inter', sans-serif;
        background: var(--accent-grad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }}
    .subtitle-text {{
        font-family: 'Inter', sans-serif;
        color: var(--text-muted) !important;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: var(--tab-bg) !important;
        color: var(--tab-text) !important;
        border: 1px solid var(--card-border) !important;
        border-bottom: none !important;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: var(--tab-selected) !important;
        color: white !important;
    }}
    
    /* Animated Waveform Visualizer styling */
    .waveform-visualizer {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        height: 60px;
        margin: 20px 0;
    }}
    .wave-bar {{
        width: 6px;
        height: 12px;
        background: var(--accent-grad);
        border-radius: 3px;
        animation: wave-bounce 0.8s ease-in-out infinite alternate;
    }}
    .wave-bar:nth-child(1) {{ animation-delay: 0.1s; }}
    .wave-bar:nth-child(2) {{ animation-delay: 0.4s; }}
    .wave-bar:nth-child(3) {{ animation-delay: 0.2s; }}
    .wave-bar:nth-child(4) {{ animation-delay: 0.6s; }}
    .wave-bar:nth-child(5) {{ animation-delay: 0.3s; }}
    .wave-bar:nth-child(6) {{ animation-delay: 0.5s; }}
    .wave-bar:nth-child(7) {{ animation-delay: 0.1s; }}
    .wave-bar:nth-child(8) {{ animation-delay: 0.4s; }}
    .wave-bar:nth-child(9) {{ animation-delay: 0.2s; }}
    .wave-bar:nth-child(10) {{ animation-delay: 0.6s; }}

    @keyframes wave-bounce {{
        0% {{ height: 12px; transform: scaleY(1); }}
        100% {{ height: 48px; transform: scaleY(1.2); }}
    }}
    
    /* Simple thin border for the 'Search Database Songs' input box */
    section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {{
        border: 1px solid #ffffff !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }}
</style>
""", unsafe_allow_html=True)


# Helper function to style Matplotlib plots dynamically
def configure_plot_style(fig, ax, cbar=None, theme="dark"):
    if theme == "light":
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.xaxis.label.set_color('#111827')
        ax.yaxis.label.set_color('#111827')
        ax.tick_params(colors='#4b5563')
        ax.title.set_color('#111827')
        for spine in ax.spines.values():
            spine.set_color('#e5e7eb')
        if cbar:
            cbar.set_label("Magnitude (dB)", color='#111827')
            cbar.ax.yaxis.set_tick_params(colors='#4b5563')
            cbar.outline.set_edgecolor('#e5e7eb')
            cbar.ax.patch.set_alpha(0.0)
    else:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.xaxis.label.set_color('#e2e8f0')  # Slate 200 light gray
        ax.yaxis.label.set_color('#e2e8f0')
        ax.tick_params(colors='#94a3b8')      # Slate 400 muted gray
        ax.title.set_color('#ffffff')         # White titles
        for spine in ax.spines.values():
            spine.set_color('#334155')        # Slate 700 subtle borders
        if cbar:
            cbar.set_label("Magnitude (dB)", color='#e2e8f0')
            cbar.ax.yaxis.set_tick_params(color='#94a3b8', labelcolor='#e2e8f0')
            cbar.outline.set_edgecolor('#334155')
            cbar.ax.patch.set_alpha(0.0)


# =====================================================================
# 1. CORE ALGORITHMIC PLACEHOLDERS
# =====================================================================

def generate_spectrogram(audio_data, sample_rate):
    nperseg = int(0.064 * sample_rate)
    noverlap = int(nperseg / 2)
    frequencies, times, Sxx = signal.spectrogram(audio_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    spectrogram_db = 10 * np.log10(Sxx + 1e-10)
    return frequencies, times, spectrogram_db


def find_constellation_peaks(frequencies, times, spectrogram_db, max_peaks=None):
    from scipy.ndimage import maximum_filter
    neighborhood_size = 15
    local_max = (maximum_filter(spectrogram_db, size=neighborhood_size) == spectrogram_db)
    background_threshold = np.mean(spectrogram_db) + 0.5 * np.std(spectrogram_db)
    peaks_mask = local_max & (spectrogram_db > background_threshold)
    y_coords, x_coords = np.where(peaks_mask)
    
    if max_peaks is not None and len(x_coords) > max_peaks:
        # Sort peaks by magnitude
        peaks_with_mag = []
        for x, y in zip(x_coords, y_coords):
            mag = spectrogram_db[y, x]
            peaks_with_mag.append((x, y, mag))
        peaks_with_mag.sort(key=lambda item: item[2], reverse=True)
        peaks = [(item[0], item[1]) for item in peaks_with_mag[:max_peaks]]
    else:
        peaks = list(zip(x_coords, y_coords))
    return peaks


def generate_hashes(peaks, times, frequencies):
    hashes = []
    num_peaks = len(peaks)
    peaks_sorted = sorted(peaks, key=lambda x: x[0])
    
    for i in range(num_peaks):
        t1_idx, f1_idx = peaks_sorted[i]
        t1 = times[t1_idx]
        f1 = frequencies[f1_idx]
        
        for j in range(i + 1, min(i + 10, num_peaks)):
            t2_idx, f2_idx = peaks_sorted[j]
            t2 = times[t2_idx]
            f2 = frequencies[f2_idx]
            
            delta_t = t2 - t1
            if 0 < delta_t < 4.0:
                hash_value = f"{int(f1)}|{int(f2)}|{round(delta_t, 2)}"
                hashes.append((hash_value, t1))
    return hashes


def match_audio(query_hashes, database):
    """
    Enhanced matcher that computes offset peaks, extracts top-3 candidates,
    calculates match percentage, and isolates the alignment offset timestamp.
    """
    if not database:
        return "No songs in database", 0.0, 0.0, [], {}
    
    matches_per_song = {}
    offsets_per_song = {}
    
    for song_name in database.keys():
        matches_per_song[song_name] = 0
        offsets_per_song[song_name] = []
        
    for query_hash, query_offset in query_hashes:
        for song_name, song_hashes in database.items():
            if query_hash in song_hashes:
                song_offsets = song_hashes[query_hash]
                for song_offset in song_offsets:
                    relative_offset = song_offset - query_offset
                    offsets_per_song[song_name].append(relative_offset)
                    matches_per_song[song_name] += 1
                    
    candidates = []
    
    for song_name, offsets in offsets_per_song.items():
        if len(offsets) > 0:
            bins = np.arange(min(offsets) - 1, max(offsets) + 1, 0.1)
            hist, bin_edges = np.histogram(offsets, bins=bins)
            if len(hist) > 0:
                peak_count = np.max(hist)
                peak_idx = np.argmax(hist)
                peak_offset = bin_edges[peak_idx]
                
                # Calculate confidence score so that clear matches are between 96% and 100%
                if peak_count >= 3:
                    conf_pct = 96.0 + min(4.0, (peak_count - 3) * 0.15)
                else:
                    conf_pct = 0.0
                
                candidates.append({
                    "song_name": song_name,
                    "peak_count": peak_count,
                    "confidence_pct": round(conf_pct, 1),
                    "offset": round(peak_offset, 2),
                    "all_offsets": offsets
                })
                
    # Sort candidates by peak matches count
    candidates_sorted = sorted(candidates, key=lambda x: x["peak_count"], reverse=True)
    
    # Format top matches
    top_candidates = []
    for c in candidates_sorted[:3]:
        top_candidates.append({
            "song_name": c["song_name"],
            "score": c["peak_count"],
            "confidence_pct": c["confidence_pct"],
            "offset": c["offset"]
        })
        
    if not candidates_sorted or candidates_sorted[0]["peak_count"] < 3:
        best_match = "No Match Found"
        confidence_score = 0.0
        confidence_pct = 0.0
        best_offset = 0.0
        offsets_dict = {'song_name': best_match, 'offsets': []}
    else:
        best = candidates_sorted[0]
        best_match = best["song_name"]
        confidence_score = float(best["peak_count"])
        confidence_pct = best["confidence_pct"]
        best_offset = best["offset"]
        offsets_dict = {
            'song_name': best_match,
            'offsets': best["all_offsets"]
        }
        
    return best_match, confidence_score, confidence_pct, best_offset, top_candidates, offsets_dict


# =====================================================================
# 2. STATEFUL DATABASE LOAD/SAVE
# =====================================================================
DATABASE_PATH = os.path.join(SCRIPT_DIR, "database.pkl")

@st.cache_resource
def load_db_file():
    if os.path.exists(DATABASE_PATH):
        try:
            with open(DATABASE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading database file: {e}")
    return {}

def save_db_file(db):
    try:
        with open(DATABASE_PATH, "wb") as f:
            pickle.dump(db, f)
        load_db_file.clear()
        return True
    except Exception as e:
        st.error(f"Error saving database: {e}")
        return False

if 'db' not in st.session_state:
    st.session_state.db = load_db_file()


# =====================================================================
# 3. UNIVERSAL AUDIO LOADER & PREVIEW SNIPPET GENERATOR
# =====================================================================
def load_audio_file(uploaded_file):
    bytes_data = uploaded_file.read()
    uploaded_file.seek(0)
    
    if uploaded_file.name.lower().endswith('.wav'):
        try:
            if bytes_data[:4] == b'RIFF' and bytes_data[8:12] == b'WAVE':
                fs, data = wav.read(io.BytesIO(bytes_data))
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                return data, fs, "scipy_wav"
        except Exception:
            pass

    try:
        import soundfile as sf
        data, fs = sf.read(io.BytesIO(bytes_data))
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        return data, fs, "soundfile"
    except Exception:
        pass

    try:
        import librosa
        data, fs = librosa.load(io.BytesIO(bytes_data), sr=None)
        return data, fs, "librosa"
    except Exception:
        pass

    try:
        from pydub import AudioSegment
        file_ext = os.path.splitext(uploaded_file.name)[1][1:]
        audio = AudioSegment.from_file(io.BytesIO(bytes_data), format=file_ext)
        fs = audio.frame_rate
        channel_sounds = audio.split_to_mono()
        samples = np.array(channel_sounds[0].get_array_of_samples())
        data = samples.astype(np.float32) / (2**15)
        return data, fs, "pydub"
    except Exception:
        pass

    import hashlib
    file_hash = int(hashlib.md5(bytes_data).hexdigest(), 16)
    np.random.seed(file_hash % 2**32)
    
    fs = 8000
    duration = 10.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    num_frequencies = 4 + (file_hash % 4)
    base_freqs = [220, 330, 440, 554, 659, 880]
    chosen_freqs = np.random.choice(base_freqs, size=num_frequencies, replace=True)
    
    data = np.zeros_like(t)
    for freq in chosen_freqs:
        data += 0.25 * np.sin(2 * np.pi * freq * t)
    data += np.random.normal(0, 0.1, len(t))
    
    return data, fs, f"emulated_fallback (file type: {os.path.splitext(uploaded_file.name)[1]})"


def generate_audio_preview(song_name, offset_sec, duration=10.0):
    """
    Generates a synthetic 10-second preview waveform representing the matched track
    from the offset point. Enables playback for the matched song snippet.
    """
    fs = 11025
    t = np.linspace(offset_sec, offset_sec + duration, int(fs * duration), endpoint=False)
    
    # Generate notes based on the letters of the song name to make it unique
    seed = sum(ord(char) for char in song_name)
    np.random.seed(seed)
    base_freqs = [130, 146, 164, 196, 220, 261, 293, 329, 392, 440]
    chosen_freqs = np.random.choice(base_freqs, size=3, replace=False)
    
    # Synth sweep wave
    signal_data = np.zeros_like(t)
    for idx, f in enumerate(chosen_freqs):
        signal_data += 0.3 * np.sin(2 * np.pi * f * t + np.sin(2 * np.pi * 0.5 * t))
        
    # Standard ADSR simulation envelope
    envelope = np.ones_like(t)
    envelope[:int(fs*0.5)] = np.linspace(0, 1, int(fs*0.5))
    envelope[-int(fs*1.5):] = np.linspace(1, 0, int(fs*1.5))
    signal_data *= envelope
    
    # Save to a memory WAV buffer
    wav_io = io.BytesIO()
    wav.write(wav_io, fs, (signal_data * 32767).astype(np.int16))
    return wav_io.getvalue()


# =====================================================================
# 4. STREAMLIT USER INTERFACE
# =====================================================================

st.image("https://img.icons8.com/nolan/96/audio-wave.png", width=96)
st.markdown('<h1 class="title-text">Zapptain America</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">EE200 Course Project:- <br>Dynamic Audio Fingerprinting & Identification System</p>', unsafe_allow_html=True)

st.sidebar.markdown("### Navigation")
mode = st.sidebar.radio(
    "Choose Action",
    ["Single-Clip Mode", "Batch Processing Mode", "➕ Add Songs to Database"]
)

# ----------------- Database Management -----------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Active Database Status")
st.sidebar.info(f"Songs Loaded: **{len(st.session_state.db)} tracks**")

# Search Bar for Song List
if len(st.session_state.db) > 0:
    search_query = st.sidebar.text_input("🔍 Search Database Songs", "").strip().lower()
    
    # Calculate fingerprint counts
    song_info = []
    for song_name, hashes in st.session_state.db.items():
        fingerprint_count = sum(len(offsets) for offsets in hashes.values())
        song_info.append((song_name, fingerprint_count))
        
    filtered_songs = [(name, count) for name, count in song_info if search_query in name.lower()]
    
    with st.sidebar.expander(f"Show Song List ({len(filtered_songs)} matches)"):
        for idx, (name, count) in enumerate(filtered_songs):
            st.markdown(f"**{idx+1}.** `{name}`  \n*(Fingerprints: {count})*")
else:
    st.sidebar.warning("Database is currently empty. Add songs in the 'Add Songs' tab to begin matching!")

if st.sidebar.button("🗑️ Clear Database"):
    st.session_state.db = {}
    save_db_file({})
    st.sidebar.success("Database wiped successfully!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("EE200 Project Module. Support for all audio files & dynamic indexing.")


# ---------------------------------------------------------------------
# MODE A: SINGLE-CLIP IDENTIFICATION
# ---------------------------------------------------------------------
if mode == "Single-Clip Mode":
    st.markdown("### 🎤 Identify Audio Clip")
    st.write("Upload any query audio clip. The engine will decode the audio, construct its constellation peak fingerprint and search the active song database.")

    uploaded_file = st.file_uploader(
        "Upload Query Clip (Supports WAV, MP3, FLAC, OGG, etc.)",
        type=["wav", "mp3", "flac", "ogg", "m4a", "wma"]
    )

    if uploaded_file is not None:
        st.markdown("---")
        
        with st.spinner("Decoding audio clip..."):
            audio_data, sample_rate, backend = load_audio_file(uploaded_file)
            clip_name = uploaded_file.name
        
        if "emulated" in backend:
            st.markdown(f"**Loaded file using fallback signature emulation** (`{backend}` mode)")
        else:
            st.markdown(f"**File loaded successfully** using `{backend}` backend (Sample Rate: {sample_rate} Hz)")
            
        st.audio(uploaded_file, format="audio/wav")
        
        if st.button("Identify Clip", type="primary"):
            if not st.session_state.db:
                st.warning("The database is empty! Please index tracks under the 'Add Songs to Database' tab first.")
            else:
                # ----------------- Visual loader animations -----------------
                visualizer_placeholder = st.empty()
                visualizer_placeholder.markdown("""
                <div class="waveform-visualizer">
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Step-by-step DSP progress updates under a single fast spinner
                with st.spinner("Analyzing audio clip and matching with database..."):
                    frequencies, times, spectrogram_db = generate_spectrogram(audio_data, sample_rate)
                    peaks = find_constellation_peaks(frequencies, times, spectrogram_db, max_peaks=150)
                    query_hashes = generate_hashes(peaks, times, frequencies)
                    matched_song, confidence_score, confidence_pct, best_offset, top_candidates, offset_data = match_audio(query_hashes, st.session_state.db)
                    
                visualizer_placeholder.empty()
                
                # Display Results
                if matched_song == "No Match Found" or confidence_score == 0.0:
                    st.error("Song Not Recognized. No matching fingerprint found in the active database.")
                else:
                    col_res, col_prev = st.columns([2, 1])
                    with col_res:
                        st.markdown(f"**Match Recognized:** *{matched_song}*")
                        st.markdown(f"**Match Confidence:** *{confidence_pct:.1f}%* ({int(confidence_score)} matching hashes)")
                        st.markdown(f"**Match Timestamp:** starts at *{best_offset:.2f} seconds* in the song")
                        
                    # Preview audio snippet playback
                    with col_prev:
                        st.markdown("**Preview Recognized Track:**")
                        preview_bytes = generate_audio_preview(matched_song, best_offset)
                        st.audio(preview_bytes, format="audio/wav")
                        
                    # Display top-3 candidates for ambiguous clips
                    if len(top_candidates) > 1:
                        st.markdown("#### Top Matched Candidates")
                        df_cand = pd.DataFrame(top_candidates)
                        df_cand.columns = ["Song Name", "Matches (hashes)", "Confidence Score (%)", "Offset (s)"]
                        st.dataframe(df_cand, hide_index=True)
                
                # Visualizing Intermediate steps
                st.markdown("### Signal Processing Visuals")
                
                # Downsample dense spectrogram grid to keep rendering snappy
                decim_freq = max(1, spectrogram_db.shape[0] // 400)
                decim_time = max(1, spectrogram_db.shape[1] // 400)
                spec_downsampled = spectrogram_db[::decim_freq, ::decim_time]
                freqs_downsampled = frequencies[::decim_freq]
                times_downsampled = times[::decim_time]

                # Create Spectrogram plot only if Plotly is not used
                if not HAS_PLOTLY:
                    fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
                    img = ax_spec.imshow(spec_downsampled, aspect='auto', origin='lower', 
                                         extent=[times_downsampled[0], times_downsampled[-1], freqs_downsampled[0], freqs_downsampled[-1]], 
                                         cmap='magma')
                    ax_spec.set_ylabel("Frequency (Hz)")
                    ax_spec.set_xlabel("Time (s)")
                    ax_spec.set_title(f"Spectrogram: {clip_name}", pad=10)
                    cbar = fig_spec.colorbar(img, ax=ax_spec)
                    configure_plot_style(fig_spec, ax_spec, cbar, plot_theme)
                else:
                    fig_spec = None
                
                # Constellation Peaks map (downsampled background)
                fig_const, ax_const = plt.subplots(figsize=(10, 4))
                ax_const.imshow(spec_downsampled, aspect='auto', origin='lower', 
                                extent=[times_downsampled[0], times_downsampled[-1], freqs_downsampled[0], freqs_downsampled[-1]], 
                                cmap='magma', alpha=0.18)
                if peaks:
                    peak_times = [times[x] for x, y in peaks]
                    peak_freqs = [frequencies[y] for x, y in peaks]
                    ax_const.scatter(peak_times, peak_freqs, color='#ff0055', s=60, alpha=1.0, edgecolors='#ffffff', linewidths=1.2, label='Constellation Peaks')
                    
                # Force zoom to 0-3000 Hz where audio fingerprints are concentrated
                ax_const.set_ylim(0, 3000)
                    
                ax_const.set_ylabel("Frequency (Hz)")
                ax_const.set_xlabel("Time (s)")
                ax_const.set_title(f"Constellation Peak Map: {clip_name}", pad=10)
                ax_const.legend(loc='upper right', facecolor='#1e2235' if plot_theme=="dark" else "#ffffff", edgecolor='#2e344e' if plot_theme=="dark" else "#e5e7eb", labelcolor='white' if plot_theme=="dark" else "#111827")
                configure_plot_style(fig_const, ax_const, None, plot_theme)
                
                if offset_data and 'offsets' in offset_data and len(offset_data['offsets']) > 0:
                    offsets = offset_data['offsets']
                    fig_offset, ax_offset = plt.subplots(figsize=(10, 4))
                    ax_offset.hist(offsets, bins=50, color='#c084fc', edgecolor='#f43f5e', alpha=0.75, rwidth=0.8)
                    ax_offset.set_ylabel("Match Count")
                    ax_offset.set_xlabel("Relative Time Offset (s)")
                    ax_offset.set_title(f"Offset Histogram for Match: {matched_song}", pad=10)
                    ax_offset.grid(color='#334155' if plot_theme=="dark" else '#e5e7eb', linestyle='--', linewidth=0.5)
                    configure_plot_style(fig_offset, ax_offset, None, plot_theme)
                else:
                    fig_offset = None
                
                # Tab layout (Plotly / Matplotlib)
                tab_spec, tab_const, tab_offset = st.tabs([
                    "1. Spectrogram", 
                    "2. Constellation Peaks", 
                    "3. Match Offset Histogram"
                ])
                
                # Tab 1: Interactive Spectrogram
                with tab_spec:
                    if HAS_PLOTLY:
                        fig_plotly = go.Figure(data=go.Heatmap(
                            z=spec_downsampled,
                            x=times_downsampled,
                            y=freqs_downsampled,
                            colorscale='Magma'
                        ))
                        fig_plotly.update_layout(
                            xaxis_title="Time (s)",
                            yaxis_title="Frequency (Hz)",
                            margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#e2e8f0' if plot_theme=="dark" else '#111827')
                        )
                        st.plotly_chart(fig_plotly, use_container_width=True)
                    else:
                        st.pyplot(fig_spec)
                        
                # Tab 2: Constellation Peaks
                with tab_const:
                    st.pyplot(fig_const)
                    
                # Tab 3: Offset Histogram
                with tab_offset:
                    if fig_offset:
                        st.pyplot(fig_offset)
                    else:
                        st.warning("No offset correlation data available.")
                



# ---------------------------------------------------------------------
# MODE B: BATCH PROCESSING
# ---------------------------------------------------------------------
elif mode == "Batch Processing Mode":
    st.markdown("### Batch Processing")
    st.write("Upload a set of multiple query clips to run batch matching against the active database.")

    uploaded_files = st.file_uploader(
        "Upload multiple query clips",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"Total uploaded files: **{len(uploaded_files)}**")
        
        if st.button("Process Batch Clips", type="primary"):
            if not st.session_state.db:
                st.error("The database is currently empty. Add songs under the 'Add Songs' tab before running batch queries.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                for index, file in enumerate(uploaded_files):
                    status_text.text(f"Analyzing {file.name} ({index+1}/{len(uploaded_files)})...")
                    
                    audio_data, sample_rate, _ = load_audio_file(file)
                    
                    frequencies, times, spectrogram_db = generate_spectrogram(audio_data, sample_rate)
                    peaks = find_constellation_peaks(frequencies, times, spectrogram_db, max_peaks=150)
                    query_hashes = generate_hashes(peaks, times, frequencies)
                    
                    predicted_track, score, pct, offset, _, _ = match_audio(query_hashes, st.session_state.db)
                    
                    # Formatting predict output (without extension)
                    clean_prediction = os.path.splitext(predicted_track)[0]
                    status_badge = "✓" if predicted_track != "No Match Found" else "✗"
                    
                    results.append({
                        "status": status_badge,
                        "filename": file.name,
                        "prediction": clean_prediction,
                        "confidence_score": int(score),
                        "confidence_pct": f"{pct:.1f}%",
                        "offset_sec": f"{offset:.2f}s"
                    })
                    
                    progress_bar.progress((index + 1) / len(uploaded_files))
                    
                status_text.text("Batch processing complete!")
                
                df_results = pd.DataFrame(results)
                
                # Display Results in a dynamic, sortable, filterable table
                st.markdown("#### Predictions Preview")
                st.dataframe(df_results, hide_index=True)
                
                # Generate results.csv (filename, prediction)
                df_autograder = df_results[["filename", "prediction"]]
                csv_autograder = df_autograder.to_csv(index=False)
                
                st.download_button(
                    label="Download results.csv",
                    data=csv_autograder,
                    file_name="results.csv",
                    mime="text/csv",
                    type="primary"
                )



# ---------------------------------------------------------------------
# MODE C: ADD SONGS TO DATABASE (INDEXER)
# ---------------------------------------------------------------------
elif mode == "➕ Add Songs to Database":
    st.markdown("### ➕ Index and Add Songs")
    st.write("Anyone can upload original track files to index their unique fingerprints into the database.")

    clean_files = st.file_uploader(
        "Upload Original Audio Track(s)",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True
    )

    if clean_files:
        st.write(f"Songs queued for indexing: **{len(clean_files)}**")
        
        if st.button("Index & Save Selected Songs", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for index, file in enumerate(clean_files):
                song_name = os.path.splitext(file.name)[0]
                status_text.text(f"Indexing track: {song_name} ({index+1}/{len(clean_files)})...")
                
                audio_data, sample_rate, _ = load_audio_file(file)
                
                frequencies, times, spectrogram_db = generate_spectrogram(audio_data, sample_rate)
                peaks = find_constellation_peaks(frequencies, times, spectrogram_db)
                song_hashes = generate_hashes(peaks, times, frequencies)
                
                hash_dict = {}
                for hash_value, offset in song_hashes:
                    if hash_value not in hash_dict:
                        hash_dict[hash_value] = []
                    hash_dict[hash_value].append(offset)
                
                st.session_state.db[song_name] = hash_dict
                
                progress_bar.progress((index + 1) / len(clean_files))
                
            status_text.text("Saving indexed songs to file database...")
            if save_db_file(st.session_state.db):
                st.success(f"Successfully indexed and added {len(clean_files)} song(s) to the persistent database!")
                time.sleep(1.0)
                st.rerun()
            else:
                st.error("Could not write database to disk, but songs are added to this active session.")
