# 🎵 AI Music Generation using LSTM and OpenAI Jukebox

A dual-approach AI music generation system combining LSTM-based sequential 
note prediction trained on MIDI datasets with OpenAI's Jukebox — a 5B 
parameter VQ-VAE transformer model — for raw audio generation and analysis.

## 🧠 Approaches Used

### 1. LSTM-Based MIDI Generation
- Parses MIDI files using music21 to extract notes and chords
- Builds note sequences and trains an LSTM model to predict the next note
- Generates new MIDI compositions from learned musical patterns
- Plays back generated MIDI using pygame

### 2. OpenAI Jukebox (5B Lyrics Model)
- Uses OpenAI's pretrained Jukebox VQ-VAE transformer for raw audio generation
- Supports artist and genre conditioning for stylistic control
- Implements multi-level hierarchical upsampling for high-quality audio output
- Analyses generated audio using librosa chromagrams for frequency and pitch 
  pattern visualization

## 📦 Tech Stack

- **Python** – Core language
- **Keras / TensorFlow** – LSTM model training
- **music21** – MIDI parsing and note extraction
- **OpenAI Jukebox** – Transformer-based audio generation
- **librosa** – Audio analysis and chromagram visualization
- **pygame** – MIDI playback
- **pydub** – MP3 to WAV conversion
- **Google Colab + Tesla P100 GPU** – Training and inference environment

## 📁 Project Structure

AI-Music-Generation-LSTM-Jukebox/
├── aimusic/
│   ├── samples/              # Generated audio samples (WAV)
│   └── sampler.ipynb         # Jukebox co-composition notebook
├── midihelper/
│   ├── midihelper.py         # MIDI note extraction and sequence builder
│   ├── play.py               # MIDI player entry point
│   └── playhelper.py         # pygame-based MIDI playback helper
├── wavhelper/
│   ├── mp32wav.py            # MP3 to WAV converter
│   └── something.mp3         # Sample input audio
├── build.sh                  # Environment setup script
└── README.md

## 🔧 Setup and Installation

### Prerequisites
```bash
sudo apt-get install freepats
sudo apt-get install timidity
pip install pygame pydub music21
pip install git+https://github.com/openai/jukebox.git
```

### LSTM Music Generation

1. Place your MIDI files in a folder (e.g. `midi_data/`)
2. Parse notes and train the model:
```bash
python midi_parser.py
python train.py
```
3. Generate new music:
```bash
python generate.py
```
4. Play the generated MIDI:
```bash
python play_midi.py --midi_file output.mid --volume 0.8
```

### MP3 to WAV Conversion
```bash
python mp3_to_wav.py --input song.mp3 --output song.wav
```

### Jukebox Generation
Open `jukebox_generation.ipynb` in Google Colab with a GPU runtime and 
follow the steps for co-composition and upsampling.

## 🎼 How It Works

**LSTM Pipeline:**
MIDI files → Note extraction via music21 → Sequence encoding → LSTM 
training → Note prediction → MIDI generation → Playback via pygame

**Jukebox Pipeline:**
Artist/genre/lyrics conditioning → Top-level token sampling → Multi-level 
hierarchical upsampling → High-fidelity WAV output → Chromagram analysis 
via librosa

## 📊 Audio Analysis
Generated audio is analysed using librosa chromagrams to visualize pitch 
class distribution and verify musical coherence across generation levels.

## ⚠️ Notes
- Jukebox upsampling is computationally intensive — full upsampling of a 
  20-second clip can take 4–12 hours on a single GPU
- Free-tier Google Colab sessions may time out during upsampling; save 
  checkpoints regularly
- LSTM model quality improves significantly with larger MIDI datasets

## 👨‍💻 Author
Developed by Harshwardhan Misal
CodeAlpha Virtual AI Internship — December 2024 to January 2025
