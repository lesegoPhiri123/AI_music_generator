# requirements: pip install madmom tensorflow keras numpy soundfile
# imports
import numpy as np
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
import soundfile as sf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Detect beats from .wav audio
def extract_beats(audio_path):
    act = RNNBeatProcessor()(audio_path)
    beats = DBNBeatTrackingProcessor(fps=100)(act)
    return beats

# 2A. LSTM-based lyric generator (Spectrum style)
def load_lstm_model(model_path, tok_path):
    model = load_model(model_path)
    tok = pickle.load(open(tok_path,'rb'))
    return model, tok

def generate_with_lstm(model, tokenizer, seed, num_words=50, maxlen=20):
    seq = tokenizer.texts_to_sequences([seed])[0]
    result = seed.split()
    for _ in range(num_words):
        padded = pad_sequences([seq], maxlen=maxlen, padding='pre')
        preds = model.predict(padded)[0]
        idx = np.argmax(preds)
        word = tokenizer.index_word.get(idx, '')
        result.append(word)
        seq.append(idx)
    return ' '.join(result)

# 2B. GPT‑2 lyric generator (rap‑finetuned)
def load_gpt2_model():
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('Elida-Sensoy/gpt2-rap-generator')
    return model, tok

def generate_with_gpt2(model, tokenizer, seed, length=100):
    inputs = tokenizer.encode(seed, return_tensors='pt')
    out = model.generate(inputs, max_length=length, do_sample=True, top_k=50)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# 3. Align lyric lines to beat timestamps
def align_to_beats(text, beats, lines=8):
    parts = text.split('\n') if '\n' in text else text.split('. ')
    chosen = parts[:lines]
    times = beats[:len(chosen)]
    return list(zip(times, chosen))

# Demo run
beats = extract_beats('beat.wav')
print(f"Beats detected: {len(beats)}")

# Use either LSTM or GPT‑2
# lstm_model, lstm_tok = load_lstm_model('model.h5', 'tokenizer.pkl')
# lyrics = generate_with_lstm(lstm_model, lstm_tok, "step up to mic", 50)

gpt2_model, gpt2_tok = load_gpt2_model()
lyrics = generate_with_gpt2(gpt2_model, gpt2_tok, "step up to mic", length=80)

aligned = align_to_beats(lyrics, beats, lines=8)
for t, line in aligned:
    print(f"{t:.2f}s → {line.strip()}")
