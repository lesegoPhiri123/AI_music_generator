"""
pipeline.py — Main entry for rap-flow AI.
Usage:
    python pipeline.py --beat beat.wav --seed "step up to mic" --model gpt2 --lines 8
"""
import os
import argparse
import random
from beat_detector import extract_beats
from lyric_generator import load_model_and_tok, generate_lyrics
from aligner import align_lines_to_beats
from flow_extractor import sample_flow
from chorus_generator import load_chorus_model, generate_chorus
from voice_cloning import (
    init_tts,
    clone_voice,
    init_xtts,
    clone_with_xtts,
    init_rvc,
    convert_with_rvc,
    synthesize_audio,
)
from util import add_lyric, add_flow, load_data

def main():
    parser = argparse.ArgumentParser(description="Rap-flow AI pipeline")
    parser.add_argument("--beat", required=True, help="Path to beat .wav file")
    parser.add_argument("--seed", default="step up to the mic", help="Seed text for lyrics")
    parser.add_argument("--model", choices=["gpt2", "lstm"], default="gpt2", help="Choose lyric model: gpt2 or lstm")
    parser.add_argument("--lines", type=int, default=8, help="Number of lyric lines to generate")
    parser.add_argument("--length", type=int, default=100, help="Max tokens/words to generate")
    parser.add_argument("--lstm_model", default="model.h5", help="Path to LSTM .h5 model")
    parser.add_argument("--lstm_tok", default="tokenizer.pkl", help="Path to LSTM tokenizer")
    parser.add_argument("--voice_wav", help="Path to a voice sample .wav for cloning")
    parser.add_argument("--synth_out", default="rap_voice.wav", help="Output path for synthesized voice")
    args = parser.parse_args()

    # 1. Beat Extraction
    beats = extract_beats(args.beat)
    print(f"Detected {len(beats)} beats.")

    # 2. Lyrics Generation
    model, tok = load_model_and_tok(args.model, args.lstm_model, args.lstm_tok)
    raw = generate_lyrics(model, tok, args.model, args.seed, args.length)
    lines = [l.strip() for l in raw.split('\n') if l.strip()][:args.lines]

    # 3. Alignment
    aligned = align_lines_to_beats(lines, beats)
    print("\n=== Rap Flow Output ===")
    for t, line in aligned:
        print(f"{t:.2f}s → {line}")

    # 4. Flow Sample
    snippet = sample_flow(artists=["2Pac", "Kendrick"])
    seed = snippet["text"]
    print("\n--- Flow Sample ---")
    print(seed)

    # 5. Chorus Generation
    chorus_model, chorus_tok = load_chorus_model()
    chorus_lines = generate_chorus(chorus_model, chorus_tok, args.seed, length=60)
    print("\n--- Chorus ---")
    for line in chorus_lines:
        print(line)

    # 6. Voice Cloning (Optional)
    if args.voice_wav:
        tts = init_tts(use_gpu=True)
        lyrics_text = "\n".join(lines)
        clone_voice(tts, lyrics_text, args.voice_wav, args.synth_out)
        print(f"🎤 Synthesized rap in your voice → {args.synth_out}")

        full_text = "\n".join(lines)
        tts = init_xtts(use_gpu=True)
        tts_wav = clone_with_xtts(tts, full_text, args.voice_wav, "temp_tts.wav")
        print("🎤 Synthesized with XTTS in your voice")

        rvc = init_rvc()
        final_wav = convert_with_rvc(rvc, tts_wav, args.synth_out)
        print(f"✅ Final rap output: {final_wav}")

    # 7. Add New Lyric and Flow
    add_lyric("I'm climbing to the top, never gonna drop.")
    add_flow("Smooth and laid-back, with a jazzy vibe.")

def add_lyric(new_lyric):
    """Appends a new lyric to the lyrics.txt file."""
    with open('data/lyrics.txt', 'a') as file:
        file.write(f'"{new_lyric}"\n')

def add_flow(new_flow):
    """Appends a new flow description to the flows.txt file."""
    with open('data/flows.txt', 'a') as file:
        file.write(f'"{new_flow}"\n')

if __name__ == "__main__":
    main()
