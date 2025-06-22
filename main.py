from beat_detector import extract_beats
from lyric_generator import load_gpt2_rap, generate_rap
from aligner import align_lines_to_beats

def main():
    beats = extract_beats('beat.wav')
    model, tok = load_gpt2_rap()
    lines = generate_rap(model, tok, "step up to the mic", length=80)
    timing = align_lines_to_beats(lines, beats)
    for t, line in timing:
        print(f"{t:.2f}s → {line}")

if __name__ == "__main__":
    print("\n--- Rhyme & Meter Analysis ---")
    meter, rhyme_scheme = analyze_meter_and_rhyme_block(lines)
    print(f"Meter: {meter}\nRhyme scheme: {rhyme_scheme}")

    print("\n--- Highlighted Rhymes ---")
    print(highlight_full_rhymes("\n".join(lines)))

    main()
