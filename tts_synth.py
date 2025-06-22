import subprocess

def synthesize_tts(lines, output="rap_out.wav", bpm=90, voice="mbrola"):
    from pathlib import Path
    audio_paths = []
    for i, line in enumerate(lines):
        txt_file = f"line_{i}.txt"
        wav_file = f"line_{i}.wav"
        Path(txt_file).write_text(f"{bpm}\n{line}\n_{line.split()[-1]}")
        subprocess.run(["python3", "espeak.py", txt_file])
        audio_paths.append(wav_file)
    # use Anchovie/Rap-synthesis to mix beat + vocals
    # you'll need its repo and dependencies installed :contentReference[oaicite:1]{index=1}
    return output
