# voice_cloning.py
import pyttsx3
from TTS.api import TTS
from rvc.api import RVC  # hypothetical import—adjust per your RVC installation

def init_xtts(model="tts_models/multilingual/multi-dataset/xtts_v2", use_gpu=True):
    return TTS(model, gpu=use_gpu)

def clone_with_xtts(tts, text, speaker_wav, outpath):
    return tts.tts_to_file(text=text, speaker_wav=speaker_wav, file_path=outpath)

def init_rvc(model_dir="rvc_model.pth", use_gpu=True):
    return RVC(model_dir, gpu=use_gpu)

def convert_with_rvc(rvc, tts_wav, outpath):
    return rvc.convert(tts_wav, outpath)


def synthesize_audio(text, output_path):
    """Converts text to speech and saves it as an audio file."""
    engine = pyttsx3.init()
    engine.save_to_file(text, output_path)
    engine.runAndWait()

