from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

def extract_beats(audio_path):
    act = RNNBeatProcessor()(audio_path)
    beats = DBNBeatTrackingProcessor(fps=100)(act)
    return beats  # timestamps in seconds
