# flow_extractor.py
import json
import random

DB_PATH = "flow_db.json"

def load_flow_db():
    with open(DB_PATH) as f:
        return json.load(f)

def sample_flow(piece_length=4, artists=None):
    db = load_flow_db()
    candidates = [ln for rec in db
                  if artists is None or rec["artist"] in artists
                  for ln in rec["lines"]]
    # pick random phrase matching number of beats
    return random.choice(candidates)
