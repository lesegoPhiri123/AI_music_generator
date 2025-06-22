# chorus_generator.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_chorus_model(model_name='Elida‐Sensoy/gpt2-rap-generator'):
    tok = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tok

def generate_chorus(model, tok, theme, length=40):
    prompt = f"[Chorus]\n{theme}\n"
    inputs = tok.encode(prompt, return_tensors='pt')
    out = model.generate(inputs, max_length=length, do_sample=True, top_k=50)
    text = tok.decode(out[0], skip_special_tokens=True)
    # Extract chorus block
    lines = [l.strip() for l in text.split('\n') if l.strip() and not l.startswith('[Verse]')]
    return lines
