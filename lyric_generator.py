from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_gpt2_rap(model_name='AnantShankhdhar/AI-Rap-Lyric-Generator'):
    tok = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tok

def generate_rap(model, tok, seed, length=80):
    inputs = tok.encode(seed, return_tensors='pt')
    out = model.generate(inputs, max_length=length, do_sample=True, top_k=50)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split('\n')  # returns list of lines
