from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def load_gpt2_rap(model_name='AnantShankhdhar/AI-Rap-Lyric-Generator'):
    tok = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tok

def generate_rap(model, tok, seed, length=80):
    inputs = tok.encode(seed, return_tensors='pt')
    out = model.generate(inputs, max_length=length, do_sample=True, top_k=50)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split('\n')  # returns list of lines


def load_model_and_tok(model_name='gpt2'):
    """
    Load GPT-2 model and tokenizer.
    Returns the model (in eval mode) and tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

def generate_lyrics(model, tokenizer, prompt, lines=8, max_length=100, temperature=1.0):
    """
    Generate lyrics using GPT-2.
    Returns a string with the generated lines separated by newline.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    sample_output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        top_k=50
    )
    text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    # Optionally trim to `lines` number of newline breaks
    return '\n'.join(text.split('\n')[:lines])
