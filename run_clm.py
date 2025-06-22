# # preprocess your 2Pac lyrics into train.txt and val.txt
# python run_clm.py \
#   --model_name_or_path gpt2-medium \
#   --train_file 2pac_train.txt \
#   --validation_file 2pac_val.txt \
#   --output_dir ./gpt2_2pac
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained('./gpt2_2pac')
model = GPT2LMHeadModel.from_pretrained('./gpt2_2pac')

prompt = "Dear mama, I"  # begin in 2Pac style
out = model.generate(tok.encode(prompt, return_tensors='pt'),
                     max_length=150, do_sample=True, top_k=50)
lyrics = tok.decode(out[0], skip_special_tokens=True).split('\n')
