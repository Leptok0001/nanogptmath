"""
Standalone script to test the model's accuracy on q/a pairs from problems.txt
"""

import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
init_from = 'resume'   # typically 'resume' to load from out/ckpt.pt
out_dir = 'out-math-char'        # where ckpt.pt is located
problems_file = '/content/nanogptmath/data/math_char/problems.txt'  # path to your q/a file

max_new_tokens = 3     # how many tokens to generate for each question
temperature = 0.8
top_k = 200
seed = 1337
device = 'cpu'
print("using CPU")
dtype = ('bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
         else 'float16')
compile_model = False

# If you want to log incorrect predictions:
log_incorrect_file = 'incorrect_predictions.txt'

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix any prefix
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError(f"Unknown init_from: {init_from}")

model.eval()
model.to(device)
if compile_model:
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Load or assume encodings
# -----------------------------------------------------------------------------
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi.get(c, 0) for c in s]  # unknown chars => 0
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# -----------------------------------------------------------------------------
# Read problems.txt into a list of (q, a) pairs
# -----------------------------------------------------------------------------
pairs = []
with open(problems_file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

# Each question is on one line, answer on next line
assert len(lines) % 2 == 0, "problems.txt should have an even number of lines (q, a, q, a, ...)"

for i in range(0, len(lines), 2):
    q = lines[i]
    a = lines[i+1]
    pairs.append((q, a))

print(f"Loaded {len(pairs)} question/answer pairs from {problems_file}.")

# -----------------------------------------------------------------------------
# Evaluate each Q/A pair
# -----------------------------------------------------------------------------
num_correct = 0
num_wrong = 0
incorrect_records = []  # store (question, correct_answer, model_answer)

with torch.no_grad(), ctx:
    for (q, a) in pairs:
        # Encode question
        q_ids = encode(q)
        x = torch.tensor(q_ids, dtype=torch.long, device=device)[None, ...]

        # Generate model answer
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
        out_tokens = y[0].tolist()

        # The entire text (prompt + new tokens)
        full_str = decode(out_tokens)
        # Optionally, strip the prompt (the question) to isolate model's answer
        # For example, if the question was "12+7=", length_of_q = len(q_ids).
        # But GPT-2 style encoding might not match char length exactly.
        # A simpler approach is to compare the substring after the question:
        # (But if your model sometimes repeats the question, you may need more robust parsing.)
        model_answer_part = full_str[len(q):].strip()

        # We'll just do a naive .strip() or maybe split on whitespace
        # e.g. the first "word" might be the model's numeric guess
        first_token = model_answer_part.split()[0] if model_answer_part.split() else model_answer_part
        predicted_answer = first_token.strip()

        # Compare to ground truth `a`
        if predicted_answer == a:
            num_correct += 1
        else:
            num_wrong += 1
            incorrect_records.append((q, a, predicted_answer))

# -----------------------------------------------------------------------------
# Print final results
# -----------------------------------------------------------------------------
total = len(pairs)
print("=============================================")
print(f"Done! Evaluated {total} questions.")
print(f"Correct: {num_correct}, Wrong: {num_wrong}")
print(f"Accuracy: {num_correct/total:.2%}")

if len(incorrect_records) > 0:
    print("\nSome incorrect predictions:\n")
    for q, correct_a, model_a in incorrect_records[:10]:  # just show up to 10
        print(f"Q: {q}")
        print(f"GT: {correct_a}, Model: {model_a}")
        print("-----")

    # If you want to log all incorrect to a file
    with open(log_incorrect_file, 'w', encoding='utf-8') as f_out:
        for q, correct_a, model_a in incorrect_records:
            f_out.write(f"Q: {q}\n")
            f_out.write(f"GT: {correct_a}\n")
            f_out.write(f"Model: {model_a}\n")
            f_out.write("---\n")
    print(f"\nAll incorrect predictions written to {log_incorrect_file}")
print("=============================================")
