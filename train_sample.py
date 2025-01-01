# train_sample.py

import random
import torch

from model import GPTConfig, GPT  # in case you need them

def get_problem(max_digits=1, ops=['+','-']):
    """
    Generates a random arithmetic problem and its answer.
    Example: "12+7=" and "19"
    """
    num1_digits = random.randint(1, max_digits)
    num2_digits = random.randint(1, max_digits)
    num1 = random.randint(10**(num1_digits - 1), 10**num1_digits - 1)
    num2 = random.randint(10**(num2_digits - 1), 10**num2_digits - 1)
    op = random.choice(ops)
    if op == '+':
        answer_val = num1 + num2
    elif op == '-':
        answer_val = num1 - num2
    else:  # op == '*'
        answer_val = num1 * num2

    problem = f"{num1}{op}{num2}="
    answer = str(answer_val)
    return problem, answer

def do_sample(
    model,
    encode,
    decode,
    prompt,
    max_new_tokens=3,
    temperature=0.95,
    top_k=20
):
    """
    Sample from `prompt` using the *in-memory* `model`.
    """
    device = next(model.parameters()).device  # the current device of the model
    was_training = model.training
    model.eval()  # ensure model is in eval mode for generation

    # Encode the prompt into token IDs
    x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    # Optionally restore the training state
    if was_training:
        model.train()

    # Decode tokens to a string
    out_tokens = y[0].tolist()
    return decode(out_tokens)
