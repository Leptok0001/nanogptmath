#prepare.py in math_char

import os
import pickle
import numpy as np

CHUNK_SIZE = 9

# Path to the problems.txt file
input_file_path = os.path.join(os.path.dirname(__file__), 'problems.txt')

with open(input_file_path, 'r', encoding='utf-8') as f:
    lines = f.read().strip().split('\n')

pairs = []
for i in range(0, len(lines), 2):
    q = lines[i]
    a = lines[i+1] if i+1 < len(lines) else ''

    # Combine question + answer
    pair = q + a

    # Pad the pair up to CHUNK_SIZE
    if len(pair) < CHUNK_SIZE:
        pair += ' ' * (CHUNK_SIZE - len(pair))
    elif len(pair) > CHUNK_SIZE:
        # Truncate or handle longer pairs here if necessary
        pair = pair[:CHUNK_SIZE]

    pairs.append(pair)

# Split these pairs into train and validation
n_pairs = len(pairs)
split_index = int(n_pairs * 0.9)
train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]

# Concatenate all pairs into a single string for train and for val
train_data = ''.join(train_pairs)
val_data = ''.join(val_pairs)

print(train_data)
print(f"Number of q/a pairs: {n_pairs}")
print(f"Training pairs: {len(train_pairs)}")
print(f"Validation pairs: {len(val_pairs)}")

# Get all unique characters from the entire dataset (train + val)
full_data = train_data + val_data
chars = sorted(list(set(full_data)))
vocab_size = len(chars)
print("All unique characters:", ''.join(chars))
print(f"Vocab size: {vocab_size}")

# Create mappings from characters to integers (stoi) and vice versa (itos)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Encode train and val data
train_ids = encode(train_data)
val_ids = encode(val_data)

print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# Convert to numpy arrays with dtype=np.uint16
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

# Write out to train.bin and val.bin
train_bin_path = os.path.join(os.path.dirname(__file__), 'train.bin')
val_bin_path = os.path.join(os.path.dirname(__file__), 'val.bin')

train_ids.tofile(train_bin_path)
val_ids.tofile(val_bin_path)

# Save meta information (vocab, itos, stoi) to meta.pkl
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Data preparation complete.")
