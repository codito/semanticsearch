#!/usr/bin/env python

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def print_token(sequence):
    inputs = tokenizer(sequence)
    z = zip(tokenizer.tokenize(sequence), inputs["input_ids"][1:-1],
            inputs["attention_mask"][1:-1])
    print(sequence)
    print([k for k in z])
    print("---")

print_token("A Titan RTX has 24GB of VRAM")
print_token("This is a short sequence.")
print_token("This is a rather long sequence. It is at least longer than the sequence A.")
