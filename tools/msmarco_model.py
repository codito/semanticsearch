#!/usr/bin/env python

from transformers.convert_graph_to_onnx import convert
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from os.path import basename, exists
import torch


model_name = "sentence-transformers/msmarco-distilbert-base-v3"


def convert_to_onnx():
    # Setup the msmarco-distilbert-base-v2 model from sentence_transformers
    # https://www.sbert.net/docs/pretrained_models.html#question-answer-retrieval-msmarco
    out = Path("src/models/{}.onnx".format(basename(model_name)))
    if exists(out):
        print("Model exists at {}".format(out))
        return
    convert(framework="pt",
            model=model_name,
            output=out, opset=11)


def test_model():
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1)\
                                            .expand(token_embeddings.size())\
                                            .float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # Load AutoModel from huggingface model repository
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def compute_embeddings(sentences):
        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True,
                                  return_tensors='pt')

        # Compute query embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        return mean_pooling(model_output, encoded_input['attention_mask'])

    # Queries we want embeddings for
    queries = ['What is the capital of France?',
               'How many people live in New York City?']

    # Passages that provide answers
    passages = ['Paris is the capital of France',
                'New York City is the most populous city in the United States, with an estimated 8,336,817 people living in the city, according to U.S. Census estimates dating July 1, 2019']

    passages = ["What is the capital of France?",
                "How many people live in New York City?"]
    queries = ["Paris is the capital city of France."]

    query_embeddings = compute_embeddings(queries)
    passage_embeddings = compute_embeddings(passages)

    similarity = cosine_similarity(passage_embeddings, query_embeddings)
    values, indexes = similarity.topk(2)
    for k, v in zip(values, indexes):
        print(k, v)


def cosine_similarity(corpus, query):
    corpus_norm = corpus / corpus.norm(1).unsqueeze(-1)
    query_norm = query / query.norm(1).unsqueeze(-1)

    return torch.mm(query_norm, corpus_norm.transpose(0, 1))


if __name__ == "__main__":
    # convert_to_onnx()
    test_model()
