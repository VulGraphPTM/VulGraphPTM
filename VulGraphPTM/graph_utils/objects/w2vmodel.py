import pickle
import os
import json
import nltk
from typing import List
from os import listdir
from os.path import join, isfile, exists
from .cpg import Node
from .embed_utils import extract_tokens_nltk
from gensim.models import Word2Vec


def generate_w2vModel(
        data_paths: list[str],
        w2v_model_path: str,
        vector_dim=100,
        epochs=5,
        alpha=0.001,
        window=5,
        min_count=1,
        min_alpha=0.0001,
        sg=0,
        hs=0,
        negative=10
):
    print("Training w2v model...")
    sentences = []
    for path in data_paths:
        with open(path, 'r') as fp:
            samples = [json.loads(line) for line in fp.readlines()]
        for sam in samples:
            sentences.append(nltk.word_tokenize(sam['func']))
    print(len(sentences))
    model = Word2Vec(sentences=sentences, vector_size=vector_dim, alpha=alpha, window=window, min_count=min_count,
                     max_vocab_size=None, sample=0.001, seed=1, workers=8, min_alpha=min_alpha, sg=sg, hs=hs,
                     negative=negative)
    print('Embedding Size : ', model.vector_size)
    for _ in range(epochs):
        model.train(sentences, total_examples=len(sentences), epochs=1)
    model.save(w2v_model_path)
    return model


def load_w2vModel(w2v_path: str):
    model = Word2Vec.load(w2v_path)
    return model
