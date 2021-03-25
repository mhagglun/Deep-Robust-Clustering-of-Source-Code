import csv
import json
import torch
import config
import pickle
import linecache
import pandas as pd
from tqdm import tqdm
from typing import Iterable
from collections import deque



class Vocabulary(object):
    def __init__(self, words: Iterable[str]):
        self.word2idx = {word: i for i, word in enumerate(['<unk>', '<pad>', *words])}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
    
    def lookup_idx(self, word):
        if word not in self.word2idx:
            return 0
        else:
            return self.word2idx[word]

    def lookup_word(self, idx):
        if idx not in self.idx2word:
            return '<unk>'
        else:
            return self.idx2word[idx]

    def __len__(self):
        return len(self.word2idx)

def load_vocabularies(vocab_path):
    with open(vocab_path, 'rb') as f:
        word2count = pickle.load(f)
        path2count = pickle.load(f)
        label2count = pickle.load(f)
    word_vocab = Vocabulary(word2count.keys())
    path_vocab = Vocabulary(path2count.keys())
    label_vocab = Vocabulary(label2count.keys())
    return word_vocab, path_vocab, label_vocab


class Code2VecDataset(torch.utils.data.Dataset):
    def __init__(self, filename: str, word_vocab: Vocabulary, path_vocab: Vocabulary, label_vocab: Vocabulary):
        self.dataset = pd.read_csv(filename, sep=' ', header=None).fillna('1,1,1')
        self.word_vocab = word_vocab
        self.path_vocab = path_vocab
        self.label_vocab = label_vocab
    
    def __len__(self):
        return len(self.dataset)

    def __tensorize(self, sample):
        x_s = torch.zeros((config.MAX_LENGTH)).long()
        path = torch.zeros((config.MAX_LENGTH)).long()
        x_t = torch.zeros((config.MAX_LENGTH)).long()

        method_name, ctxs = sample

        label = self.label_vocab.lookup_idx(method_name)
        tmp_x_s, tmp_path, tmp_x_t = zip(*[(
            self.word_vocab.lookup_idx(ctx[0]),
            self.path_vocab.lookup_idx(ctx[1]),
            self.word_vocab.lookup_idx(ctx[2])
        ) for ctx in ctxs])
        x_s, path, x_t = torch.LongTensor(tmp_x_s), torch.LongTensor(tmp_path), torch.LongTensor(tmp_x_t)
        return label, x_s, path, x_t

    def __getitem__(self, index):
        source = self.dataset.iloc[index].values
        
        label1 = source[0]
        pathcontexts_1 = source[1:]
        
        label1, x_s1, path1, x_t1 = self.__tensorize((label1, pathcontexts_1))

        return label1, x_s1, path1, x_t1

