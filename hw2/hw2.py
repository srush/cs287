import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable as V
from torch.nn import Parameter
from torch.nn.utils import clip_grad_norm

import torchtext
from torchtext.vocab import Vectors, GloVe

from tqdm import tqdm
import numpy as np
import random
from IPython import embed

TOP_K = 20
random.seed(1111)
torch.manual_seed(1111)
#torch.cuda.manual_seed_all(1111)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=-1)

    parser.add_argument("--model", choices=["Ngram", "NnLm", "LstmLm"], default="LstmLm")
    parser.add_argument("--nhid", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=1)

    parser.add_argument("--tieweights", action="store_true")

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--optim", choices=["SGD", "Adam"], default="Adam")

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lrd", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--bptt", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--clip", type=float, default=5)

    # Adam parameters
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    # SGD parameters
    parser.add_argument("--mom", type=float, default=0.99)
    parser.add_argument("--dm", type=float, default=0)
    parser.add_argument("--nonag", action="store_true", default=False)

    # Execution options
    parser.add_argument("--evaluatemodel", type=str, default=None)
    parser.add_argument("--savemodel", action="store_true")

    return parser.parse_args()

args = parse_args()

# Maybe we should subclass LanguageModelingDataset?
TEXT = torchtext.data.Field()
train, valid, test = torchtext.datasets.LanguageModelingDataset.splits(
    path="data/",
    train="train.txt", validation="valid.txt", test="test.txt", text_field=TEXT)

TEXT.build_vocab(train)

train_iter, valid_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, valid, test), batch_size=args.bsz, device=args.devid, bptt_len=args.bptt, repeat=False)

class Lm(nn.Module):
    def __init__(self):
        super(Lm, self).__init__()

    def train_epoch(self):
        raise NotImplementedError("Implement train_epoch")

    def validate(self):
        raise NotImplementedError("Implement validate")

    def generate_predictions(self):
        raise NotImplementedError("Implement generate_predictions")

class NnLm(Lm):
    def __init__(self):
        super(NnLm, self).__init__()

class LstmLm(Lm):
    def __init__(self):
        super(LstmLm, self).__init__()

# Linear Interpolation on Unigram, Bigram and Trigram
class Ngram(nn.Module):
    def __init__(self):
        super(Ngram, self).__init__()
        self.total_counts = 0
        self.counts = [{}, {}, {}, {}]
        self.alphas = [0.2, 0.3, 0.5]
        assert np.sum(self.alphas) == 1.0
        self.a = 1 # smoothing factor
        self.V = len(TEXT.vocab)
        self.counts[0][""] = 0

    def add_dict(self, my_dict, ind, delta):
        if not ind in my_dict:
            my_dict[ind] = 0
        my_dict[ind] += delta

    def train_batch(self, batch):
        batch = batch.text.data
        length, batch_size = batch.size()
        for idx in range(batch_size):
            for p in range(length):
                self.counts[0][""] += 1
                # update unigram
                self.add_dict(self.counts[1], batch[p, idx], 1)
                # update bigram
                if p >= 1:
                    self.add_dict(self.counts[2], (batch[p-1, idx], batch[p, idx]), 1)
                # update trigram
                if p >= 2:
                    self.add_dict(self.counts[3], (batch[p-2, idx], batch[p-1, idx], batch[p, idx]), 1)

    def train(self, data_iter):
        for batch in tqdm(train_iter):
            self.train_batch(batch)

        # do we want to ignore <eos> in unigram model?

    def calc_prob(self, words):
        ngrams = ["", words[-1], (words[-2], words[-1]), (words[-3], words[-2], words[-1])]
        # NB: let's just do add-one smoothing for now
        probs = [(1.0 * self.counts[i+1].get(ngrams[i+1], 0) + self.a) / \
                 (1.0 * self.counts[i].get(ngrams[i], 0) + self.V * self.a) \
                 for i in range(3)]
        word_prob = 0
        for i in range(3):
            word_prob += self.alphas[i] * probs[i]
        return word_prob

    def validate_batch(self, batch):
        batch = batch.text.data
        length, batch_size = batch.size()

        avg_nll = 0
        for idx in range(batch_size):
            nll = 0
            for p in range(2, length):
                words = [TEXT.vocab.stoi[batch[p-2+i,idx]] for i in range(3)]
                word_prob = self.calc_prob(words)
                nll += -np.log(word_prob)
                embed()
            nll /= (length - 2.0)
            avg_nll += nll

        return avg_nll / batch_size

    """ Calculate average perplexity """
    def validate(self, data_iter):
        nll = 0
        for batch in tqdm(data_iter, desc="validate"):
            nll += self.validate_batch(batch)
        return np.exp2([nll])[0]

    def generate(self, prev2word, prev1word):  # last two words; return a list of 20 candidates
        prev2word = TEXT.vocab.stoi[prev2word]
        prev1word = TEXT.vocab.stoi[prev1word]
        word_probs = [self.calc_prob([prev2word, prev1word, i]) for i in range(self.V)]
        best_word_ids = np.argsort(word_probs)[::-1][:TOP_K]
        best_words = [TEXT.vocab.itos[word_id] for word_id in best_word_ids]
        return best_words

    def generate_predictions(self, input_file="data/input.txt", output_file="output.txt"):
        f = open(input_file)
        lines = f.readlines()
        f.close()

        f = open(output_file, "w")
        f.write("id,word\n")
        for i, sent in tqdm(enumerate(lines), desc="generate"):
            words = sent.split(" ")
            predict_words = self.generate(words[-3], words[-2])
            f.write("%d,%s\n" % (i + 1, " ".join(predict_words)))
        f.close()

def ngram_model(args):
    model = Ngram()
    model.train(train_iter)

    train_perp = model.validate(train_iter)
    valid_perp = model.validate(valid_iter)
    print("Train Perplexity: %.3lf\nValidation Perplexity: %.3lf\n" % (train_perp, valid_perp))

    model.generate_predictions()
    print("See Generated output in output.txt\n")

if __name__ == "__main__":
    if args.model == "Ngram":
        ngram_model(args)



