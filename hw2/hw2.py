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

DEBUG = False
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

#train_iter_ngram, valid_iter_ngram, test_iter_ngram = torchtext.data.BucketIterator.splits(
#    (train, valid, test), batch_size=args.bsz, device=args.devid, repeat=False)

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

        self.unigram_A = {}
        self.bigram_A = {}
        self.trigram_A = {}

        self.unigram_B = 0
        self.bigram_B = {}
        self.trigram_B = {}

        # A / B
        self.unigram_probs = {}
        self.bigram_probs = {}
        self.trigram_probs = {}

        # TODO: better implementation - less memory

        self.alphas = [0.2, 0.3, 0.5]
        assert np.sum(self.alphas) == 1.0
        self.a = 1 # smoothing factor
        self.V = len(TEXT.vocab)

    def add_dict(self, my_dict, ind, delta):
        if not ind in my_dict:
            my_dict[ind] = 0
        my_dict[ind] += delta

    def train_batch(self, batch):
        batch = batch.text.data
        length, batch_size = batch.size()
        #embed()
        
        for idx in range(batch_size):
            for p in range(2, length):
                w1, w2, w3 = batch[p-2, idx], batch[p-1, idx], batch[p, idx]

                # update unigram
                if w3 not in self.unigram_A:
                    self.unigram_A[w3] = 0
                self.unigram_A[w3] += 1
                self.unigram_B += 1

                # update bigram
                if w2 not in self.bigram_A:
                    self.bigram_A[w2] = {}
                    self.bigram_B[w2] = 0

                if w3 not in self.bigram_A[w2]:
                    self.bigram_A[w2][w3] = 0

                self.bigram_A[w2][w3] += 1
                self.bigram_B[w2] += 1

                # update trigram
                if (w1, w2) not in self.trigram_A:
                    self.trigram_A[(w1, w2)] = {}
                    self.trigram_B[(w1, w2)] = 0
                if w3 not in self.trigram_A[(w1, w2)]:
                    self.trigram_A[(w1, w2)][w3] = 0
                self.trigram_A[(w1, w2)][w3] += 1
                self.trigram_B[(w1, w2)] += 1
        

    def train(self, data_iter):
        for batch in tqdm(train_iter):
            self.train_batch(batch)

        """
        # get unigram probs
        for w3 in self.unigram_A:
            self.unigram_probs[w3] = (1.0 * self.unigram_A[w3] + self.a) / (1.0 * self.unigram_B + self.a * self.V)
        # get bigram probs
        for w2 in self.bigram_A:
            for w3 in self.bigram_A[w2]:
                self.bigram_probs[(w2, w3)] = (1.0 * self.bigram_A[w2][w3] + self.a) / (1.0 * self.bigram_B[w2] + self.a * self.V)
        # get trigram probs
        for (w1, w2) in self.trigram_A:
            for w3 in self.trigram_A[(w1, w2)]:
                self.trigram_probs[(w1, w2, w3)] = (1.0 * self.trigram_A[(w1,w2)][w3] + self.a) / (1.0 * self.trigram_B[(w1,w2)] + self.a * self.V)
        """
        if DEBUG:
            for w2 in self.bigram_A:
                for w3 in self.bigram_A[w2]:
                    def getword(i):
                        return TEXT.vocab.itos[i]
                    print("bigramA [%s, %s]: %d" % (getword(w2), getword(w3),  self.bigram_A[w2][w3]))
            for (w1, w2) in self.trigram_A:
                for w3 in self.trigram_A[(w1, w2)]:
                    def getword(i):
                        return TEXT.vocab.itos[i]
                    print("trigramA [%s, %s, %s]: %d" % (getword(w1), getword(w2), getword(w3), self.trigram_A[(w1,w2)][w3]))
            embed()
        # do we want to ignore <eos> in unigram model?

    def calc_prob(self, words, debug=False):
        w1, w2, w3 = words

        word_prob = []
        word_prob.append(self.alphas[0] * (1.0 * self.unigram_A.get(w3, 0) + self.a) / (self.unigram_B + self.a * self.V))
        word_prob.append(self.alphas[1] * (1.0 * self.bigram_A.get(w2, {}).get(w3, 0) + self.a) / (1.0 * self.bigram_B.get(w2, 0) + self.a * self.V))
        word_prob.append(self.alphas[2] * (1.0 * self.trigram_probs.get((w1, w2), {}).get(w3, 0) + self.a) / (1.0 * self.trigram_B.get((w1,w2), 0) + self.a * self.V))
        return np.sum(word_prob)
        
        if debug:
            embed()
        return word_prob


    def validate_batch(self, batch):
        batch = batch.text.data
        length, batch_size = batch.size()

        avg_nll = 0
        for idx in range(batch_size):

            nll = 0
            if DEBUG:
                w = [TEXT.vocab.itos[batch[i,idx]] for i in range(length)]
                print("now %s" % (" ".join(w)))

            for p in range(2, length):
                words = [batch[p-2+i,idx] for i in range(3)]
                word_prob = self.calc_prob(words, bool(p <= 4) and DEBUG)
                nll += -np.log(word_prob)
            nll /= (length - 2.0)
            avg_nll += nll

        return avg_nll / batch_size

    """ Calculate average perplexity """
    def validate(self, data_iter):
        nll = 0
        for batch in tqdm(data_iter, desc="validate"):
            nll += self.validate_batch(batch)
        return np.exp([nll])[0]

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



