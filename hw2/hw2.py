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

import random

random.seed(1111)
torch.manual_seed(1111)
torch.cuda.manual_seed_all(1111)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=-1)

    parser.add_argument("--model", choices=["NnLm", "LstmLm"], default="LstmLm")
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


TEXT = torchtext.data.Field()
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path="data",
    train="train.txt", validation="valid.txt", test="test.txt", text_field=TEXT)

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val), batch_size=args.bsz, device=args.devid, bptt_len=args.bptt, repeat=False)

test_iter = torchtext.data.BPTTIterator.splits(
    (test), batch_size=args.bsz, device=args.devid, bptt_len=32, repeat=False)

class Lm(nn.Module):
    def __init__(self):
        super(Lm, self).__init__()

class NnLm(Lm):
    def __init__(self):
        super(NnLm, self).__init__()

class LstmLm(Lm):
    def __init__(self):
        super(LstmLm, self).__init__()





