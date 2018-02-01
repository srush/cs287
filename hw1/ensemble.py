import argparse

import math

import torchtext
from torchtext.vocab import Vectors, GloVe

import torch
from torch.autograd import Variable as V
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm

from tqdm import tqdm

from IPython import embed
import numpy as np

torch.manual_seed(1111)
#torch.cuda.manual_seed_all(1111)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", choices=['NB', 'NB2', "LR", "LR2", "CBoW", "CNN", "CNNLOL", "CNNLSTM", "LSTM"])
    parser.add_argument("-gpu", type=int, default=-1)
    parser.add_argument("-lr", type=float, default=1)
    parser.add_argument("-lrd", type=float, default=0.9)
    parser.add_argument("-epochs", type=int, default=10)
    parser.add_argument("-dropout", type=float, default=0.5)
    parser.add_argument("-bsz", type=int, default=32)
    parser.add_argument("-mom", type=float, default=0.9)
    parser.add_argument("-dm", type=float, default=0)
    parser.add_argument("-wd", type=float, default=1e-4)
    parser.add_argument("-nonag", action="store_true", default=False)
    parser.add_argument("-binarize_nb", action="store_true")

    parser.add_argument("-clip", type=float, default=0.1)
    parser.add_argument("-optim", type=str, required=False, default="SGD")
    parser.add_argument("-output_file", type=str, default=None)
    parser.add_argument("-model_dir", type=str, required=False, default="model/")
    parser.add_argument("-model_name", type=str, required=False, default="tmp")
    parser.add_argument("-load_model", type=bool, required=False, default=False)
    args = parser.parse_args()
    return args

args = parse_args()



# Our input $x$
TEXT = torchtext.data.Field()

# Our labels $y$
LABEL = torchtext.data.Field(sequential=False)
train, valid, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

TEXT.build_vocab(train)
LABEL.build_vocab(train)
LABEL.vocab.itos = ['positive', 'negative']
LABEL.vocab.stoi['positive'] = 0
LABEL.vocab.stoi['negative'] = 1

print('len(TEXT.vocab)', len(TEXT.vocab))
print('len(LABEL.vocab)', len(LABEL.vocab))

labels = [ex.label for ex in train.examples]

train_iter, _, _ = torchtext.data.BucketIterator.splits(
    (train, valid, test), batch_size=args.bsz, device=args.gpu, repeat=False)

_, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, valid, test), batch_size=10, device=args.gpu)


# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
simple_vec = TEXT.vocab.vectors.clone()

url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.en.vec', url=url))
complex_vec = TEXT.vocab.vectors

loss = nn.BCEWithLogitsLoss()


def output_test_nb(model):
    "All models should be able to be run with following command."
    upload = []
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.text)
        _, argmax = probs.max(1)
        upload += list(argmax.data)

    with open(args.output_file, "w") as f:
        f.write("Id,Cat\n")
        for i,u in enumerate(upload):
            f.write(str(i) + "," + str(u+1) + "\n")

def output_test(model):
    "All models should be able to be run with following command."
    upload = []
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        yhat = F.sigmoid(model(batch.text)) > 0.5
        upload += yhat.tolist()

    with open(args.output_file, "w") as f:
        f.write("Id,Cat\n")
        for i,u in enumerate(upload):
            f.write(str(i) + "," + str(u+1) + "\n")

# TODO: fix this; forward no longer returns # of correct example
def validate_nb(model, valid):
    correct = 0.
    total = 0.
    ugh = iter(valid)
    for i in tqdm(range(len(valid))):
        batch = next(ugh)
        x = batch.text
        y = batch.label
        correct += model(x, y)
        total += y.size(0)
    return correct , total, correct / total

def train_nb(nb):
    ugh = iter(train_iter)
    for i in range(len(train_iter)):
        batch = next(ugh)
        x = batch.text
        y = batch.label
        nb.update_counts(x, y)
    if args.model == "NB2":
        nb.get_probs()
    return nb
    

def validate(model, valid):
    model.eval()
    correct = 0.
    total = 0.
    if True:
    #with torch.no_grad():
        for batch in valid:
            x = batch.text
            y = batch.label
            yhat = F.sigmoid(model(x)) > 0.5
            results = yhat.long() == y
            correct += results.float().sum().data[0]
            total += results.size(0)
    return correct, total, correct / total

class NB2(nn.Module):
    # See http://www.aclweb.org/anthology/P/P12/P12-2.pdf#page=118
    def __init__(self, vocab, dropout, alpha=1.):
        super(NB2, self).__init__()
        self.vsize = len(vocab.itos)
        self.nclass = 2
        self.alpha = alpha
        self.ys = list(range(self.nclass))
        self.p = torch.zeros((self.vsize))
        self.q = torch.zeros((self.vsize))
        self.N = torch.zeros((self.nclass))

        self.w = torch.zeros((self.vsize))

    def transform(self, x):
        length, batch_size = x.size()
        f = torch.zeros((self.vsize, batch_size))
        for i in range(batch_size):  # TODO: make it more efficient
            f[:,i][x[:,i]] += 1  
        f = (f > 0).float()
        return f

    def get_probs(self):
        self.p += self.alpha
        self.q += self.alpha
        p_normed = self.p / torch.norm(self.p, p=1)
        q_normed = self.q / torch.norm(self.q, p=1)
        self.w = torch.log(p_normed / q_normed)
        self.b = np.log(self.N[1] / self.N[0])

    def _score(self, x, y):
        length, batch_size = x.size()
        f = self.transform(x)
        predict = torch.sign(torch.matmul(self.w, f) + self.b)
        predict = (predict > 0).long()
        #print("in score")
        #embed()
        correct = torch.sum(predict == y)

        return correct
 
    def get_score(self, x, y):   # of correct examples,
        x = x.data
        y = y.data
        return self._score(x, y)
    
    def forward(self, x):
        x = x.data
        length, batch_size = x.size()
        f = self.transform(x)
        predict = torch.sign(torch.matmul(self.w, f) + self.b)
        predict = (predict > 0).long()
        return predict

    def update_counts(self, x, y):
        xd = x.data  # (length, batch_size)
        yd = y.data.ne(0) # (batch_size)
        length, batch_size = xd.size()
        f = self.transform(xd)
        for i in range(batch_size):
            if yd[i] == 1:
                self.p += f[:,i]
                self.N[1] += 1
            else:
                self.q += f[:,i]
                self.N[0] += 1


class LR2(nn.Module):
    def __init__(self, vocab, dropout, binarize=True):
        super(LR2, self).__init__()
        self.vsize = len(vocab.itos)
        # Since this is binary LRRwe can just use BCEWithLogitsLoss
        self.lut = nn.Linear(self.vsize, 1)
        self.bias = Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.binarize = binarize
    
    def transform(self, x):
        length, batch_size = x.size()
        f = torch.zeros((self.vsize, batch_size))
        for i in range(batch_size):  # TODO: make it more efficient
            f[:,i][x[:,i].data] += 1  
        f = (f > 0).float()
        return V(f)

    def forward(self, input):
        # input will be seqlen x bsz, so we want to use the weights from the lut
        # and sum them up to get the logits.
        x = self.transform(input)  # vsize, batch_size
        x = torch.transpose(x, 0, 1) # batch_size, vsize
        if self.binarize:
            x = (x > 0).float()
        if self.dropout is not None:
            x = self.dropout(x)
        output = self.lut(x).view(input.size(1))
        return output + self.bias


def validate_ensemble(models, data_iter):
    for model in models:
        model.eval()
    correct = 0.
    total = 0.
    if True:  # HACK
    #with torch.no_grad():
        for batch in data_iter:
            x = batch.text
            y = batch.label

            batch_size = y.size(0)

            yhats = torch.zeros((len(models), batch_size)).long()
            for i, model in enumerate(models):
                if i != 0:
                    yhat = (F.sigmoid(model(x)) > 0.5).long().data
                else:  # NB: assume models[0] = NB2
                    yhat = 1 - model(x).long()  # NB: for NB2, labels are opposite
                yhats[i] = yhat
            #embed()
            agg_yhat = torch.sum(yhats, dim=0)
            agg_yhat = agg_yhat >= (len(models) / 2)
            results = V(agg_yhat).long() == y
            correct += results.float().sum().data[0]
            total += results.size(0)
    return correct, total, correct / total


def load_model(model, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])


models = []
# store all model in model/final/

# NB2
model_nb2 = NB2(TEXT.vocab, len(LABEL.vocab.itos))
checkpoint = torch.load("model/final/NB2")
model_nb2.w = checkpoint["w"]
model_nb2.b = checkpoint["b"]

models.append(model_nb2)

# LR
model_lr2 = LR2(TEXT.vocab, args.dropout)
load_model(model_lr2, "model/final/LR2")
models.append(model_lr2)

# CNN
# TODO: add CNN, CNNLOL, other versions of LR, CBOW, NB

# train_acc = validate_ensemble(models, train_iter) 
# print("Train ACC: %.3lf" % train_acc)
valid_acc = validate_ensemble(models, valid_iter)
print("Valid ACC:", valid_acc)
test_acc = validate_ensemble(models, test_iter)
print("Test ACC:" , test_acc)



