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

NBWEIGHT = 1
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
    parser.add_argument("-output_file", type=str, default="ensemble.out")
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

    def output_dict(self):
        f = open("NB.dict","w")
        f.write("======== NB2 ========\n")
        for i in range(self.vsize):
            try:
                f.write("NBword(%s): %.3lf\n" % (TEXT.vocab.itos[i], self.w[i]))
            except:
                f.write("skip - unicode error\n")
        f.write("======== NB2 ======== \n")
        f.close()

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
     
    def output_dict(self):
        f = open("LR.dict","w")
        f.write("======== LR2 ======== \n")
        for i in range(self.vsize):
            try:
                f.write("LRword(%s): %.3lf\n" % (TEXT.vocab.itos[i], self.lut.weight[0][i].data[0]))
            except:
                f.write("skip - unicode error\n")
        f.write("======== LR2 ======== \n")
        f.close()
   
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

class CBoW(nn.Module):
    def __init__(self, vocab, dropout=0, binarize=True, average=False):
        super(CBoW, self).__init__()
        self.vsize = len(vocab.itos)
        self.binarize = binarize
        self.average = average
        self.static_lut = nn.Embedding(self.vsize, vocab.vectors.size(1))
        self.static_lut.weight.data.copy_(vocab.vectors)
        self.static_lut.requires_grad = False
        self.lut = nn.Embedding(self.vsize, vocab.vectors.size(1))
        self.lut.weight.data.copy_(vocab.vectors)
        self.lut.weight.data += self.lut.weight.data.new(self.lut.weight.data.size()).normal_(0, 0.01)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.proj = nn.Sequential(
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 1)
        )

    def forward(self, input):
        if self.average:
            hidden = torch.cat([self.lut(input).sum(0), self.static_lut(input).mean(0)], -1)
        else:
            hidden = torch.cat([self.lut(input).sum(0), self.static_lut(input).sum(0)], -1)
        if self.dropout:
            hidden = self.dropout(hidden)
        return self.proj(hidden).squeeze(-1)

class CNNLOL(nn.Module):
    def __init__(self, vocab, dropout=0.):
        super(CNNLOL, self).__init__()
        self.vsize = len(vocab.itos)
        self.static_lut = nn.Embedding(self.vsize, vocab.vectors.size(1))
        self.static_lut.weight.data.copy_(vocab.vectors)
        self.static_lut.requires_grad = False
        self.lut = nn.Embedding(self.vsize, vocab.vectors.size(1))
        self.lut.weight.data.copy_(simple_vec)
        #self.lut.requires_grad = False
        self.convs = nn.ModuleList([
            nn.Conv1d(600, 200, 3, padding=1),
            nn.Conv1d(600, 200, 5, padding=2),
        ])
        self.proj = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(500, 1)
        )

    def forward(self, input):
        word_rep = torch.cat([self.lut(input), self.static_lut(input)], -1).permute(1, 2, 0)
        cnns, _ = torch.cat([conv(word_rep) for conv in self.convs] + [word_rep], 1).max(2)
        return self.proj(cnns).squeeze(-1)

class LSTM(nn.Module):
    def __init__(self, vocab, dropout=0):
        super(LSTM, self).__init__()
        self.vsize = len(vocab.itos)
        self.dropout = dropout
        self.static_lut = nn.Embedding(self.vsize, vocab.vectors.size(1))
        self.static_lut.weight.data.copy_(vocab.vectors)
        self.static_lut.requires_grad = False
        self.lut = nn.Embedding(self.vsize, vocab.vectors.size(1))
        self.lut.weight.data.copy_(simple_vec)
        self.lut.requires_grad = False
        self.lstm = nn.LSTM(600, 300, 2, bidirectional=True, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            #nn.Linear(1200 * 2, 1)
            nn.Linear(2400, 1200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1200, 1)
        )

    def forward(self, input):
        word_rep = torch.cat([self.lut(input), self.static_lut(input)], -1)
        _, (h, c) = self.lstm(word_rep)
        z = torch.cat([h.permute(1,0,2), c.permute(1,0,2)], -1).view(input.size(1), -1)
        return self.proj(z).squeeze(-1)

def validate_ensemble(models, data_iter, out=True):
    for model in models:
        model.eval()
    correct = 0.
    total = 0.
    def output_text(xlist):
        word_list = [TEXT.vocab.itos[idx] for idx in xlist]
        sentence = " ".join(word_list)
        return sentence

    from sets import Set
    stop_words = []

    if True:  # HACK
    #with torch.no_grad():
        for batch in tqdm(data_iter):
            x = batch.text
            y = batch.label

            batch_size = y.size(0)

            yhats = torch.zeros((len(models), batch_size)).long()
            for i, model in enumerate(models):
                if i != 0:
                    yhat = (F.sigmoid(model(x)) > 0.5).long().data
                else:  # NB: assume models[0] = NB2
                    yhat = (1 - model(x).long()) * NBWEIGHT  # NB: for NB2, labels are opposite
                yhats[i] = yhat
            #embed()
            agg_yhat = torch.sum(yhats, dim=0)
            agg_yhat = agg_yhat >= ((len(models) + NBWEIGHT - 1) / 2)
            results = V(agg_yhat).long() == y
            if out:
                for i in tqdm(range(batch_size), desc="output_bad"):
                    #embed()
                    if agg_yhat[i] != y[i].data[0]:
                        xlist = x[:,i].data.tolist()
                        sent = output_text(xlist)
                        print("--------- BAD START --------")
                        print("sent: %s" % sent)
                        nb_values = []
                        lr_values = []
                        for v in xlist:
                            nb_values.append("%.3lf" % (-models[0].w[v])) 
                            lr_values.append("%.3lf" % models[1].lut.weight[0][v].data[0])
                        print("nb_values: %s" % " ".join(nb_values))
                        print("lr_values: %s" % " ".join(lr_values))

                        # argmin negative values -> positive (label=0)
                        # argmax positive values -> negative (label=1)
                        if y[i].data[0] == 0:
                            idx = np.argmax(nb_values)
                        else:
                            idx = np.argmin(nb_values)
                        #embed()
                        vid = xlist[idx]
                        stop_words.append(TEXT.vocab.itos[vid])
                        print("add %s | idx %d | vid %d" % (TEXT.vocab.itos[vid], idx, vid))
                        print("our: %d | true: %d" % (agg_yhat[i], y[i].data[0]))
                        print("all yhats: %s" % str(yhats[:,i]))
                        print("--------- BAD  END--------\n\n")

            correct += results.float().sum().data[0]
            total += results.size(0)
    print("list=", stop_words)
    newf = open("stop_words.greedy", "w")

    for word in stop_words:
        newf.write("%s\n" % word)
    newf.close()


    return correct, total, correct / total

def output_ensemble(models):
    "All models should be able to be run with following command."
    upload = []

    for model in models:
        model.eval()
    correct = 0.
    total = 0.
    if True:  # HACK
    #with torch.no_grad():
        for batch in test_iter:
            x = batch.text
            y = batch.label

            #print("x=%s", str(x))
            batch_size = y.size(0)

            yhats = torch.zeros((len(models), batch_size)).long()
            for i, model in enumerate(models):
                if i != 0:
                    yhat = (F.sigmoid(model(x)) > 0.5).long().data
                else:  # NB: assume models[0] = NB2
                    yhat = (1 - model(x).long()) * NBWEIGHT  # NB: for NB2, labels are opposite
                yhats[i] = yhat
            #embed()
            agg_yhat = torch.sum(yhats, dim=0)
            agg_yhat = agg_yhat >= ((len(models) + NBWEIGHT - 1) / 2)
            upload += agg_yhat.tolist()

    with open(args.output_file, "w") as f:
        f.write("Id,Cat\n")
        for i,u in enumerate(upload):
            f.write(str(i) + "," + str(u+1) + "\n")

def load_model(model, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    print("Model Loaded from %s" % filename)


models = []
# store all model in model/final/

# NB2
model_nb2 = NB2(TEXT.vocab, len(LABEL.vocab.itos))
checkpoint = torch.load("model/final/NB2")
model_nb2.w = checkpoint["w"]
model_nb2.b = checkpoint["b"]
#model_nb2.output_dict()
models.append(model_nb2)

# LR
model_lr2 = LR2(TEXT.vocab, args.dropout)
load_model(model_lr2, "model/final/LR2")
#model_lr2.output_dict()

models.append(model_lr2)


# CBOW
model_cbow = CBoW(TEXT.vocab)
checkpoint = torch.load("model/final/CBoW")
model_cbow.load_state_dict(checkpoint)
models.append(model_cbow)



# CNNLOL
model_cnnlol = CNNLOL(TEXT.vocab)
print("start loading")
checkpoint = torch.load("model/final/CNNLOL")
model_cnnlol.load_state_dict(checkpoint)
print("finish loading cnnlol")
models.append(model_cnnlol)

# LSTM
model_lstm = LSTM(TEXT.vocab)
checkpoint = torch.load("model/final/LSTM")
model_lstm.load_state_dict(checkpoint)
models.append(model_lstm)

train_acc = validate_ensemble(models, train_iter, out=True) 
print("Train ACC:", train_acc)
#valid_acc = validate_ensemble(models, valid_iter, out=True)
#print("Valid ACC:", valid_acc)
test_acc = validate_ensemble(models, test_iter, out=False)
print("Test ACC:" , test_acc)



output_ensemble(models)


