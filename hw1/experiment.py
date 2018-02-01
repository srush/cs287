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

"""
batch = next(iter(train_iter))
print("Size of text batch [max sent length, batch size]", batch.text.size())
print("Second in batch", batch.text[:, 0])
print("Converted back to string: ", " ".join([TEXT.vocab.itos[i] for i in batch.text[:, 0].data]))
print("Size of label batch [batch size]", batch.label.size())
print("Second in batch", batch.label[0])
print("Converted back to string: ", LABEL.vocab.itos[batch.label.data[0]])
"""

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

def validate_nb(model, valid):
    correct = 0.
    total = 0.
    for batch in valid:
        x = batch.text
        y = batch.label
        yhat = model(x)
        _, argmax = yhat.max(1)
        if args.model == "NB2":
            results = torch.autograd.Variable(argmax) == y
            correct += float(results.sum().data[0])
        else:
            results = argmax == y
            correct += float(results.sum())

        total += results.size(0)
    return correct , total, correct / total

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

# Models
class NB(nn.Module):
    # See http://www.aclweb.org/anthology/P/P12/P12-2.pdf#page=118
    def __init__(self, vocab, dropout, binarize=True, alpha=1):
        super(NB, self).__init__()
        self.vsize = len(vocab.itos)
        self.nclass = 2
        self.binarize = binarize
        self.ys = list(range(self.nclass))
        self.xycounts = Parameter(torch.FloatTensor(len(vocab.itos), self.nclass).fill_(alpha))
        self.ycounts = Parameter(torch.FloatTensor(self.nclass).fill_(0))

    def _score(self, x, y):
        if self.binarize:
            score = V(torch.FloatTensor(x.size(1)).fill_(0))
            for j in range(x.size(1)):
                tokens = torch.LongTensor(list(set(x[:,j].tolist())))
                score[j] += (self.xycounts[tokens][:, y].log() - self.xycounts[:, y].sum().log()).sum()
            return score + self.ycounts[y].log()
        else:
            return (self.xycounts[x, y].log() - self.xycounts[:, y].sum().log()).sum(0) \
                + self.ycounts[y].log()# - self.ycounts.sum().log()

    def forward(self, input):
        # p(y|x) = p(y) \prod_i p(x|y) / p(x)
        x = input.data
        scores = torch.cat([self._score(x, y).view(-1, 1) for y in self.ys], 1)
        return scores

    def update_counts(self, x, y):
        xd = x.data
        yd = y.data
        if self.binarize:
            for j in range(yd.size(0)):
                tokens = set(xd[:,j].tolist())
                label = yd[j]
                for token in tokens:
                    self.xycounts[token, label] += 1
                self.ycounts[label] += 1
        else:
            idxs = xd * 2
            idxs[yd.view(1,-1).ne(0).expand_as(xd)] += 1
            self.xycounts.data.put_(idxs, torch.ones_like(xd).float(), accumulate=True)
            self.ycounts.data.put_(yd, torch.ones_like(yd).float(), accumulate=True)

class NB2(nn.Module):
    # See http://www.aclweb.org/anthology/P/P12/P12-2.pdf#page=118
    def __init__(self, vocab, dropout, alpha=1):
        super(NB2, self).__init__()
        self.vsize = len(vocab.itos)
        self.nclass = 2
        self.alpha = alpha
        self.ys = list(range(self.nclass))
        self.xycounts = torch.zeros(len(vocab.itos), self.nclass) #.fill_(alpha)
        self.ycounts = torch.zeros(self.nclass) #.fill_(alpha)

    def get_probs(self):
        # NB: now, only added alpha for xyprob
        self.yprob = torch.log(self.ycounts.clone()) - np.log(torch.sum(self.ycounts))
        #self.xyprob = torch.log((self.xycounts.clone() + self.alpha)) - torch.log(self.ycounts.view(1,-1).expand_as(self.xycounts) + self.vsize * self.alpha)
        self.xyprob = torch.log((self.xycounts.clone() + self.alpha)) 
    
    def _score(self, x, y):
        length, batch_size = x.size()
        log_prob = torch.zeros((batch_size))
        for i in range(length):
            log_prob += self.xyprob[x[i]][:,y]
        log_prob += self.yprob[y]
        return log_prob

    def forward(self, input):
        x = input.data
        scores = torch.cat([self._score(x, y).view(-1, 1) for y in self.ys], 1)
        return scores

    def update_counts(self, x, y):
        xd = x.data  # (length, batch_size)
        yd = y.data.ne(0) # (batch_size)
        
        length, batch_size = xd.size()
        for l in range(length):
            for i in range(batch_size):
                self.xycounts[xd[l][i], yd[i]] += 1
        for i in range(batch_size):
            self.ycounts[yd[i]] += 1

class LR(nn.Module):
    def __init__(self, vocab, dropout, binarize=False):
        super(LR, self).__init__()
        self.vsize = len(vocab.itos)
        # Since this is binary LRRwe can just use BCEWithLogitsLoss
        self.lut = nn.Embedding(self.vsize, 1, padding_idx=1)
        self.bias = Parameter(torch.zeros(1))
        self.binarize = binarize

    def forward(self, input):
        # input will be seqlen x bsz, so we want to use the weights from the lut
        # and sum them up to get the logits.
        if self.binarize:
            output = V(torch.FloatTensor(input.size(1)))
            for j in range(input.size(1)):
                tokens = torch.LongTensor(list(set(input[:,j].tolist())))
                output[j] = self.lut(V(tokens)).sum()
            return output + self.bias
        else:
            return self.lut(input).squeeze(-1).sum(0) + self.bias


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


class CBoW(nn.Module):
    def __init__(self, vocab, dropout=0, binarize=True):
        super(CBoW, self).__init__()
        self.vsize = len(vocab.itos)
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
        hidden = torch.cat([self.lut(input).sum(0), self.static_lut(input).sum(0)], -1)
        if self.dropout:
            hidden = self.dropout(hidden)
        return self.proj(hidden).squeeze(-1)

class CNN(nn.Module):
    def __init__(self, vocab, dropout=0):
        super(CNN, self).__init__()
        self.vsize = len(vocab.itos)
        self.static_lut = nn.Embedding(self.vsize, vocab.vectors.size(1))
        self.static_lut.weight.data.copy_(vocab.vectors)
        #self.static_lut.requires_grad = False
        self.lut = nn.Embedding(self.vsize, vocab.vectors.size(1))
        self.lut.weight.data.copy_(simple_vec)
        #self.lut.requires_grad = False
        self.convs = nn.ModuleList([
            nn.Conv1d(600, 100, 3),
            nn.Conv1d(600, 100, 4, padding=1),
            nn.Conv1d(600, 100, 5, padding=1),
        ])
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(300, 1)
        )

    def forward(self, input):
        word_rep = torch.cat([self.lut(input), self.static_lut(input)], -1).permute(1, 2, 0)
        cnns = torch.cat([F.relu(conv(word_rep)).max(2)[0] for conv in self.convs], 1)
        return self.proj(cnns).squeeze(-1)

class CNNLOL(nn.Module):
    def __init__(self, vocab, dropout=0):
        super(CNN, self).__init__()
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

class CNNLSTM(nn.Module):
    def __init__(self, vocab, dropout=0):
        super(CNNLSTM, self).__init__()
        self.vsize = len(vocab.itos)
        self.dropout = dropout
        self.static_lut = nn.Embedding(self.vsize, vocab.vectors.size(1))
        self.static_lut.weight.data.copy_(vocab.vectors)
        self.static_lut.requires_grad = False
        self.lut = nn.Embedding(self.vsize, vocab.vectors.size(1))
        self.lut.weight.data.copy_(simple_vec)
        self.lut.requires_grad = False
        self.convs = nn.ModuleList([
            nn.Conv1d(600, 200, 3, padding=1),
            nn.Conv1d(600, 200, 5, padding=2),
        ])
        self.lstm = nn.LSTM(1000, 500, bidirectional=True)
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1000, 1)
        )

    def forward(self, input):
        word_rep = torch.cat([self.lut(input), self.static_lut(input)], -1).permute(1, 2, 0)
        cnns = F.relu(torch.cat([conv(word_rep) for conv in self.convs] + [word_rep], 1))
        _, (h, c) = self.lstm(cnns.permute(2, 0, 1))
        z = torch.cat([h.permute(1,0,2), c.permute(1,0,2)], -1).view(input.size(1), -1)
        return self.proj(z).squeeze(-1)

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

def train_model(model, valid_fn, args, loss=nn.BCEWithLogitsLoss(), epochs=10, lr=1):
    ugh = iter(train_iter)
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = optim.Adagrad(params, lr = lr, weight_decay = args.wd)
    #optimizer = optim.Adadelta(params, lr = lr, weight_decay = args.wd)
    if args.optim == "SGD":
        optimizer = optim.SGD(params, lr = lr, weight_decay = args.wd, momentum=args.mom, dampening=args.dm, nesterov=not args.nonag)
    elif args.optim == "Adam":
        optimizer = optim.Adam(params, lr = lr, weight_decay = args.wd)
    elif args.optim == "Adadelta":
        optimizer = optim.Adadelta(params, lr = lr, weight_decay = args.wd)
    elif args.optim == "Adamax":
        optimizer = optim.Adamax(params, lr = lr, weight_decay = args.wd)
    else:
        print("No optimizer found")
        return
    #optimizer = optim.LBFGS(params)
    #for p in optimizer.param_groups[0]['params']:
    #    optimizer.state[p]['sum'].fill_(1e-10)

    if args.load_model:
        checkpoint = torch.load(args.model_dir + args.model_name)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        print("Load from %s" % (args.model_dir + args.model_name))

    best_epoch = -1
    best_val_acc = -1
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        print("Epoch {} | lr {}".format(epoch, lr))
        for batch in tqdm(train_iter):
        #for i in tqdm(range(len(train_iter))):
            #batch = next(ugh)
            x = batch.text
            y = batch.label
            """
            optimizer.zero_grad()
            bloss = loss(model(x), y.float())
            train_loss += bloss
            bloss.backward()
            if args.clip > 0:
                clip_grad_norm(params, args.clip)
            optimizer.step()
            """
            def closure():
                nonlocal train_loss
                optimizer.zero_grad()
                bloss = loss(model(x), y.float())
                bloss.backward()
                train_loss += bloss
                return bloss
            optimizer.step(closure)
            #clip_param_norm(params, 3)
        train_loss /= len(train_iter)
        print("Train loss: " + str(train_loss.data[0]))
        train_acc = valid_fn(model, train_iter)
        print("Train acc: " + str(train_acc))
        val_acc = valid_fn(model, valid_iter)
        print("Valid acc: " + str(val_acc))
        lr *= args.lrd
        if val_acc[2] > best_val_acc:
            best_epoch = epoch
            best_val_acc = val_acc[2]
        #if (epoch+1) % 5 == 0:
            #lr /= 2
            #for p in optimizer.param_groups[0]['params']:
            #    optimizer.state[p]['sum'].fill_(1e-10)
            #    optimizer.state[p]['acc_delta'].fill_(1e-10)

        def save_checkpoint(state, filename):
            torch.save(state, filename)
            print("Saved to %s" % filename)

        filename = args.model_dir + args.model_name + ".%s.epoch%d" % (args.model, epoch)
        save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

    print("best epoch: %d  | best_val_acc = %.3lf" % (best_epoch, best_val_acc))
    filename = args.model_dir + args.model_name + ".%s.epoch%d" % (args.model, best_epoch)
    print("best model: %s" % filename)
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])

models = [NB, NB2, LR, LR2, CBoW, CNN, CNNLOL, CNNLSTM, LSTM]

if "NB" in args.model:
    model = list(filter(lambda x: x.__name__ == args.model, models))[0](
        TEXT.vocab, len(LABEL.vocab.itos), binarize=args.binarize_nb)
    nb = train_nb(model)
    print(validate_nb(nb, train_iter))
    print(validate_nb(nb, valid_iter))
    if args.output_file:
        output_test_nb(nb)
else:
    model = list(filter(lambda x: x.__name__ == args.model, models))[0](TEXT.vocab, args.dropout)
    if args.gpu > 0:
        model.cuda(args.gpu)
    print(model)
    train_model(model, validate, args, epochs=args.epochs, lr=args.lr)
    print(validate(model, train_iter))
    print(validate(model, valid_iter))
    print(validate(model, test_iter))
    if args.output_file:
        output_test(model)

