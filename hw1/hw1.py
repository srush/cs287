import argparse
 
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

torch.manual_seed(1111)
torch.cuda.manual_seed_all(1111)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", choices=['NB', "LR", "CBoW", "CNN", "CNNLSTM", "LSTM"])
    parser.add_argument("-gpu", type=int, default=-1)
    parser.add_argument("-lr", type=float, default=1)
    parser.add_argument("-lrd", type=float, default=0.9)
    parser.add_argument("-epochs", type=int, default=10)
    parser.add_argument("-dropout", type=float, default=0.5)
    parser.add_argument("-bsz", type=int, default=32)
    parser.add_argument("-mom", type=float, default=0.9)
    parser.add_argument("-dm", type=float, default=0)
    parser.add_argument("-wd", type=float, default=1e-4)
    parser.add_argument("-nonag", action="store_false")
    parser.add_argument("-clip", type=float, default=0.1)
    parser.add_argument("-output_file", type=str, default=None)
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
    (train, valid, test), batch_size=args.bsz, device=args.gpu)

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
        results = argmax == y
        correct += float(results.sum())
        total += results.size(0)
    return correct , total, correct / total

def validate(model, valid):
    model.eval()
    correct = 0.
    total = 0.
    with torch.no_grad():
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
    def __init__(self, vocab, dropout, alpha=1):
        super(NB, self).__init__()
        self.vsize = len(vocab.itos)
        self.nclass = 2
        self.alpha = alpha
        self.ys = list(range(self.nclass))
        self.xycounts = Parameter(torch.FloatTensor(len(vocab.itos), self.nclass).fill_(alpha))
        self.ycounts = Parameter(torch.FloatTensor(self.nclass).fill_(alpha))

    def _score(self, x, y):
        #return ((self.xycounts[x, y]+1.) / (self.ycounts[y]+self.vsize)).prod(0)
        # ~50% acc, i'm guessing becuase of p(y)
        #return ((self.xycounts[x, y]+1).log_() - (self.ycounts[y]+self.vsize).log_() + self.ycounts[y].log_()).sum(0)
        # ~80% acc
        return ((self.xycounts[x, y]+self.alpha).log_() - (self.ycounts[y]+self.vsize*self.alpha).log_()).sum(0)

    def forward(self, input):
        # p(y|x) = p(y) \prod_i p(x|y) / p(x)
        x = input.data
        scores = torch.cat([self._score(x, y).view(-1, 1) for y in self.ys], 1)
        return scores
        #Z = sum(scores)
        #return scores / Z


    def update_counts(self, x, y):
        # Let's do a sparse to dense update? What's the point, just loop
        xd = x.data
        yd = y.data
        idxs = xd * 2
        idxs[yd.view(1,-1).ne(0).expand_as(xd)] += 1
        self.xycounts.data.put_(idxs, torch.ones_like(xd).float(), accumulate=True)
        self.ycounts.data.put_(yd, torch.ones_like(yd).float(), accumulate=True)

class LR(nn.Module):
    def __init__(self, vocab, dropout):
        super(LR, self).__init__()
        self.vsize = len(vocab.itos)
        # Since this is binary LRRwe can just use BCEWithLogitsLoss
        self.lut = nn.Embedding(self.vsize, 1)
        self.bias = Parameter(torch.zeros(1))

    def forward(self, input):
        # input will be seqlen x bsz, so we want to use the weights from the lut
        # and sum them up to get the logits.
        return self.lut(input).squeeze(-1).sum(0) + self.bias

class CBoW(nn.Module):
    def __init__(self, vocab, dropout=0):
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

def train_nb():
    nb = NB(TEXT.vocab, len(LABEL.vocab.itos))
    ugh = iter(train_iter)
    for i in range(len(train_iter)):
        batch = next(ugh)
        x = batch.text
        y = batch.label
        nb.update_counts(x, y)
    return nb

def train_model(model, valid_fn, loss=nn.BCEWithLogitsLoss(), epochs=10, lr=1):
    ugh = iter(train_iter)
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = optim.Adagrad(params, lr = lr, weight_decay = args.wd)
    #optimizer = optim.Adadelta(params, lr = lr, weight_decay = args.wd)
    optimizer = optim.SGD(params, lr = lr, weight_decay = args.wd, momentum=args.mom, dampening=args.dm, nesterov=not args.nonag)
    #for p in optimizer.param_groups[0]['params']:
    #    optimizer.state[p]['sum'].fill_(1e-10)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        print("Epoch {} | lr {}".format(epoch, lr))
        for i in tqdm(range(len(train_iter))):
            batch = next(ugh)
            x = batch.text
            y = batch.label
            optimizer.zero_grad()
            bloss = loss(model(x), y.float())
            train_loss += bloss
            bloss.backward()
            clip_grad_norm(params, args.clip)
            optimizer.step()
            #clip_param_norm(params, 3)
        train_loss /= len(train_iter)
        print("Train loss: " + str(train_loss.data[0]))
        val_acc = valid_fn(model, valid_iter)
        print("Valid acc: " + str(val_acc))
        lr *= args.lrd
        #if (epoch+1) % 5 == 0:
            #lr /= 2
            #for p in optimizer.param_groups[0]['params']:
            #    optimizer.state[p]['sum'].fill_(1e-10)
            #    optimizer.state[p]['acc_delta'].fill_(1e-10)



models = [NB, LR, CBoW, CNN, CNNLSTM, LSTM]

model = list(filter(lambda x: x.__name__ == args.model, models))[0](TEXT.vocab, args.dropout)
if args.model == "NB":
    nb = train_nb()
    print(validate_nb(nb, valid_iter))
    if args.output_file:
        output_test_nb(nb)
else:
    if args.gpu > 0:
        model.cuda(args.gpu)
    print(model)
    train_model(model, validate, epochs=args.epochs, lr=args.lr)
    #print(validate(model, valid_iter))
    if args.output_file:
        output_test(model)
