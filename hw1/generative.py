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

torch.manual_seed(1111)
torch.cuda.manual_seed_all(1111)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)

    parser.add_argument("--nhid", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=1)

    parser.add_argument("--tieweights", action="store_true")

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--optim", choices=["SGD", "Adam"], default="Adam")

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lrd", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--bsz", type=int, default=32)
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

    parser.add_argument("--dooutput", action="store_true")

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
vsize = len(TEXT.vocab.itos)
bos = "<bos>"
TEXT.vocab.itos.append(bos)
TEXT.vocab.stoi[bos] = vsize
eos = "<eos>"
TEXT.vocab.itos.append(eos)
TEXT.vocab.stoi[eos] = vsize + 1
positive = "<positive>"
TEXT.vocab.itos.append(positive)
TEXT.vocab.stoi[positive] = vsize + 2
negative= "<negative>"
TEXT.vocab.itos.append(negative)
TEXT.vocab.stoi[negative] = vsize + 3
assert(vsize + 4 == len(TEXT.vocab.itos))
vsize = len(TEXT.vocab.itos)

pad_id = TEXT.vocab.stoi["<pad>"]
unk_id = TEXT.vocab.stoi["<unk>"]
bos_id = TEXT.vocab.stoi[bos]
eos_id = TEXT.vocab.stoi[eos]
positive_id = TEXT.vocab.stoi[positive]
negative_id = TEXT.vocab.stoi[negative]

LABEL.build_vocab(train)
LABEL.vocab.itos = ['positive', 'negative']
LABEL.vocab.stoi['positive'] = 0
LABEL.vocab.stoi['negative'] = 1

print('len(TEXT.vocab)', len(TEXT.vocab))
print('len(LABEL.vocab)', len(LABEL.vocab))

labels = [ex.label for ex in train.examples]

train_iter, _, _ = torchtext.data.BucketIterator.splits(
    (train, valid, test), batch_size=args.bsz, device=-1, repeat=False)

_, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, valid, test), batch_size=10, device=-1)

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
#simple_vec = TEXT.vocab.vectors.clone()

#url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec'
#TEXT.vocab.load_vectors(vectors=Vectors('wiki.en.vec', url=url))
#complex_vec = TEXT.vocab.vectors


def output_test(model):
    "All models should be able to be run with following command."
    upload = []
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        yhat = F.sigmoid(model(batch.text)) > 0.5
        upload += yhat.tolist()

    with open(args.model + ".txt", "w") as f:
        f.write("Id,Cat\n")
        for i,u in enumerate(upload):
            f.write(str(i) + "," + str(u+1) + "\n")

def validate(model, valid):
    model.eval()
    correct = 0.
    total = 0.
    with torch.no_grad():
        for batch in valid:
            x = batch.text
            y = batch.label
            x, y = model.prep_sample(x, y)
            preds = model(x)
            T, N, V = preds.size()
            # if the score of the positive is higher than negative
            # we want to predict the label 0
            yhat = preds[-1, :, positive_id] < preds[-1, :, negative_id] 
            results = yhat.long().cpu() == batch.label
            correct += results.float().sum().data[0]
            total += results.size(0)
    return correct, total, correct / total

# Models
class LstmGen(nn.Module):
    # ignore nhid, lol
    def __init__(self, vocab, nhid, nlayers, tie_weights, dropout=0):
        super(LstmGen, self).__init__()
        self.vsize = len(vocab.itos)
        self.bos = vocab.stoi[bos]
        self.eos = vocab.stoi[eos]
        self.positive = vocab.stoi[positive]
        self.negative = vocab.stoi[negative]

        self.lut = nn.Embedding(self.vsize, vocab.vectors.size(1))
        self.lut.weight.data.copy_(vocab.vectors)
        self.rnn = nn.LSTM(300, 300, nlayers, bidirectional=False, dropout=dropout)
        self.decoder = nn.Linear(300, self.vsize)
        if tie_weights:
            self.tie_weights = tie_weights
            self.decoder.weight = self.lut.weight

        self.buffer = torch.LongTensor().cuda(args.gpu)
        # Pinned memory cannot be resized.
        self.workbuffer = torch.LongTensor()

    def forward(self, input):
        vectors = self.lut(input)
        out, (h, c) = self.rnn(vectors)
        return self.decoder(out)

    def prep_sample(self, x, y):
        T, N = x.size()
        self.workbuffer.resize_(T+3, N)
        # Samples should look like
        #     a b c d
        # coming in, but
        #     <bos> a b c d <eos> <positive>
        # coming out.
        self.workbuffer[1:-2,:].copy_(x.data)
        self.workbuffer[0,:].fill_(self.bos)
        self.workbuffer[-2,:].fill_(self.eos)
        self.workbuffer[-1].masked_fill_(y.data.eq(0), self.positive)
        self.workbuffer[-1].masked_fill_(y.data.eq(1), self.negative)
        self.buffer.resize_(self.workbuffer.size())
        self.buffer.copy_(self.workbuffer, async=True)
        return V(self.buffer[:-1,:]), V(self.buffer[1:,:])


def train_model(model, valid_fn, loss=nn.CrossEntropyLoss(), epochs=args.epochs, lr=args.lr):
    params = [p for p in model.parameters() if p.requires_grad]

    if args.optim == "SGD":
        optimizer = optim.SGD(
            params, lr = lr, weight_decay = args.wd, momentum=args.mom, dampening=args.dm, nesterov=not args.nonag)
    elif args.optim == "Adam":
        optimizer = optim.Adam(params, lr = lr, weight_decay = args.wd, amsgrad=False)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lrd)
    for epoch in range(epochs):
        scheduler.step()
        model.train()
        train_loss = 0
        print()
        print("Epoch {} | lr {}".format(epoch, scheduler.get_lr()))
        gradnorms = torch.FloatTensor(len(train_iter)) 
        for i, batch in enumerate(tqdm(train_iter)):
            x = batch.text
            y = batch.label
            x, y = model.prep_sample(x, y)
            def closure():
                nonlocal train_loss
                optimizer.zero_grad()
                bloss = loss(model(x).view(-1, vsize), y.view(-1))
                bloss.backward()
                gradnorms[i] = nn.utils.clip_grad_norm(params, args.clip)
                train_loss += bloss
                return bloss
            optimizer.step(closure)
        train_loss /= len(train_iter)
        print("Train loss: " + str(train_loss.data[0]))
        print("Max grad norm: {}, Avg grad norm: {}".format(gradnorms.max(), gradnorms.mean()))
        #train_acc = valid_fn(model, train_iter)
        #print("Train acc: " + str(train_acc))
        valid_acc = valid_fn(model, valid_iter)
        print("Valid acc: " + str(valid_acc))


def save_model(model, train, valid):
    name = "{}_train{}_valid_{}".format(args.model, train, valid)
    torch.save(model.cpu().state_dict(), name)

model = LstmGen(TEXT.vocab, args.nhid, args.nlayers, args.dropout, args.tieweights)
if args.gpu > 0:
    model.cuda(args.gpu)
print(model)

weight = torch.Tensor(vsize).fill_(1)
weight[pad_id] = 0 
weight[positive_id] = 5 
weight[negative_id] = 5
if args.gpu >= 0:
    weight = weight.cuda(args.gpu)
loss = nn.CrossEntropyLoss(weight=weight)

train_model(model, validate, loss=loss, epochs=args.epochs, lr=args.lr)
_, _, train_acc = validate(model, train_iter)
_, _, valid_acc = validate(model, valid_iter)
_, _, test_acc = validate(model, test_iter)
print("train: {}, valid: {}, test: {}".format(train_acc, valid_acc, test_acc))
#save_model(model, train_acc, valid_acc)
if args.dooutput:
    output_test(model)
