import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable as V
from torch.nn import Parameter
from torch.nn.utils import clip_grad_norm

import torchtext

from tqdm import tqdm
import numpy as np
import random
from IPython import embed

import math

DEBUG = False
TOP_K = 20
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=-1)

    parser.add_argument("--model", choices=["Ngram", "NnLm", "LstmLm", "Ensemble", "Cache"], default="LstmLm")
    parser.add_argument("--nhid", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=1)

    parser.add_argument("--tieweights", action="store_true")
    parser.add_argument("--maxnorm", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--optim", choices=["SGD", "Adam"], default="Adam")

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lrd", type=float, default=0.25)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--bptt", type=int, default=32)
    parser.add_argument("--clip", type=float, default=5)

    # Cache parameters
    parser.add_argument('--window', type=int, default=2000)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--lambdasm", type=float, default=0.1)

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

random.seed(1111)
torch.manual_seed(1111)
if args.devid >= 0:
    torch.cuda.manual_seed_all(1111)
    torch.backends.cudnn.enabled = False
    print("Cuddn is enabled: {}".format(torch.backends.cudnn.enabled))


# Maybe we should subclass LanguageModelingDataset?
TEXT = torchtext.data.Field()
train, valid, test = torchtext.datasets.LanguageModelingDataset.splits(
    path="data/",
    train="train.txt", validation="valid.txt", test="test.txt", text_field=TEXT)

TEXT.build_vocab(train)
padidx = TEXT.vocab.stoi["<pad>"]

train_iter, valid_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, valid, test), batch_size=(args.bsz if args.model != "Cache" else 1), device=args.devid, bptt_len=args.bptt, repeat=False)

#train_iter_ngram, valid_iter_ngram, test_iter_ngram = torchtext.data.BucketIterator.splits(
 #   (train, valid, test), batch_size=args.bsz, device=args.devid, repeat=False)

class Lm(nn.Module):
    def __init__(self):
        super(Lm, self).__init__()

    def train_epoch(self, iter, loss, optimizer, params):
        self.train()

        train_loss = 0
        nwords = 0

        hid = None
        for batch in tqdm(iter):
            optimizer.zero_grad()
            x = batch.text
            y = batch.target
            out, hid = model(x, hid if hid is not None else None)
            bloss = loss(out.view(-1, model.vsize), y.view(-1))
            bloss.backward()
            train_loss += bloss
            nwords += y.ne(padidx).int().sum()
            if args.clip > 0:
                clip_grad_norm(params, args.clip)

            optimizer.step()

        return train_loss.data[0], nwords.data[0]

    def validate(self, iter, loss):
        self.eval()

        valid_loss = 0
        nwords = 0

        hid = None
        for batch in iter:
            x = batch.text
            y = batch.target
            out, hid = model(x, hid if hid is not None else None)
            valid_loss += loss(out.view(-1, model.vsize), y.view(-1))
            nwords += y.ne(padidx).int().sum()
            #nwords += y.nelement()
        return valid_loss.data[0], nwords.data[0]

    def generate_predictions(self):
        self.eval()
        data = torchtext.datasets.LanguageModelingDataset(
            path="data/input.txt",
            text_field=TEXT)
        data_iter = torchtext.data.BPTTIterator(data, 211, 12, device=args.devid, train=False)
        # Well, if I do a bsz of 211 and join the prediction together later,
        # should be fine, but whatever.
        outputs = [[] for _ in range(211)]
        print()
        print("Generating Kaggle predictions")
        for batch in tqdm(data_iter):
            # T x N x V
            scores, idxs = self(batch.text, None)[0][-3].topk(20, dim=-1)
            for i in range(idxs.size(0)):
                outputs[i].append([TEXT.vocab.itos[x] for x in idxs[i].tolist()])
        with open(self.__class__.__name__ + ".preds.txt", "w") as f:
            f.write("id,word\n")
            ok = 1
            for sentences in outputs:
                f.write("\n".join(["{},{}".format(ok+i, " ".join(x)) for i, x in enumerate(sentences)]))
                f.write("\n")
                ok += len(sentences)


class NnLm(Lm):
    def __init__(self, vocab, nhid, kW=3, nlayers=1, dropout=0, tieweights=True):
        super(NnLm, self).__init__()
        self.vsize = len(vocab.itos)
        self.kW = kW
        self.nhid = nhid

        self.lut = nn.Embedding(self.vsize, nhid, max_norm=args.maxnorm)
        self.conv = nn.Conv1d(nhid, nhid, kW, stride=1)
        self.drop = nn.Dropout(dropout)
        m = []
        for i in range(nlayers-1):
            m.append(nn.Linear(nhid, nhid))
            m.append(nn.Tanh())
            m.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*m)
        self.proj = nn.Linear(nhid, self.vsize)
        if tieweights:
            self.proj.weight = self.lut.weight

    def forward(self, input, hid):
        emb = self.lut(input)
        T, N, H = emb.size()
        pad = emb.new(self.kW-1, N, H)
        pad.data.fill_(0)
        pad.requires_grad = False
        emb = torch.cat([pad, emb], 0)
        hid = self.conv(emb.permute(1,2,0)).permute(2,0,1)
        return self.proj(self.mlp(hid)), hid


class LstmLm(Lm):
    def __init__(self, vocab, nhid, nlayers=1, dropout=0, tieweights=True):
        super(LstmLm, self).__init__()
        self.vsize = len(vocab.itos)

        self.lut = nn.Embedding(self.vsize, nhid, max_norm=args.maxnorm)
        self.rnn = nn.LSTM(
            input_size=nhid,
            hidden_size=nhid,
            num_layers=nlayers,
            dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(nhid, self.vsize)
        if tieweights:
            self.proj.weight = self.lut.weight

    def forward(self, input, hid):
        emb = self.lut(input)
        hids, hid = self.rnn(emb, hid)
        return self.proj(self.drop(hids)), tuple(map(lambda x: x.detach(), hid))

    def forward_all(self, input, hid):
        emb = self.lut(input)
        hids, hid = self.rnn(emb, hid)
        return self.proj(self.drop(hids)), hids, tuple(map(lambda x: x.detach(), hid))

class Ensemble(Lm):
    def __init__(self, nnlm, rnnlm):
        super(Ensemble, self).__init__()
        self.nnlm = nnlm
        self.rnnlm = rnnlm
        self.vsize = nnlm.vsize

    def forward(self, input, hid):
        nnlmout, _ = self.nnlm(input, None)
        rnnlmout, hid = self.rnnlm(input, hid)
        return ((F.softmax(nnlmout, dim=-1) + F.softmax(rnnlmout, dim=-1)) / 2).log(), hid


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

        self.alphas = [0.7, 0.2, 0.1]
        #assert np.sum(self.alphas) == 1.0, np.sum(self.alphas) 
        self.a = 1 # smoothing factor
        self.V = len(TEXT.vocab)


    def train_batch(self, batch):
        batch = batch.text.data
        length, batch_size = batch.size()
        embed()
        
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
        
        if self.bigram_B.get(w2, 0) > 0:
            word_prob.append(self.alphas[1] * self.bigram_A.get(w2, {}).get(w3, 0) / self.bigram_B.get(w2, 0))
        else:
            word_prob.append(0.0)
        if self.trigram_B.get((w1, w2), 0) > 0:
            word_prob.append(self.alphas[2] * self.trigram_A.get((w1,w2), {}).get(w3, 0) / self.trigram_B.get((w1,w2),0))
        else:
            word_prob.append(0.0)
        
        #word_prob.append(self.alphas[1] * (1.0 * self.bigram_A.get(w2, {}).get(w3, 0) + self.a) / (1.0 * self.bigram_B.get(w2, 0) + self.a * self.V))
        #word_prob.append(self.alphas[2] * (1.0 * self.trigram_probs.get((w1, w2), {}).get(w3, 0) + self.a) / (1.0 * self.trigram_B.get((w1,w2), 0) + self.a * self.V))
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

def one_hot(idx, size, devid=-1):
    vec = np.zeros((1, size), dtype=np.float32)
    vec[0][idx] = 1
    vec_var = V(torch.from_numpy(vec))
    if devid >= 0:
        vec_var = vec_var.cuda()
    return vec_var

def evaluate_cache(model, data_iter, batch_size=1, window=args.window):
    model.eval()

    total_loss = 0
    total_len = 0
    vsize = len(TEXT.vocab)
    hid = None  # TODO: init hidden?

    next_word_history = None
    pointer_history = None
    
    for batch in tqdm(data_iter, desc="evaluate cache"):
        data = batch.text
        target = batch.target
        total_len += data.size(0)
        # Given batch size = 1, data/target's shape: (bptt, 1)
        output, rnn_outs, hidden = model.forward_all(data, hid)
        # output: (bptt, batch_size=1, vsize)
        # rnn_outs: (bptt, batch_size=1, hidden_dim)
        output_flat = output.squeeze(1)
        rnn_out = rnn_outs.squeeze(1)

        # Fill pointer history
        start_idx = len(next_word_history) if next_word_history is not None else 0
        if next_word_history is None:
            next_word_history = torch.cat([one_hot(t.data[0], vsize, args.devid) for t in target])
        else:
            next_word_history = torch.cat([next_word_history, torch.cat([one_hot(t.data[0], vsize, args.devid) for t in target])])
        if pointer_history is None:
            pointer_history = V(rnn_out.data)
        else:
            pointer_history = torch.cat([pointer_history, V(rnn_out.data)], dim=0)

        # Pointer manual cross entropy
        loss = 0
        softmax_output_flat = torch.nn.functional.softmax(output_flat, dim=1)
        # softmax_output_flat: (bptt, vsize)
        for idx, vocab_loss in enumerate(softmax_output_flat):
            p = vocab_loss
            if start_idx + idx > window:
                valid_next_word = next_word_history[start_idx + idx - window:start_idx + idx]
                valid_pointer_history = pointer_history[start_idx + idx - window:start_idx + idx]
                logits = torch.mv(valid_pointer_history, rnn_out[idx])
                theta = args.theta
                ptr_attn = torch.nn.functional.softmax(theta * logits, dim=0).view(-1, 1)
                ptr_dist = (ptr_attn.expand_as(valid_next_word) * valid_next_word).sum(0).squeeze()
                lambdah = args.lambdasm
                p = lambdah * ptr_dist + (1 - lambdah) * vocab_loss
            target_loss = p[target[idx].data]
            loss += (-torch.log(target_loss)).data[0]
        total_loss += loss / batch_size

        next_word_history = next_word_history[-window:]
        pointer_history = pointer_history[-window:]
    return total_loss / total_len

def generate_cache(model, batch_size=1, window=10, input_file="data/input.txt", output_file="cache.out"):
    # let's assume we can still apply caching on input.txt (consecutive)
    data = torchtext.datasets.LanguageModelingDataset(
        path=input_file,
        text_field=TEXT)
    data_iter = torchtext.data.BPTTIterator(data, batch_size, 12, device=args.devid, train=False)
    
    next_word_history = None
    pointer_history = None
    vsize = len(TEXT.vocab)
    hid = None  # TODO: init hidden?
    outputs = []
    for batch in tqdm(data_iter, desc="generate cache"):
        #embed()
        data = batch.text[:10]
        target = data[1:]

        output, rnn_outs, hidden = model.forward_all(data, hid)
        # output: (bptt, batch_size=1, vsize)
        # rnn_outs: (bptt, batch_size=1, hidden_dim)
        output_flat = output.squeeze(1)
        rnn_out = rnn_outs.squeeze(1)[:-1]

        #embed()
        # Fill pointer history
        start_idx = len(next_word_history) if next_word_history is not None else 0
        if next_word_history is None:
            next_word_history = torch.cat([one_hot(t.data[0], vsize, args.devid) for t in target])
        else:
            next_word_history = torch.cat([next_word_history, torch.cat([one_hot(t.data[0], vsize, args.devid) for t in target])])
        if pointer_history is None:
            pointer_history = V(rnn_out.data)
        else:
            pointer_history = torch.cat([pointer_history, V(rnn_out.data)], dim=0)

        # Pointer manual cross entropy
        loss = 0
        softmax_output_flat = torch.nn.functional.softmax(output_flat, dim=1)

        idx = softmax_output_flat.size(0) - 1
        vocab_loss = softmax_output_flat[idx]
        p = vocab_loss
        if start_idx + idx > window:
            valid_next_word = next_word_history[start_idx + idx - window:start_idx + idx]
            valid_pointer_history = pointer_history[start_idx + idx - window:start_idx + idx]
            logits = torch.mv(valid_pointer_history, rnn_out[idx])
            theta = args.theta
            ptr_attn = torch.nn.functional.softmax(theta * logits, dim=0).view(-1, 1)
            ptr_dist = (ptr_attn.expand_as(valid_next_word) * valid_next_word).sum(0).squeeze()
            lambdah = args.lambdasm
            p = lambdah * ptr_dist + (1 - lambdah) * vocab_loss

        # TODO(demi): do generation here!
        embed()
        scores, idxs = torch.topk(p, 20)
        outputs.append([TEXT.vocab.itos[x] for x in idxs.tolist()])

        next_word_history = next_word_history[-window:]
        pointer_history = pointer_history[-window:]

    with open(output_file, "w") as f:
        f.write("id,word\n")
        ok = 1
        for sentences in outputs:
            f.write("\n".join(["{},{}".format(ok+i, " ".join(x)) for i, x in enumerate(sentences)]))
            f.write("\n")
            ok += len(sentences)

if __name__ == "__main__":
    if args.model == "Ngram":
        ngram_model(args)
    elif args.model == "Ensemble":
        nnlm = torch.load("NnLm.pth")
        rnnlm = torch.load("LstmLm.pth")
        model = Ensemble(nnlm, rnnlm)
        if args.devid >= 0:
            model.cuda(args.devid)

        weight = torch.FloatTensor(model.vsize).fill_(1)
        weight[padidx] = 0
        if args.devid >= 0:
            weight = weight.cuda(args.devid)
        loss = nn.CrossEntropyLoss(weight=V(weight), size_average=False)
        test_loss, test_words = model.validate(test_iter, loss)
        print("Test: {}".format(math.exp(test_loss / test_words)))

        model.generate_predictions()

    elif args.model == "Cache":
        model = torch.load("LstmLm.pth")
        if args.devid >= 0:
            model.cuda(args.devid)

        generate_cache(model)

        avg_valid_loss = evaluate_cache(model, valid_iter, 1)
        avg_test_loss = evaluate_cache(model, test_iter, 1)
        print("avg_valid_loss", avg_valid_loss)
        print("avg_valid_perp", math.exp(avg_valid_loss))
        print("avg_test_loss", avg_test_loss)
        print("avg_test_perp", math.exp(avg_test_loss))
    
    else:
        models = {model.__name__: model for model in [NnLm, LstmLm]}
        model = models[args.model](
            TEXT.vocab, args.nhid, nlayers=args.nlayers, dropout=args.dropout, tieweights=args.tieweights)
        print(model)
        if args.devid >= 0:
            model.cuda(args.devid)

        weight = torch.FloatTensor(model.vsize).fill_(1)
        weight[padidx] = 0
        if args.devid >= 0:
            weight = weight.cuda(args.devid)
        loss = nn.CrossEntropyLoss(weight=V(weight), size_average=False)

        params = [p for p in model.parameters() if p.requires_grad]
        if args.optim == "Adam":
            optimizer = optim.Adam(params, lr = args.lr, weight_decay = args.wd, betas=(args.b1, args.b2))
        elif args.optim == "SGD":
            optimizer = optim.SGD(
                params, lr = args.lr, weight_decay = args.wd,
                nesterov = not args.nonag, momentum = args.mom, dampening = args.dm)
        # TODO: hack scheduler to halve every epoch after first reduce.
        schedule = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=1, factor=args.lrd, threshold=1e-3)

        for epoch in range(args.epochs):
            print("Epoch {}, lr {}".format(epoch, optimizer.param_groups[0]['lr']))
            train_loss, train_words = model.train_epoch(
                iter=train_iter, loss=loss, optimizer=optimizer, params=params)
            valid_loss, valid_words = model.validate(valid_iter, loss)
            schedule.step(valid_loss)
            print("Train: {}, Valid: {}".format(
                math.exp(train_loss / train_words), math.exp(valid_loss / valid_words)))

        test_loss, test_words = model.validate(test_iter, loss)
        print("Test: {}".format(math.exp(test_loss / test_words)))
        model.generate_predictions()
        torch.save(model.cpu(), model.__class__.__name__ + ".pth")
