## LSTM w/ adam, no weight tying

Train: 51.75419829716999, Valid: 84.9553327520044, Test: 82.23019965874008
```
python hw2.py  --model LstmLm --devid 3 --lr 0.01 --clip 1 --optim Adam --nlayers 2 --nhid 512 --dropout 0.5 --epochs 30 --bsz 128 --bptt 3
```

## LSTM w/ adam, weight tying

Train: 49.6024828712182, Valid: 78.7980867591008, Test: 76.05102234030406
```
python hw2.py  --model LstmLm --devid 3 --lr 0.01 --clip 1 --optim Adam --nlayers 2 --nhid 512 --dropout 0.5 --epochs 30 --bsz 128 --bptt 32 --tieweights
```

## LSTM w/ adam, weight tying

Train: 50.97615970483675, Valid: 77.52746013760232, Test: 75.21362771905142
```
python hw2.py  --model LstmLm --devid 3 --lr 0.01 --lrd 0.25 --clip 5 --optim Adam --nlayers 2 --nhid 512 --dropout 0.5 --epochs 30 --bsz 128 --bptt 32 --tieweights
```

## Same

Train: 43.79502974945982, Valid: 76.17829400790994, Test: 73.27243286840336
```
python hw2.py  --model LstmLm --devid 3 --lr 0.01 --clip 2 --optim Adam --nlayers 2 --nhid 512 --dropout 0.5 --epochs 30 --bsz 128 --bptt 32 --tieweights
```

## LSTM w/ adam, weight tying, train longer

Train: 42.90723478099889, Valid: 75.74921810452948, Test: 72.84913233987702
```
python hw2.py  --model LstmLm --devid 1 --lr 0.01 --clip 2 --optim Adam --nlayers 2 --nhid 512 --dropout 0.5 --epochs 50 --bsz 128 --bptt 32 --tieweights
```

## NNLM w/ adam, weight tying

Train: 66.99094327467738, Valid: 227.08821469306596, Test: 208.32806255668248
```
python hw2.py  --model NnLm --devid 3 --lr 0.001 --clip 0 --optim Adam --nlayers 2 --nhid 512 --dropout 0 --epochs 30 --bsz 64 --bptt 64 --tieweights
```

## NNLM w/ adam, no weight tying

Train: 51.73658883899987, Valid: 174.66593207552353, Test: 161.97340620733382
```
python hw2.py  --model NnLm --devid 2 --lr 0.001 --clip 0 --optim Adam --nlayers 2 --nhid 512 --dropout 0 --epochs 10 --bsz 64 --bptt 64
```

## NNLM w/ adam, no weight tying, dropout

Train: 71.58999479091389, Valid: 158.07431086368382, Test: 146.13046578572258
```
python hw2.py  --model NnLm --devid 3 --lr 0.001 --clip 0 --optim Adam --nlayers 2 --nhid 512 --dropout 0.5 --epochs 20 --bsz 64 --bptt 64
```

## NNLM w/ adam, no weight tying, dropout, maxnorm embedding d d

Train: 63.69241726931047, Valid: 165.18956092843882, Test: 152.0618240821467
```
python hw2.py  --model NnLm --devid 0 --lr 0.001 --clip 0 --optim Adam --nlayers 2 --nhid 512 --dropout 0.5 --epochs 20 --bsz 64 --bptt 64 --maxnorm 3
```

## Ensemble of best NNLM with best RNNLM
(Softmax(nnlm) + Softmax(rnnlm))/2. adding the logits (multiplicative) did not work and gave test ppl > 500.

Test: 79.61830767766449

## Linear Interpolation of NGrams (smoothing on unigram, bigram and trigram)
(weights: 0.7, 0.2, 0.1)
python hw2.py --model=Ngram --bptt=1000000
Train: 540.660, Valid: 565.158

## Linear Interpolation of NGrams (smoothing on unigram only)
(weights: 0.7, 0.2, 0.1)
python hw2.py --model=Ngram --bptt=1000000
Train: 37.602, Valid: 241.977

## LSTM w/ Cache Model
(2000, 0.1, 1.0)
python hw2.py --model=Cache --bptt 5000 --devid 0
Valid: 71.03, Test: 69.23

