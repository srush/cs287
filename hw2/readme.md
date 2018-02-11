LSTM w/ adam, no weight tying

Train: 51.75419829716999, Valid: 84.9553327520044, Test: 82.23019965874008
```
python hw2.py  --model LstmLm --devid 3 --lr 0.01 --clip 1 --optim Adam --nlayers 2 --nhid 512 --dropout 0.5 --epochs 30 --bsz 128 --bptt 3
```

LSTM w/ adam, weight tying

Train: 49.6024828712182, Valid: 78.7980867591008, Test: 76.05102234030406
```
python hw2.py  --model LstmLm --devid 3 --lr 0.01 --clip 1 --optim Adam --nlayers 2 --nhid 512 --dropout 0.5 --epochs 30 --bsz 128 --bptt 32 --tieweights
```
