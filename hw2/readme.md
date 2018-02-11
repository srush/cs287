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

LSTM w/ adam, weight tying

Train: 50.97615970483675, Valid: 77.52746013760232, Test: 75.21362771905142
```
python hw2.py  --model LstmLm --devid 3 --lr 0.01 --lrd 0.25 --clip 5 --optim Adam --nlayers 2 --nhid 512 --dropout 0.5 --epochs 30 --bsz 128 --bptt 32 --tieweights
```

Same

Train: 43.79502974945982, Valid: 76.17829400790994, Test: 73.27243286840336
```
python hw2.py  --model LstmLm --devid 3 --lr 0.01 --clip 2 --optim Adam --nlayers 2 --nhid 512 --dropout 0.5 --epochs 30 --bsz 128 --bptt 32 --tieweights
```
