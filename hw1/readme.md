Got to ~82 max acc with:
```
python hw1.py -gpu 3 -lr 0.1 -lrd 0.8 -epochs 15 -dropout 0.6 -bsz 32 -model CNNLOL -mom 0.90
```
It's pretty noisy, though :P

Switched up CNN to be more faithful to yoon's 2014 model (~0.815 valid acc):
```
python hw1.py -gpu 3 -lr 0.1 -lrd 0.8 -epochs 25 -dropout 0.5 -bsz 64 -model CNN -mom 0.90
```

