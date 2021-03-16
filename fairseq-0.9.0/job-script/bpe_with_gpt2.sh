#!/bin/bash
source /home/steven/Documents/GITHUB/NMTModelAttack/ma/bin/activate

ROOT=/home/steven/Documents/GITHUB/NMTModelAttack/dataset
DATA=$ROOT/dataset
SRC=zh
TRT=en
Encoder=$ROOT/pretrain_models/roberta.large/roberta-large-vocab.json
Vocab=$ROOT/pretrain_models/roberta.large/merge.txt

python -m fairseq.examples.roberta.multiprocessing_bpe_encoder.py \
    --encoder-json $Encoder \
    --vocab-bpe $Vocab \
    --inputs $DATA/wmt17_without_un.$SRC $DATA/wmt17_without_un.$TRT \
    --outputs $DATA/gpt/train.bpe.$SRC $DATA/gpt/train.bpe.$TRT \
    --keep-empty \
    --workers 8; \
