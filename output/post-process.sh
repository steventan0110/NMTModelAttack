#!/bin/bash
source /home/steven/Documents/GITHUB/NMTModelAttack/ma/bin/activate

MOSES=/home/steven/Documents/GITHUB/mosesdecoder
DETOK=$MOSES/scripts/tokenizer/detokenizer.perl
SCRIPT=/home/steven/Documents/GITHUB/NMTModelAttack/output/post-process.py
FILE=$1

cat $FILE | grep ^H- | cut -d'-' -f2- | cut -f3- > hypothesis.en
cat $FILE | grep ^T- | cut -f2- > ref.en

$DETOK -l en < hypothesis.en > hypothesis.detok.en
$DETOK -l en < ref.en > ref.detok.en

python $SCRIPT hypothesis.detok.en ref.detok.en

