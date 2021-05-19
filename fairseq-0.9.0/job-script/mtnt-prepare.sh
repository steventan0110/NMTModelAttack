#!/bin/bash

SCRIPTS=/home/steven/Documents/GITHUB/mosesdecoder/scripts
MOSES_TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl

# Load config
source /home/steven/Documents/GITHUB/NMTModelAttack/ma/bin/activate

CORPUS_DIR=/home/steven/Documents/GITHUB/NMTModelAttack/dataset/fr-en

# File name
CORPUS_FILE="${CORPUS_DIR}/train.fr"
CORPUS_FILE_EN="${CORPUS_DIR}/train.en"

# Create corpus dir
mkdir -p $CORPUS_DIR

# Download data
wget -nv -O ${CORPUS_DIR}/europarl.tgz http://www.statmt.org/europarl/v7/fr-en.tgz
wget -nv -O ${CORPUS_DIR}/commoncrawl.tgz http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
wget -nv -O ${CORPUS_DIR}/un.tgz http://www.statmt.org/wmt13/training-parallel-un.tgz
wget -nv -O ${CORPUS_DIR}/nc.tgz http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz
wget -nv -O ${CORPUS_DIR}/giga.tar http://www.statmt.org/wmt10/training-giga-fren.tar

# Extract French and english training files
tar -zxvf ${CORPUS_DIR}/europarl.tgz -C ${CORPUS_DIR} europarl-v7.fr-en.fr europarl-v7.fr-en.en
tar -zxvf ${CORPUS_DIR}/commoncrawl.tgz -C ${CORPUS_DIR} commoncrawl.fr-en.fr commoncrawl.fr-en.en
tar -zxvf ${CORPUS_DIR}/un.tgz -C ${CORPUS_DIR} un/undoc.2000.fr-en.fr un/undoc.2000.fr-en.en
tar -zxvf ${CORPUS_DIR}/nc.tgz -C ${CORPUS_DIR} news-commentary-v10.fr-en.fr news-commentary-v10.fr-en.en
tar -xvf ${CORPUS_DIR}/giga.tar -C ${CORPUS_DIR} giga-fren.release2.fixed.fr.gz giga-fren.release2.fixed.en.gz
gzip -d ${CORPUS_DIR}/giga-fren.release2.fixed.fr.gz
gzip -d ${CORPUS_DIR}/giga-fren.release2.fixed.en.gz

# Concatenate
cat ${CORPUS_DIR}/giga-fren.release2.fixed.fr ${CORPUS_DIR}/commoncrawl.fr-en.fr ${CORPUS_DIR}/news-commentary-v10.fr-en.fr ${CORPUS_DIR}/un/undoc.2000.fr-en.fr ${CORPUS_DIR}/europarl-v7.fr-en.fr > $CORPUS_FILE
cat ${CORPUS_DIR}/giga-fren.release2.fixed.en ${CORPUS_DIR}/commoncrawl.fr-en.en ${CORPUS_DIR}/news-commentary-v10.fr-en.en ${CORPUS_DIR}/un/undoc.2000.fr-en.en ${CORPUS_DIR}/europarl-v7.fr-en.en > $CORPUS_FILE_EN

# Tokenize
#$MOSES_TOKENIZER -l fr -threads 4 < ${CORPUS_FILE} > ${CORPUS_FILE}.temp
#mv ${CORPUS_FILE}.temp $CORPUS_FILE
#$MOSES_TOKENIZER -l en -threads 4 < ${CORPUS_FILE_EN} > ${CORPUS_FILE_EN}.temp
#mv ${CORPUS_FILE_EN}.temp $CORPUS_FILE_EN

# Delete residual files and folders
#rm ${CORPUS_DIR}/*.{tar,gz,tgz}