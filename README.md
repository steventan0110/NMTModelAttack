## Doubly Trained Neural Machine Translation System for Adversarial Attack and Data Augmentation

### Languages Experimented:
- Data Overview:

    |Source|Target|Training Data|Valid1|Valid2|Test data
    |--|--|--|--|--|--|
    |ZH|EN|WMT17 without UN corpus|WMT2017 newstest|WMT2018 newstest| WMT2020 newstest|
    |DE|EN|WMT17|WMT2017 newstest|WMT2018 newstest|WMT2014 newstest|
    |FR|EN|WMT14 without UN corpus|WMT2015 newsdiscussdev|WMT2015 newsdiscusstest|WMT2014 newstest|
    
- Corpus Statistics:
    |Lang-pair|Data Type|#Sentences|#tokens (English side)|
    |--|--|--|--|
    |zh-en|Train|9355978|161393634|
    ||Valid1|2001|47636|
    ||Valid2|3981|98308|
    ||test|2000|65561|
    |de-en|Train|4001246|113777884|
    ||Valid1|2941|74288|
    ||Valid2|2970|78358|
    ||test|3003|78182|
    |fr-en|Train|23899064|73523616|
    ||Valid1|1442|30888|
    ||Valid2|1435|30215|
    ||test|3003|81967|

### Scripts (as shown in paper's appendix)
- Set-up:
    - To execute the scripts shown below, it's required that fairseq version 0.9 is installed along with COMET. The way to easily install them after cloning this repo is executing following commands (under root of this repo):
        ```bash
        cd fairseq-0.9.0
        pip install --editable ./
        cd ../COMET
        pip install .
        ```
    - It's also possible to directly install COMET through pip:   `pip install unbabel-comet`, but the recent version might have different dependency on other packages like fairseq. Please check COMET's [official website](https://github.com/Unbabel/COMET) for the updated information.
    - To make use of script that relies on COMET model (in case of `dual-comet`), a model from COMET should be downloaded. It can be easily done by running following script:
        ```python
        from comet.models import download_model
        download_model("wmt-large-da-estimator-1719")
        ```
- Pretrain the model:
    ```bash
    fairseq-train $DATADIR \
        --source-lang $src \
        --target-lang $tgt \
        --save-dir $SAVEDIR \
        --share-decoder-input-output-embed \
        --arch transformer_wmt_en_de \
        --optimizer adam --adam-betas ’(0.9, 0.98)’ --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 --warmup-updates 4000 \
        --lr 0.0005 --min-lr 1e-09 \
        --dropout 0.3 --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 2048 --update-freq 16 \
        --seed 2 
    ```

- Adversarial Attack:
    ```bash
    fairseq-train $DATADIR \
        --source-lang $src \
        --target-lang $tgt \
        --save-dir $SAVEDIR \
        --share-decoder-input-output-embed \
        --train-subset valid \
        --arch transformer_wmt_en_de \
        --optimizer adam --adam-betas ’(0.9, 0.98)’ --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 --warmup-updates 4000 \
        --lr 0.0005 --min-lr 1e-09 \
        --dropout 0.3 --weight-decay 0.0001 \
        --criterion dual_bleu --mrt-k 16 \
        --batch-size 2 --update-freq 64 \
        --seed 2 \
        --restore-file $PREETRAIN_MODEL \
        --reset-optimizer \
        --reset-dataloader 
    ```
- Data Augmentation:
    ```bash
    fairseq-train $DATADIR \
        -s $src -t $tgt \
        --train-subset valid \
        --valid-subset valid1 \
        --left-pad-source False \
        --share-decoder-input-output-embed \
        --encoder-embed-dim 512 \
        --arch transformer_wmt_en_de \
        --dual-training \
        --auxillary-model-path $AUX_MODEL \
        --auxillary-model-save-dir $AUX_MODEL_SAVE \
        --optimizer adam --adam-betas ’(0.9, 0.98)’ --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt \
        --warmup-init-lr 0.000001 --warmup-updates 1000 \
        --lr 0.00001 --min-lr 1e-09 \
        --dropout 0.3 --weight-decay 0.0001 \
        --criterion dual_comet/dual_mrt --mrt-k 8 \
        --comet-route $COMET_PATH \
        --batch-size 4 \
        --skip-invalid-size-inputs-valid-test \
        --update-freq 1 \
        --on-the-fly-train --adv-percent 30 \
        --seed 2 \
        --restore-file $PRETRAIN_MODEL \
        --reset-optimizer \
        --reset-dataloader \
        --save-dir $CHECKPOINT_FOLDER 
    ```

### Generation and Test:
-   For Chinese-English, we use sentencepiece to perform the BPE so it's required to be removed in generation step. For all test we use beam size = 5. Noitce that we modified the code in fairseq-gen to use `sacrebleu.tokenizers.TokenizerZh()` to tokenize Chinese when the direction is en-zh.  
    ```bash
    fairseq-generate $DATA-FOLDER \
        -s zh -t en \
        --task translation \
        --gen-subset $file \
        --path $CHECKPOINT \
        --batch-size 64 --quiet \
        --lenpen 1.0 \
        --remove-bpe sentencepiece \
        --sacrebleu \
        --beam 5
    ```
- For French-Enlish, German-English, we modified the script to detokenize the moses tokenizer (which we used to preprocess the data). To reproduce the result, use following script:

    ```bash
    fairseq-generate $DATA-FOLDER \
        -s de/fr -t en \
        --task translation \
        --gen-subset $file \
        --path $CHECKPOINT \
        --batch-size 64 --quiet \
        --lenpen 1.0 \
        --remove-bpe \
        ---detokenize-moses \
        --sacrebleu \
        --beam 5
    ```
    Here `--detokenize-moses` would call detokenizer during the generation step and detokenize predictions before evaluating it. It would slow the generation step. Another way to manually do this is to retrieve prediction and target sentences from output file of fairseq and manually apply detokenizer from [detokenizer.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/mosestokenizer/detokenizer.py).
    
### BibTex
```
@misc{tan2021doublytrained,
      title={Doubly-Trained Adversarial Data Augmentation for Neural Machine Translation}, 
      author={Weiting Tan and Shuoyang Ding and Huda Khayrallah and Philipp Koehn},
      year={2021},
      eprint={2110.05691},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
