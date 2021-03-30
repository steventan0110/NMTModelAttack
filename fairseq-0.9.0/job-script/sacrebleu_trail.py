from fairseq.bleu import SacrebleuScorer
import sacrebleu

def main():
    scorer = sacrebleu
    sys = ["这是一句测试"]
    ref = ["这是一据测试"]
    tok = sacrebleu.tokenizers.TokenizerZh()
    print(tok)
    token_sys = []
    token_ref = []
    for item in sys:
        token_sys.append(tok(item))
    for item in ref:
        token_ref.append(tok(item))
    print(token_sys)
    print(token_ref)



    s = scorer.corpus_bleu(token_sys, [token_ref])
    print(s)

if __name__ == '__main__':
    main()
