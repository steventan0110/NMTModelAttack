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

    # tensor([48.4940, 81.0988], grad_fn= < SumBackward1 >)
    # tensor([[276, 3579, 2189, 951, 13, 4655, 12, 947, 13, 8385, 5921, 170,
    #          1738, 2840, 1295, 190, 238, 8477, 1994, 6, 178, 391, 204, 350,
    #          1163, 109, 3489, 7, 726, 5, 2],
    #         [1327, 6125, 20, 6660, 3435, 583, 586, 93, 1298, 625, 19, 5695,
    #          5119, 93, 16, 3817, 6161, 4, 39, 16, 6, 1362, 674, 8,
    #          6, 1574, 533, 20, 143, 5, 2]])
