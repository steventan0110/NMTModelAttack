import sys
import sacrebleu
def main(argv):
    filename_hypo = argv[0]
    filename_ref = argv[1]
    with open(filename_hypo, 'r') as f1:
        hypo = f1.read()
    with open(filename_ref, 'r') as f2:
        ref = f2.read()

    hypo = hypo.split('\n')
    ref = ref.split('\n')

    assert len(hypo) == len(ref)
    print(ref)
    bleu = sacrebleu.corpus_bleu(hypo, [ref])
    print(bleu.score)
    print(bleu)

if __name__ == '__main__':
    main(sys.argv[1:])