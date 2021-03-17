def main(src, trg, ref):
    with open("/home/steven/Documents/GITHUB/NMTModelAttack/data-bin/dict.zh.txt", 'r') as f:
        zh_dict = f.read()
    with open("/home/steven/Documents/GITHUB/NMTModelAttack/data-bin/dict.en.txt", 'r') as f1:
        en_dict = f1.read()

    # print(zh_dict)
    zh_stoi = {}
    zh_itos = {}
    en_stoi = {}
    en_itos = {}
    for lines in zh_dict.split('\n'):
        # print(lines.split(' '))
        token, idx = lines.split(' ')
        zh_stoi[token] = int(idx)
        zh_itos[int(idx)] = token
    for lines in en_dict.split('\n'):
        token, idx = lines.split(' ')
        en_stoi[token] = int(idx)
        en_itos[int(idx)] = token
    zh_itos[0] = "<s>"
    zh_itos[1] = "<pad>"
    zh_itos[2] = "</s>"
    zh_itos[3] = "<unk>"
    en_itos[0] = "<s>"
    en_itos[1] = "<pad>"
    en_itos[2] = "</s>"
    en_itos[3] = "<unk>"
    # tranlate source
    src_output = []
    trg_output = []
    ref_output = []
    for number in src:
        src_output.append(zh_itos[number])
    for number in trg:
        trg_output.append(en_itos[number])
    for number in ref:
        ref_output.append(en_itos[number])
    print(src_output)
    print(trg_output)
    print(ref_output)








if __name__ == '__main__':
    src = [6, 82475, 27283, 4, 29082, 1677, 573, 29738, 59030,
    465, 75092, 10766, 18321, 1403, 10436, 4, 37428, 264,
    5477, 43533, 153351, 100485, 40169, 187896, 30, 2]


    trg= [  9578,     70,  28271,      4,   2685,  38134,    111,   6181,    111,
          19882,    100,  26983,      7,     47,  19922,   6098, 146619,      5,
            136,     70,  11034,    136,   1733,    621,  28032,      5,      2,
            111,    111,    111,    111,    111,    111,    111,    111,    111,
            111,    111,    111,    111,    111,    111,    111,    111,    111,
            111,    111,    111,    111,    111,    111,    111]

    ref= [  2161,     70,  28271,      4,     10,  96551,    111,  26950,    621,
              19882,    100, 130367,      7,     47,  58359,     70,  48938,      4,
                136,     70,  11034,    136,   1733,    621, 169022,      5,      2,
                  1,      1,      1,      1,      1,      1,      1,      1,      1,
                  1,      1,      1,      1,      1,      1,      1,      1,      1,
                  1,      1,      1,      1,      1,      1,      1]
    trg =[114588,    214,    214,     38,      2]
    src = [6, 36727, 71831, 38, 2]
    ref = [189483, 1865, 214, 38, 2]
    main(src=src, trg=trg, ref=ref)
    # ['▁', '惊', '险', '!', '</s>']
    # ['▁Breath', 'tak', 'ing', '!', '</s>']

    # ['▁', '总体', '而言', ',', '市民', '去', '大', '兴', '机场', '有', '多种', '交通', '工具', '可', '选', ',', '费用', '和', '时间', '也在', '较为',
    #  '合理的', '范围', '之内', '。', '</s>']
    # ['▁Over', '▁the', '▁whole', ',', '▁there', '▁wide', '▁of', '▁transport', '▁of', '▁available', '▁for', '▁travel',
    #  's', '▁to', '▁visit', '▁Tai', '▁airport', '.', '▁and', '▁the', '▁cost', '▁and', '▁time', '▁are', '▁within', '.',
    #  '</s>', '▁of', '▁of', '▁of', '▁of', '▁of', '▁of', '▁of', '▁of', '▁of', '▁of', '▁of', '▁of', '▁of', '▁of', '▁of',
    #  '▁of', '▁of', '▁of', '▁of', '▁of', '▁of', '▁of', '▁of', '▁of', '▁of']
    # ['▁On', '▁the', '▁whole', ',', '▁a', '▁variety', '▁of', '▁means', '▁are', '▁available', '▁for', '▁citizen', 's',
    #  '▁to', '▁reach', '▁the', '▁Airport', ',', '▁and', '▁the', '▁cost', '▁and', '▁time', '▁are', '▁reasonable', '.',
    #  '</s>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>',
    #  '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>',
    #  '<pad>']



# softmax output:  tensor([[114588,    214,    214,     38,      2]], device='cuda:0')
# [[{'tokens': tensor([114588,     38,      2], device='cuda:0'), 'score': -1.4586753845214844, 'attention': tensor([[0.1478, 0.0830, 0.0940],
#         [0.2581, 0.1109, 0.1714],
#         [0.2494, 0.1201, 0.1572],
#         [0.0721, 0.2517, 0.0846],
#         [0.2727, 0.4343, 0.4928]], device='cuda:0'), 'alignment': None, 'positional_scores': tensor([-3.1153, -1.1383, -0.1224], device='cuda:0')}]]
