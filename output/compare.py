def main():
    """
    compare two files from fairseq.generate's output
    """
    src = 'zh'
    tgt = 'en'
    f1 = 'mrt-'+src+'-'+tgt+'/baseline.out'
    epoch = 'e10.out'
    f2 = 'mrt-'+src+'-'+tgt+'/'+epoch

    with open(f1, 'r') as h1:
        d1 = h1.read()
    with open(f2, 'r') as h2:
        d2 = h2.read()

    src_data = []
    tgt_data = []
    hypo1 = []
    hypo2 = []
    all_data = list(zip(d1.split('\n'), d2.split('\n')))

    for i, item in enumerate(all_data):
        if i < 10 or i > len(all_data)-3: # remove header
            continue
        left, right = item[0], item[1]
        if left.startswith('S-'):
            src_data.append(left.split('\t')[1])
        elif left.startswith('T-'):
            tgt_data.append(left.split('\t')[1])
        elif left.startswith('H-'):
            hypo1.append(left.split('\t')[2])
            hypo2.append(right.split('\t')[2])
        else:
            continue

    output = 'mrt-'+src+'-'+tgt+'/diff-'+epoch
    sample = open(output, 'w')
    counter = 0
    for i in range(len(src_data)):
        if hypo1[i] != hypo2[i]:
            t1 = 'SRC: ' + src_data[i] + '\n'
            t2 = 'TGT: ' + tgt_data[i] + '\n'
            t3 = 'Hypo1: ' + hypo1[i] + '\n'
            t4 = 'Hypo2: ' + hypo2[i] + '\n'
            sample.writelines(t1+t2+t3+t4)
            counter += 1
    print('Different Translations: ', counter)
    sample.close()











if __name__ == '__main__':
    main()