from fairseq.models.roberta import XLMRModel
from comet.models.encoders.encoder_base import Encoder
from comet.tokenizers import XLMRTextEncoder
from torchnlp.download import download_file_maybe_extract
from torchnlp.utils import lengths_to_mask
import os

XLMR_BASE_URL = "https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz"
XLMR_BASE_MODEL_NAME = "xlmr.base/model.pt"
filepath = "/home/steven/Documents/GITHUB/NMTModelAttack/dataset"
def main():
    train_file = os.path.join(filepath, 'wmt17_without_un')
    pretrained_model = "xlmr.base"
    saving_directory = '/home/steven/Documents/GITHUB/NMTModelAttack/pretrain_models/'
    download_file_maybe_extract(
        XLMR_BASE_URL,
        directory=saving_directory,
        check_files=[XLMR_BASE_MODEL_NAME],
    )
    xlmr = XLMRModel.from_pretrained(
        saving_directory + pretrained_model, checkpoint_file="model.pt"
    )
    tokenizer = XLMRTextEncoder(
        xlmr.encode, xlmr.task.source_dictionary.__dict__["indices"]
    )
    vocab = xlmr.task.source_dictionary.__dict__["indices"]
    vocab_str =""
    for key, value in vocab.items():
        vocab_str += str(key)
        vocab_str += " "
        vocab_str += str(value)
        vocab_str += "\n"
    with open(saving_directory + pretrained_model+"/xlmr_vocab.txt", 'w') as f:
        f.write(vocab_str)
    encoder_output = tokenizer.encode("表演的明星是X女孩团队——由一对具有天才技艺的艳舞女孩们组成，其中有些人受过专业的训练。")
    print(encoder_output)






if __name__ == '__main__':
    main()