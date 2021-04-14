import math

from fairseq import utils
from . import FairseqCriterion, register_criterion
import torch.nn.functional as F
import torch
import math
import numpy as np

@register_criterion('comet_score')
class CometCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.alpha = args.comet_alpha

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--comet-alpha', default=0.5, type=float, metavar='D',
                            help="alpha value for comet prediction score's weight")
        # fmt: on

    def gumbel_softmax(self, input, tau=1, hard=False):
        return F.gumbel_softmax(input, tau=tau, hard=hard)

    def greedy_decode(self, model, sample):
        model.eval()
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }
        src_tokens = encoder_input['src_tokens']
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        eos = self.task.target_dictionary.eos()
        pad = self.task.target_dictionary.pad()
        # print("eos index: ", eos, "pad index: ", pad)
        src_length = (src_tokens.ne(eos) & src_tokens.ne(pad)).long().sum(dim=1)
        bsz = input_size[0]
        src_len = input_size[1]
        # print(src_tokens, src_len, src_length)
        encoder_outs = model.encoder(src_tokens, src_lengths=src_length)
        # max_len = model.max_decoder_positions() - 1
        max_len = 200
        tokens = src_tokens.new_full((bsz, max_len+2), pad)
        # tokens = torch.empty(bsz, max_len + 2).fill_(pad).to('cuda:0').long()
        tokens[:, 0] = eos
        for step in range(max_len + 1):
            lprob, avg_att = model.forward_decoder(tokens[:, :step + 1], encoder_out=encoder_outs, temperature=1)
            lprob = lprob[:, -1, :]
            lprob[:, pad] = -math.inf
            new_token = torch.argmax(lprob, dim=1)
            tokens[:, step + 1] = new_token
        # truncate the garbage prediction after eos tokens
        results = []
        for i in range(bsz):
            cur_arr = tokens[i, 1:].detach().cpu().numpy()
            end_idx = np.where(cur_arr == eos)[0]
            print(end_idx)
            # temp = torch.tensor(cur_arr[0:end_idx+1])
            # print(temp)
            print(self.task.source_dictionary.string(src_tokens[i, :], "sentencepiece", escape_unk=True))
            print(self.task.target_dictionary.string(cur_arr, "sentencepiece", escape_unk=True))
            results.append(temp)
        return results

    def forward(self, input, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model, comet_model = input[0], input[1]
        result = self.greedy_decode(model, sample)

        raise Exception
        # print("src shape: ", sample['net_input']['src_tokens'].shape)
        # print("target shape: ", sample['target'].shape)
        net_output, _ = model(**sample['net_input']) #bz X tgt-len X output_dimension (vocab size)
        bz = net_output.size(0)
        vocab_size = net_output.size(2)
        softmax_output = self.gumbel_softmax(net_output, hard=True)

        index_convert = torch.ones(vocab_size)
        index_convert[0] = 0
        index_convert = torch.cumsum(index_convert, 0)
        softmax_output = softmax_output @ index_convert
        softmax_output = softmax_output.long() # convert softmax prediction into token format
        src_tokens = sample['net_input']['src_tokens']
        softmax_output = torch.argmax(net_output, dim=2)  # discrete tokenizer
        # wrap up data for prediction
        prediction_data = {'src': src_tokens,
                           'mt': softmax_output, # bz X tgt-len
                           'ref': sample['target']}

        encoder = model.encoder
        embed = encoder.embed_scale * encoder.embed_tokens(src_tokens)
        # print(embed.shape) # bz X src-len X embed-dim=1024
        # source_data = {'src': src_tokens,
        #                 'mt': embed,
        #                 'ref': src_tokens}

        prediction_score = comet_model.predict_vector(prediction_data, cuda=False, batch_size=bz)
        # source_side_score = comet_model.predict_vector(source_data, cuda=False, batch_size=bz)
        # print("scores: ", prediction_score, source_side_score)
        # loss = self.alpha * prediction_score + (1-self.alpha) * (-source_side_score)
        loss = prediction_score
        print(prediction_data)
        print("predcition score: ", prediction_score)
        # convert loss into a scaler (avg the batch loss)
        loss = torch.sum(loss) / bz
        print("loss: ", loss)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': loss / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
