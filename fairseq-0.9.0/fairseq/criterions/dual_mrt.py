import math

from fairseq import utils, bleu
from . import FairseqCriterion, register_criterion
import torch.nn.functional as F
import torch
import math
import numpy as np
import sacrebleu

@register_criterion('dual_mrt')
class DualMRT(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.alpha = args.mrt_alpha
        self.k = args.mrt_k

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # sharpness value for MRT, according to the paper 5e-3 is good hyper param
        parser.add_argument('--mrt-alpha', default=5e-3, type=float, metavar='D',
                            help="alpha value for MRT sharpness")
        parser.add_argument('--mrt-k', default=50, type=int, metavar='D',
                            help="k value for MRT sub sample size")

    def subsample(self, model, sample):
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
        encoder_outs = model.encoder(src_tokens, src_lengths=src_length)

        # max_len = model.max_decoder_positions() - 1
        max_len = sample['net_input']['prev_output_tokens'].size(-1) + 5

        # perform sub sample search
        out_indice = src_tokens.new_full(
            (self.k, bsz, max_len+2),
            pad
        )
        num_subsample = 0
        eos_indice = src_tokens.new_full(
            (self.k, bsz),
            0
        )
        while num_subsample < self.k - 1:
            tokens = src_tokens.new_full((bsz, max_len + 2), pad)
            tokens[:, 0] = eos
            sentence_prob = src_tokens.new_full((bsz, 1), 0.0)
            # used for finished sentence
            is_decoding = src_tokens.new_ones(bsz).bool()
            end_indice = src_tokens.new_zeros(bsz)
            for step in range(max_len + 1):
                if is_decoding.sum() == 0:
                    break
                lprob, avg_att = model.forward_decoder(tokens[:, :step + 1], encoder_out=encoder_outs, temperature=1)
                lprob = lprob[:, -1, :]
                lprob[:, pad] = -math.inf
                # apply softmax because probability needs to be tracked
                lprob = torch.softmax(lprob, dim=1)
                # new_token = torch.argmax(lprob, 1).squeeze(-1)  # greedy search, used for checking model performance
                new_token = torch.multinomial(lprob, 1).squeeze(-1) # bz x 1, need to squeeze later
                new_token = new_token.masked_fill_(
                    ~is_decoding,
                    pad
                )
                is_decoding = is_decoding * torch.ne(new_token, eos)
                end_indice += step * (~torch.ne(new_token, eos))
                tokens[:, step + 1] = new_token

            # check duplication
            hasDuplicate = False
            for i in range(out_indice.size(0)):
                if torch.equal(out_indice[i, :, :], tokens):
                    hasDuplicate = True
                    break
            if not hasDuplicate:
                for bz in range(bsz):
                    tokens[bz, end_indice[bz]+1] = pad
                out_indice[num_subsample, :, :] = tokens
                eos_indice[num_subsample, :] = end_indice
                # out_prob[num_subsample, :] = sentence_prob.squeeze(-1)
                num_subsample += 1
        # add in the gold translation
        pad_len = out_indice.size(2) - sample['net_input']['prev_output_tokens'].size(1)
        end_of_sent = out_indice.new_full((bsz, 1), eos)
        padding = out_indice.new_full((bsz, pad_len), pad)
        out_indice[self.k-1, :, :] = torch.cat((sample['net_input']['prev_output_tokens'], padding), dim=1)
        eos_indice[self.k-1, :] = src_tokens.new_zeros(bsz)
        # k x bz x len => bz x k x len
        out_indice = out_indice.permute(1, 0, 2)
        # k x bz => bz x k
        # out_prob = out_prob.permute(1, 0)
        eos_indice = eos_indice.permute(1, 0) # bz X k
        return out_indice, eos_indice

    def forward(self, model, sample, aux_model, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        eos = self.task.target_dictionary.eos()
        pad = self.task.target_dictionary.pad()


        raise Exception

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
