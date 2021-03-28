import math

from fairseq import utils, bleu
from . import FairseqCriterion, register_criterion
import torch.nn.functional as F
import torch
import math
import numpy as np

@register_criterion('mrt_bleu')
class MRTBLEU(FairseqCriterion):
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
        max_len = 200

        # perform sub sample search
        out_indice = src_tokens.new_full(
            (self.k, bsz, max_len+2),
            pad
        )
        out_prob = src_tokens.new_full(
            (self.k, bsz),
            0.0
        )
        for i in range(self.k):
            tokens = src_tokens.new_full((bsz, max_len + 2), pad)
            tokens[:, 0] = eos
            sentence_prob = src_tokens.new_full((bsz, 1), 1)
            # used for finished sentence
            is_decoding = src_tokens.new_ones(bsz).bool()
            for step in range(max_len + 1):
                if is_decoding.sum() == 0:
                    break
                lprob, avg_att = model.forward_decoder(tokens[:, :step + 1], encoder_out=encoder_outs, temperature=1)
                lprob = lprob[:, -1, :]
                lprob[:, pad] = -math.inf
                # apply softmax because probability needs to be tracked
                lprob = torch.softmax(lprob, dim=1)
                new_token = torch.multinomial(lprob, 1) # bz x 1, need to squeeze later
                # retrieve the probability
                prob = torch.gather(lprob, 1, new_token)
                # pad the already finished sentence and fix the prob
                new_token = new_token.squeeze().masked_fill_(
                    ~is_decoding,
                    pad
                )
                prob[~is_decoding] = 1
                # TODO: prob accumulated is too small, maybe replaced by log operation/division by small number
                sentence_prob = sentence_prob * prob
                tokens[:, step + 1] = new_token
                # Update is_decoding flag.
                is_decoding = is_decoding * torch.ne(new_token, eos)
            # add #bsz inferenced sample into output, along with their prob

            out_indice[i, :, :] = tokens
            out_prob[i, :] = sentence_prob.squeeze(-1)
        # k x bz x len => bz x k x len
        out_indice = out_indice.permute(1, 0, 2)
        # k x bz => bz x k
        out_prob = out_prob.permute(1, 0)

        return out_indice, out_prob

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # generate a sub sample space of size k
        indice, prob = self.subsample(model, sample)
        bz = prob.size(0)
        prob = prob * self.alpha
        denom = torch.sum(prob, dim=1, keepdim=True)
        # print(prob, denom)
        Q = torch.div(prob, denom) # have nan for Q because value too small

        # convert indices into sentence and compute score using sacrebleu
        scorer = bleu.SacrebleuScorer()
        all_score = prob.new_full((bz, self.k), 0)
        # TODO: use two for loop, might be able to speed up with parallelization?
        for batch in range(bz):
            scorer.reset()
            tgt_token = utils.strip_pad(sample['target'][batch, :], self.task.target_dictionary.pad()).int()
            tgt_sent = self.task.target_dictionary.string(tgt_token, "sentencepiece", escape_unk=True)
            # print(tgt_sent)
            for j in range(self.k):
                sys_token = indice[batch, j]
                sys_token = utils.strip_pad(sys_token, self.task.target_dictionary.pad()).int()
                sys_sent = self.task.target_dictionary.string(sys_token, "sentencepiece", escape_unk=True)
                # print(sys_sent)
                scorer.add_string(tgt_sent, sys_sent)
                bleu_score = scorer.score()
                all_score[batch, j] = bleu_score
            print()

        # print(all_score)

        # compute risk and perform backprop
        risk = torch.sum(Q * all_score, 1)
        loss = 1 - risk.mean()
        print(risk, loss)

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
