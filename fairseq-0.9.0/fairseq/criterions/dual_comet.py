import math

from fairseq import utils, bleu
from . import FairseqCriterion, register_criterion
from fairseq.sequence_generator import SequenceGenerator
import torch.nn.functional as F
import torch
import math
import numpy as np
import sacrebleu

@register_criterion('dual_comet')
class DualCOMET(FairseqCriterion):
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

    def get_auxillary_sample(self, sample):
        eos = self.task.target_dictionary.eos()
        pad = self.task.target_dictionary.pad()
        src_tokens = sample['net_input']['src_tokens']
        bz = src_tokens.size(0)
        src_len = src_tokens.size(1)
        aux_sample = {'target': src_tokens}
        eos_indice = (src_tokens == eos).nonzero(as_tuple=True)[1]
        prev_output_tokens = src_tokens.roll(1, 1)
        prev_output_tokens[:, 0] = eos
        for i in range(bz):
            if eos_indice[i] < src_len - 1:
                # need to update the eos to pad
                prev_output_tokens[i, eos_indice[i]+1] = pad
        net_input = {'src_tokens': sample['target'], 'prev_output_tokens': prev_output_tokens}
        aux_sample['net_input'] = net_input
        return aux_sample


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

    def compute_Q(self, model, input, eos_indice, prev_output_token, bz, tgt_size, sample):
        eos = self.task.target_dictionary.eos()
        pad = self.task.target_dictionary.pad()

        net_output = model(**input['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)

        custom_target = prev_output_token.roll(-1, 1)  # shift to left, along dim=1
        custom_target[:, -1] = pad
        eos_indice = eos_indice.reshape(-1, 1)
        for i in range(bz * self.k):
            if (i+1) % self.k == 0:
                batch_num = int((i+1)/self.k - 1)
                # gold sentence case
                pad_sent = custom_target.new_ones(tgt_size - sample['target'].size(1)) * pad
                custom_target[i, :] = torch.cat((sample['target'][batch_num], pad_sent))
            else:
                custom_target[i, eos_indice[i]] = eos

        sent_prob = lprobs.gather(-1, custom_target.unsqueeze(-1)).squeeze()
        non_pad_mask = custom_target.ne(pad)
        sent_prob = sent_prob * non_pad_mask
        sent_prob = sent_prob.sum(dim=-1).view(bz, self.k)
        sent_prob = sent_prob * self.alpha
        Q = sent_prob.exp() / sent_prob.exp().sum(1, keepdim=True)
        return Q

    def compute_bleu(self, bz, Q, sample, isZh, indice):
        scorer = bleu.SacrebleuScorer()
        all_score = Q.new_full((bz, self.k), 0)

        # TODO: use two for loop, might be able to speed up with parallelization?
        for batch in range(bz):
            scorer.reset()
            tgt_token = utils.strip_pad(sample['target'][batch, :], self.task.target_dictionary.pad()).int()
            tgt_sent = self.task.target_dictionary.string(tgt_token, "sentencepiece", escape_unk=True)
            if isZh:
                tok = sacrebleu.tokenizers.TokenizerZh()
                tgt_sent = tok(tgt_sent)
                # print(tgt_sent)
            for j in range(self.k):
                scorer.reset()
                sys_token = indice[batch, j]
                sys_token = utils.strip_pad(sys_token, self.task.target_dictionary.pad()).int()
                sys_sent = self.task.target_dictionary.string(sys_token, "sentencepiece", escape_unk=True)

                if isZh:
                    tok = sacrebleu.tokenizers.TokenizerZh()
                    sys_sent = tok(sys_sent)
                scorer.add_string(tgt_sent, sys_sent)
                bleu_score = scorer.score()
                all_score[batch, j] = bleu_score
        # print()
        return all_score

    def forward(self, model, sample, aux_model, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # # test generator
        # generator = SequenceGenerator(self.task.target_dictionary,
        #                               sampling=False)
        # with torch.no_grad():
        #     greturn = generator.generate([model], sample)
        #
        # for item in greturn[0]:
        #     token = item['tokens']
        #     print(token)
        #     sys_token = utils.strip_pad(token, self.task.target_dictionary.pad()).int()
        #     sys_sent = self.task.target_dictionary.string(sys_token, "sentencepiece", escape_unk=True)
        #     print(sys_sent)

        aux_model = aux_model.cuda()
        eos = self.task.target_dictionary.eos()
        pad = self.task.target_dictionary.pad()

        # construct auxillary sample for auxillary model
        aux_sample = self.get_auxillary_sample(sample)

        with torch.no_grad():
            indice, eos_indice = self.subsample(model, sample)
            aux_indice, aux_eos_indice = self.subsample(aux_model, aux_sample)

        model.train()
        aux_model.train()
        bz = indice.size(0)
        tgt_size = indice.size(2)
        aux_tgt_size = aux_indice.size(2)

        prev_output_token = indice.reshape(-1, indice.size(2))
        aux_prev_output_token = aux_indice.reshape(-1, aux_indice.size(2))

        custom_input = sample
        custom_input['net_input']['src_tokens'] = sample['net_input']['src_tokens'].repeat_interleave(self.k, dim=0)
        custom_input['net_input']['src_lengths'] = sample['net_input']['src_lengths'].repeat_interleave(self.k, dim=0)
        custom_input['net_input']['prev_output_tokens'] = prev_output_token

        aux_custom_input = aux_sample
        aux_src_token = aux_sample['net_input']['src_tokens']
        aux_custom_input['net_input']['src_tokens'] = aux_sample['net_input']['src_tokens'].repeat_interleave(self.k, dim=0)
        aux_len = (aux_src_token.ne(eos) & aux_src_token.ne(pad)).long().sum(dim=1)

        aux_custom_input['net_input']['src_lengths'] = aux_len.repeat_interleave(self.k, dim=0)
        aux_custom_input['net_input']['prev_output_tokens'] = aux_prev_output_token

        Q = self.compute_Q(model, custom_input, eos_indice, prev_output_token, bz, tgt_size, sample)
        aux_Q = self.compute_Q(aux_model, aux_custom_input, aux_eos_indice, aux_prev_output_token, bz, aux_tgt_size, aux_sample)
        # print(Q, aux_Q)

        score = self.compute_bleu(bz, Q, sample, False, indice)
        aux_score = self.compute_bleu(bz, aux_Q, aux_sample, True, aux_indice)
        # print(score, aux_score)

        # minimize Qscore, maximize auxQscore
        beta = 0.8
        loss = beta * torch.sum(Q * score) + (1-beta) * torch.sum((1-aux_Q)*aux_score)

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
