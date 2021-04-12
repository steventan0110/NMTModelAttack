import math

from fairseq import utils, bleu
from . import FairseqCriterion, register_criterion
import torch.nn.functional as F
import torch
import math
import numpy as np
import sacrebleu


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    # temp = nll_loss.view(2, -1)
    # print(temp.sum(dim=1))
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@register_criterion('dual_nll')
class DualNLL(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

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

    def forward(self, model, sample, aux_model, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        aux_model = aux_model.cuda()
        eos = self.task.target_dictionary.eos()
        pad = self.task.target_dictionary.pad()

        # construct auxillary sample for auxillary model
        aux_sample = self.get_auxillary_sample(sample)
        aux_src_token = aux_sample['net_input']['src_tokens']
        aux_len = (aux_src_token.ne(eos) & aux_src_token.ne(pad)).long().sum(dim=1)
        aux_sample['net_input']['src_lengths'] = aux_len

        model.train()
        aux_model.train()
        net_output = model(**sample['net_input'])
        aux_output = aux_model(**aux_sample['net_input'])
        sample_loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        aux_loss, _ = self.compute_loss(aux_model, aux_output, aux_sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        beta = 0.8
        loss = (1-beta) * aux_loss - beta * sample_loss
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        # print(lprobs.shape)
        # print(target)
        # raise Exception()
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

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
