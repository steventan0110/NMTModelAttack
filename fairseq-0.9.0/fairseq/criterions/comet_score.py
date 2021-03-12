import math

from fairseq import utils
from COMET.comet.models import download_model, load_checkpoint
from . import FairseqCriterion, register_criterion
import torch.nn.functional as F

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

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        print("src shape: ", sample['net_input']['src_tokens'].shape)
        print("target shape: ", sample['target'].shape)
        net_output, _ = model(**sample['net_input']) #bz X tgt-len X output_dimension (vocab size)
        bz = net_output.size(0)
        softmax_output = self.gumbel_softmax(net_output)
        src_tokens = sample['net_input']['src_tokens']

        # wrap up data for prediction
        prediction_data = {'src': src_tokens,
                           'mt': softmax_output,
                           'ref': sample['target']}

        encoder = model.encoder
        embed = encoder.embed_scale * encoder.embed_tokens(src_tokens)
        # print(embed.shape) # bz X src-len X embed-dim=1024
        source_data = {'src': src_tokens,
                        'mt': embed,
                        'ref': src_tokens}

        # pre-downloaded estimator from COMET
        route = "/home/steven/.cache/torch/unbabel_comet/wmt-large-da-estimator-1719/_ckpt_epoch_1.ckpt"
        model = load_checkpoint(route)
        prediction_score = model.predict_vector(prediction_data, cuda=False, batch_size=bz)

        source_side_score = model.predict_vector(source_data, cuda=False, batch_size=bz)



        loss = self.alpha * prediction_score + (1-self.alpha) * source_side_score
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
