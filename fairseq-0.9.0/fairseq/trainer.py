# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""

import contextlib
import math
import os
import sys
from collections import OrderedDict
from itertools import chain

import torch
from fairseq import checkpoint_utils, distributed_utils, models, optim, utils
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.optim import lr_scheduler
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.models import transformer


class Trainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, criterion, dummy_batch=None, oom_batch=None):
        self.args = args
        self.task = task

        # copy model and criterion to current device
        self._criterion = criterion
        self._model = model
        self.cuda = torch.cuda.is_available() and not args.cpu
        if args.fp16:
            self._criterion = self._criterion.half()
            self._model = self._model.half()
        if self.cuda:
            self._criterion = self._criterion.cuda()
            self._model = self._model.cuda()

        self._dummy_batch = dummy_batch
        self._oom_batch = oom_batch or dummy_batch

        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._prev_grad_norm = None
        self._wrapped_criterion = None
        self._wrapped_model = None

        # Fast stats sync avoids memcpy and is 7% faster when tested on 16 nodes.
        # It is less flexible and syncs only the default stats.
        self._all_reduce_list = [0.0] * 6
        self.fast_stat_sync = args.fast_stat_sync
        self.init_meters(args)

    def init_meters(self, args):
        self.meters = OrderedDict()
        self.meters["train_loss"] = AverageMeter()
        self.meters["train_nll_loss"] = AverageMeter()
        self.meters["valid_loss"] = AverageMeter()
        self.meters["valid_nll_loss"] = AverageMeter()
        self.meters["wps"] = TimeMeter()  # words per second
        self.meters["ups"] = TimeMeter()  # updates per second
        self.meters["wpb"] = AverageMeter()  # words per batch
        self.meters["bsz"] = AverageMeter()  # sentences per batch
        self.meters["gnorm"] = AverageMeter()  # gradient norm
        self.meters["clip"] = AverageMeter()  # % of updates clipped
        self.meters["oom"] = AverageMeter()  # out of memory
        if args.fp16:
            self.meters["loss_scale"] = AverageMeter()  # dynamic loss scale
        self.meters["wall"] = TimeMeter()  # wall time in seconds
        self.meters["train_wall"] = StopwatchMeter()  # train wall time in seconds

    @property
    def criterion(self):
        if self._wrapped_criterion is None:
            if (
                utils.has_parameters(self._criterion)
                and self.args.distributed_world_size > 1
                and not self.args.use_bmuf
            ):
                self._wrapped_criterion = models.DistributedFairseqModel(
                    self.args, self._criterion
                )
            else:
                self._wrapped_criterion = self._criterion
        return self._wrapped_criterion

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.args.distributed_world_size > 1 and not self.args.use_bmuf:
                self._wrapped_model = models.DistributedFairseqModel(
                    self.args, self._model
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):

        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters(), self.criterion.parameters()),
            )
        )
        # if we train on adversarial data, we need 2 optimizer, one for embed only another for whole model
        if self.args.on_the_fly_train:
            params_adv = list(
                filter(
                    lambda p: True,
                    chain(self.model.parameters(), self.criterion.parameters()),
                )
            )
            self._optimizer2 = optim.build_optimizer(self.args, params_adv)

        if self.args.fp16:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                print(
                    "| WARNING: your device does NOT support faster training with --fp16, "
                    "please switch to FP32 which is likely to be faster"
                )
            if self.args.memory_efficient_fp16:
                self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(
                    self.args, params
                )
            else:
                self._optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            # if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
            #     print("| NOTICE: your device may support faster training with --fp16")
            self._optimizer = optim.build_optimizer(self.args, params)

        if self.args.use_bmuf:
            self._optimizer = optim.FairseqBMUF(self.args, self._optimizer)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        if self.args.on_the_fly_train:
            self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self._optimizer2)
        else:
            self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)
        # self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)
        self._lr_scheduler.step_update(0)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            extra_state["train_meters"] = self.meters
            checkpoint_utils.save_state(
                filename,
                self.args,
                self.get_model().state_dict(),
                self.get_criterion(),
                self.optimizer,
                self.lr_scheduler,
                self.get_num_updates(),
                self._optim_history,
                extra_state,
            )

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = None, [], None

        try:
            from fairseq.fb_pathmgr import fb_pathmgr

            bexists = fb_pathmgr.isfile(filename)
        except (ModuleNotFoundError, ImportError):
            bexists = os.path.exists(filename)


        def print_model(model):
            displayed=False
            for en_dec in model.children():
                # en_dec is encoder or decoder, they share the embedding in my construction
                # keep embedding requires_grad
                for m in en_dec.children():
                    # print(m)
                    if isinstance(m, torch.nn.SinusoidalPositionalEmbedding):
                        for p in m.parameters():
                            print(p.data)
                        break
                print("####################################33")

        if bexists:
            state = checkpoint_utils.load_checkpoint_to_cpu(filename)
            # revise the state for training on the opposite direction
            # load model parameters
            if self.args.dual_model_pretrain:
                from collections import OrderedDict
                # only keep the encoder embedding of loaded model
                encoder_emb = state['model']['encoder.embed_tokens.weight']
                state['model'] = OrderedDict()
                # use encoder embed to update decoder embed
                state['model']['decoder.embed_tokens.weight'] = encoder_emb
            try:
                isStrict = False if self.args.dual_model_pretrain else True
                self.get_model().load_state_dict(
                    state["model"], strict=isStrict, args=self.args
                )

                if utils.has_parameters(self.get_criterion()):
                    self.get_criterion().load_state_dict(
                        state["criterion"], strict=True
                    )
            except Exception:
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(filename)
                )

            extra_state = state["extra_state"]
            self._optim_history = state["optimizer_history"]
            last_optim_state = state.get("last_optimizer_state", None)

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert (
                last_optim["criterion_name"] == self.get_criterion().__class__.__name__
            ), "Criterion does not match; please reset the optimizer (--reset-optimizer)."
            assert (
                last_optim["optimizer_name"] == self.optimizer.__class__.__name__
            ), "Optimizer does not match; please reset the optimizer (--reset-optimizer)."

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self.set_num_updates(last_optim["num_updates"])

        if extra_state is not None:
            epoch = extra_state["train_iterator"]["epoch"]
            print(
                "| loaded checkpoint {} (epoch {} @ {} updates)".format(
                    filename, epoch, self.get_num_updates()
                )
            )

            self.lr_step(epoch)

            if "train_meters" in extra_state and not reset_meters:
                self.meters.update(extra_state["train_meters"])
                del extra_state["train_meters"]

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in self.meters.values():
                    if isinstance(meter, TimeMeter):
                        meter.reset()
        else:
            print("| no existing checkpoint found {}".format(filename))

        return extra_state

    def get_train_iterator(
        self,
        epoch,
        combine=True,
        load_dataset=True,
        data_selector=None,
        shard_batch_itr=True,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        if load_dataset:
            print("| loading train data for epoch {}".format(epoch))
            self.task.load_dataset(
                self.args.train_subset,
                epoch=epoch,
                combine=combine,
                data_selector=data_selector,
            )
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args.train_subset),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(), self.model.max_positions()
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=self.args.distributed_world_size if shard_batch_itr else 1,
            shard_id=self.args.distributed_rank if shard_batch_itr else 0,
            num_workers=self.args.num_workers,
            epoch=epoch,
        )

    def train_step(self, samples, comet_model=None, aux_model=None, dummy_batch=False, raise_oom=False):
        """Do forward, backward and parameter update."""
        if self._dummy_batch is None:
            self._dummy_batch = samples[0]

        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        if not dummy_batch:
            self.meters["train_wall"].start()

        # forward and backward pass
        logging_outputs, sample_sizes, ooms = [], [], 0
        for i, sample in enumerate(samples):
            sample = self._prepare_sample(sample)
            if sample is None:
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients
                sample = self._prepare_sample(self._dummy_batch)
                ignore_grad = True
            else:
                ignore_grad = False

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.args.distributed_world_size > 1
                    and hasattr(self.model, "no_sync")
                    and i < len(samples) - 1
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            try:
                with maybe_no_sync():
                    # forward and backward

                    # with torch.autograd.set_detect_anomaly(True):

                    if self.args.on_the_fly or self.args.on_the_fly_train:
                        import copy
                        # print(self.model.encoder.embed_tokens.weight[0:5, 0:5])
                        embed_copy = self.model.encoder.embed_tokens.weight.detach().clone()
                        # print('model weight before noise update:', self.model.encoder.embed_tokens.weight.sum())
                        reserved_src_tokens = sample['net_input']['src_tokens']
                        reserved_sample = copy.deepcopy(sample)

                    if self.args.train_baseline:
                        import random
                        import copy
                        random.seed(41)
                        # generate replacement and deletion samples in the same way I used for adversarial approach
                        reserved_sample = copy.deepcopy(sample)
                        embed_copy = self.model.encoder.embed_tokens.weight.detach().clone()
                        model_embed = embed_copy / embed_copy.norm(dim=1, keepdim=True)
                        src_tokens = sample['net_input']['src_tokens']
                        token_embed = embed_copy[src_tokens]
                        token_embed = token_embed / token_embed.norm(dim=2, keepdim=True)
                        adv_sample_prob = token_embed @ torch.transpose(model_embed, 0, 1)
                        _, adv_sample_tokens = adv_sample_prob.topk(3, dim=2)
                        row, col = src_tokens.size(0), src_tokens.size(1)
                        temp = src_tokens.clone()
                        pad_mask = src_tokens.eq(self.task.src_dict.pad())
                        for i in range(row):
                            for j in range(col - 1):
                                if pad_mask[i, j]:
                                    continue
                                else:
                                    if random.random() < self.args.adv_percent * 0.01:
                                        # perform one of 2 actions: delete or replace with closet
                                        if random.random() < self.args.ratio * 0.01:
                                            temp[i, j] = adv_sample_tokens[i, j, 2]
                                        else:
                                            temp[i, j] = adv_sample_tokens[i, j, 0]
                            # adv_token = utils.strip_pad(temp[i, :], self.task.src_dict.pad())
                            # print(self.task.src_dict.string(adv_token, "sentencepiece"))
                        reserved_sample['ntokens'] *= 2
                        reserved_sample['net_input']['src_tokens'] = torch.cat(
                            [reserved_sample['net_input']['src_tokens'], temp], dim=0)
                        reserved_sample['net_input']['src_lengths'] = reserved_sample['net_input'][
                            'src_lengths'].repeat(2)
                        reserved_sample['net_input']['prev_output_tokens'] = reserved_sample['net_input'][
                            'prev_output_tokens'].repeat(2, 1)
                        reserved_sample['target'] = reserved_sample['target'].repeat(2, 1)
                        sample = reserved_sample

                    loss, sample_size, logging_output = self.task.train_step(
                        sample, self.model, self.criterion, self.optimizer, ignore_grad,
                        comet_model=comet_model, aux_model=aux_model
                    )

                if not ignore_grad:
                    logging_outputs.append(logging_output)
                    sample_sizes.append(sample_size)

                    if self.fast_stat_sync:
                        self._all_reduce_list[0] += sample_size
                        self._all_reduce_list[1] += logging_output.get(
                            "nsentences", 0.0
                        )
                        self._all_reduce_list[2] += logging_output.get("loss", 0.0)
                        self._all_reduce_list[3] += logging_output.get("nll_loss", 0.0)
                        self._all_reduce_list[4] += logging_output.get("ntokens", 0.0)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if raise_oom:
                        raise e
                    print(
                        "| WARNING: attempting to recover from OOM in forward/backward pass",
                        file=sys.stderr,
                    )
                    ooms += 1
                    self.zero_grad()
                else:
                    raise e

            if self.fast_stat_sync:
                self._all_reduce_list[5] += ooms

        if ooms > 0 and self._oom_batch is not None:
            self.handle_ooms(ooms)

        if dummy_batch:
            return None

        # gather logging outputs from all replicas
        if self.fast_stat_sync:
            # rework all_gather_list
            all_reduce_list_tensor = torch.cuda.DoubleTensor(self._all_reduce_list)
            if self._sync_stats():
                torch.distributed.all_reduce(all_reduce_list_tensor)
            # Normalize loss and nll_loss by "sample_size"
            # and convert to log base 2
            all_reduce_list_tensor[2:4].div_(
                (all_reduce_list_tensor[0:1] * torch.log(torch.cuda.DoubleTensor([2])))
            )
            self._all_reduce_list = all_reduce_list_tensor.tolist()
            logging_output = {}
            [
                sample_size,
                logging_output["nsentences"],
                logging_output["loss"],
                logging_output["nll_loss"],
                logging_output["ntokens"],
                ooms,
            ] = self._all_reduce_list
        elif self._sync_stats():
            logging_outputs, sample_sizes, ooms, prev_norms = zip(
                *distributed_utils.all_gather_list(
                    [logging_outputs, sample_sizes, ooms, self._prev_grad_norm]
                )
            )
            logging_outputs = list(chain.from_iterable(logging_outputs))
            sample_sizes = list(chain.from_iterable(sample_sizes))
            ooms = sum(ooms)

            if not self.args.use_bmuf:
                assert all(norm == prev_norms[0] for norm in prev_norms) or all(
                    math.isnan(norm) or math.isinf(norm) for norm in prev_norms
                ), "Fatal error: gradients are inconsistent between workers"
        # print(self.fast_stat_sync, self._sync_stats())
        self.meters["oom"].update(ooms, len(samples))
        if ooms == self.args.distributed_world_size * len(samples):
            print("| WARNING: OOM in all workers, skipping update")
            self.zero_grad()
            return None

        if not self.fast_stat_sync:
            # aggregate logging outputs and sample sizes
            logging_output = self.task.aggregate_logging_outputs(
                logging_outputs, self.get_criterion()
            )
            sample_size = self.task.grad_denom(sample_sizes, self.get_criterion())

        if not all(k in logging_output for k in ["ntokens", "nsentences"]):
            raise Exception(
                (
                    "Please update the {}.aggregate_logging_outputs() method to "
                    "return ntokens and nsentences"
                ).format(self.task.__class__.__name__)
            )

        try:
            # normalize grads by sample size
            if sample_size > 0:
                self.optimizer.multiply_grads(
                    self.args.distributed_world_size / float(sample_size)
                )

            # clip grads
            grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)
            self._prev_grad_norm = grad_norm

            # take an optimization step
            # print(self.model.encoder.embed_tokens.weight[0:5, 0:5])
            self.optimizer.step()
            # print(self.model.encoder.embed_tokens.weight[0:5, 0:5])


            # generate adv tokens and train on NLL again
            if self.args.on_the_fly_train:
                import random
                random.seed(41)
                self.zero_grad()
                src_token = reserved_src_tokens
                target_token = sample['target']
                pad_mask = src_token.eq(self.task.src_dict.pad())
                norm_embed_copy = embed_copy / embed_copy.norm(dim=1, keepdim=True)
                token_embed = self.model.encoder.embed_tokens.weight[src_token]
                token_embed = token_embed / token_embed.norm(dim=2, keepdim=True)
                adv_sample_prob = token_embed @ torch.transpose(norm_embed_copy, 0, 1)
                _, adv_sample_tokens = adv_sample_prob.topk(3, dim=2)
                row, col = src_token.size(0), src_token.size(1)
                temp = src_token.clone()
                for i in range(row):
                    for j in range(col - 1):
                        if pad_mask[i, j]:
                            continue
                        else:
                            if random.random() < self.args.adv_percent * 0.01:
                                # perform one of 2 actions: delete or replace with closet
                                if random.random() < self.args.ratio * 0.01:
                                    temp[i, j] = adv_sample_tokens[i, j, 2]
                                else:
                                    temp[i, j] = adv_sample_tokens[i, j, 0]

                    adv_token = utils.strip_pad(temp[i, :], self.task.src_dict.pad())
                    # print(self.task.src_dict.string(adv_token, "sentencepiece"))
                # undo the gradient update
                self.model.encoder.embed_tokens.weight.data.copy_(embed_copy)

                # print(reserved_sample)
                # print(temp)
                # prepare new sample for nll loss:
                reserved_sample['ntokens'] *= 2
                reserved_sample['net_input']['src_tokens'] = torch.cat([reserved_sample['net_input']['src_tokens'], temp], dim=0)
                reserved_sample['net_input']['src_lengths'] = reserved_sample['net_input']['src_lengths'].repeat(2)
                reserved_sample['net_input']['prev_output_tokens'] = reserved_sample['net_input']['prev_output_tokens'].repeat(2, 1)
                reserved_sample['target'] = reserved_sample['target'].repeat(2, 1)
                # compute NLL loss following label-smoothed xent:
                # update requires grad to true so that whole model is updated:
                for m in self.model.children():
                    for p in m.parameters():
                        p.requires_grad = True

                self.args.label_smoothing = 0.1 # manual append the required additional param, hacky
                xent = LabelSmoothedCrossEntropyCriterion(self.args, self.task)
                loss, sample_size, logging_output = xent(self.model, reserved_sample)
                self._optimizer2.backward(loss)
                self._optimizer2.multiply_grads(
                    self.args.distributed_world_size / float(sample_size)
                )
                # clip grads
                grad_norm = self._optimizer2.clip_grad_norm(self.args.clip_norm)
                self._prev_grad_norm = grad_norm
                # take an optimization step
                # print(self.model.encoder.embed_tokens.weight[0:5, 0:5])
                self._optimizer2.step()
                # print(self.model.encoder.embed_tokens.weight[0:5, 0:5])

                # update requires_grad back to False so that next loop can be executed correctly
                for en_dec in self.model.children():
                    for m in en_dec.children():
                        if isinstance(m, torch.nn.Embedding) and isinstance(en_dec, transformer.TransformerEncoder):
                            continue
                        for p in m.parameters():
                            p.requires_grad = False
                aux_model.decoder.embed_tokens.weight = self.model.encoder.embed_tokens.weight

            self.set_num_updates(self.get_num_updates() + 1)

            # task specific update per step
            self.task.update_step(self._num_updates)
            if self.args.on_the_fly:
                import random
                random.seed(41)
                src_token = reserved_src_tokens
                target_token = sample['target']
                pad_mask = src_token.eq(self.task.src_dict.pad())
                norm_embed_copy = embed_copy / embed_copy.norm(dim=1, keepdim=True)
                token_embed = self.model.encoder.embed_tokens.weight[src_token]
                token_embed = token_embed / token_embed.norm(dim=2, keepdim=True)
                adv_sample_prob = token_embed @ torch.transpose(norm_embed_copy, 0, 1)
                _, adv_sample_tokens = adv_sample_prob.topk(3, dim=2)
                row, col = src_token.size(0), src_token.size(1)
                temp = src_token.clone()
                for i in range(row):
                    for j in range(col-1):
                        if pad_mask[i, j]:
                            continue
                        else:
                            if random.random() < self.args.adv_percent * 0.01:
                                temp[i, j] = adv_sample_tokens[i, j, 2]
                    adv_token = utils.strip_pad(temp[i, :], self.task.src_dict.pad())
                    token = utils.strip_pad(src_token[i, :], self.task.src_dict.pad())
                    # print(self.task.src_dict.string(token, "sentencepiece"))
                    target_token_temp = utils.strip_pad(target_token[i, :], self.task.tgt_dict.pad())
                    src_file = open(self.args.src_file, 'a')
                    tgt_file = open(self.args.tgt_file, 'a')
                    print(self.task.tgt_dict.string(target_token_temp, "sentencepiece"), file=tgt_file)
                    print(self.task.src_dict.string(adv_token, "sentencepiece"), file=src_file)
                    # print(self.task.tgt_dict.string(target_token, "sentencepiece"))
                    # print(self.task.src_dict.string(adv_token, "sentencepiece"))
                    src_file.close()
                    tgt_file.close()
                # undo the gradient update

                self.model.encoder.embed_tokens.weight = torch.nn.Parameter(embed_copy)
                aux_model.decoder.embed_tokens.weight = torch.nn.Parameter(embed_copy)

            # update meters
            ntokens = logging_output.get("ntokens", 0)
            nsentences = logging_output.get("nsentences", 0)
            self.meters["wps"].update(ntokens)
            self.meters["ups"].update(1.0)
            self.meters["wpb"].update(ntokens)
            self.meters["bsz"].update(nsentences)
            self.meters["gnorm"].update(grad_norm)
            self.meters["clip"].update(
                1.0
                if grad_norm > self.args.clip_norm and self.args.clip_norm > 0
                else 0.0
            )
            self.meters["train_loss"].update(logging_output.get("loss", 0), sample_size)
            if "train_acc" in self.meters:
                self.meters["train_acc"].update(
                    logging_output.get("acc", 0), sample_size
                )

            if "nll_loss" in logging_output:
                self.meters["train_nll_loss"].update(
                    logging_output.get("nll_loss", 0), ntokens
                )

            # clear CUDA cache to reduce memory fragmentation
            if (
                self.args.empty_cache_freq > 0
                and (
                    (self.get_num_updates() + self.args.empty_cache_freq - 1)
                    % self.args.empty_cache_freq
                )
                == 0
                and torch.cuda.is_available()
                and not self.args.cpu
            ):
                torch.cuda.empty_cache()
        except OverflowError as e:
            print("| WARNING: overflow detected, " + str(e))
            self.zero_grad()
            logging_output = None
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._log_oom(e)
                print("| ERROR: OOM during optimization, irrecoverable")
            raise e

        if self.args.fp16:
            self.meters["loss_scale"].reset()
            self.meters["loss_scale"].update(self.optimizer.scaler.loss_scale)

        self.clear_buffered_stats()
        self.meters["train_wall"].stop()

        return logging_output

    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            sample = self._prepare_sample(sample)
            if sample is None:
                sample = self._prepare_sample(self._dummy_batch)
                ignore_results = True
            else:
                ignore_results = False

            try:
                xent = LabelSmoothedCrossEntropyCriterion(self.args, self.task)
                _loss, sample_size, logging_output = self.task.valid_step(
                    sample, self.model, self.criterion, xent=xent
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if not raise_oom:
                        print(
                            "| WARNING: ran out of memory in validation step, retrying batch"
                        )
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = None  # free some memory
                        if self.cuda:
                            torch.cuda.empty_cache()
                        return self.valid_step(sample, raise_oom=True)
                raise e

            if ignore_results:
                logging_output, sample_size = {}, 0

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_output, sample_size = zip(
                *distributed_utils.all_gather_list([logging_output, sample_size])
            )
            logging_output = list(logging_output)
            sample_size = list(sample_size)
        else:
            logging_output = [logging_output]
            sample_size = [sample_size]

        # aggregate logging outputs and sample sizes
        logging_output = self.task.aggregate_logging_outputs(
            logging_output, self.get_criterion()
        )
        sample_size = self.task.grad_denom(sample_size, self.get_criterion())

        # update meters for validation
        ntokens = logging_output.get("ntokens", 0)
        self.meters["valid_loss"].update(logging_output.get("loss", 0), sample_size)
        if "valid_acc" in self.meters:
            self.meters["valid_acc"].update(logging_output.get("acc", 0), sample_size)

        if "nll_loss" in logging_output:
            self.meters["valid_nll_loss"].update(
                logging_output.get("nll_loss", 0), ntokens
            )

        return logging_output

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, dummy_batch=True)
        self.zero_grad()

    def handle_ooms(self, number_of_ooms):
        """
        c10d accumulates/syncs gradients between gpus during backward pass.
        In case of OOMs, gpus may fail to sync, so we manually iterate
        extra to make sure each gpu makes same number of iterations.
        """
        for _ in range(number_of_ooms):
            self.train_step([self._oom_batch], True)

    def zero_grad(self):
        self.optimizer.zero_grad()
        try:
            self._optimizer2.zero_grad()
        except Exception:
            pass


    def clear_buffered_stats(self):
        self._all_reduce_list = [0.0] * 6

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(self.get_num_updates())

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_criterion(self):
        """Get the (non-wrapped) criterion instance."""
        return self._criterion

    def get_meter(self, name):
        """Get a specific meter by name."""
        if name not in self.meters:
            return None
        return self.meters[name]

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None

        if self.cuda:
            sample = utils.move_to_cuda(sample)

        def apply_half(t):
            if t.dtype is torch.float32:
                return t.half()
            return t

        if self.args.fp16:
            sample = utils.apply_to_sample(apply_half, sample)

        return sample

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)

    def _sync_stats(self):
        return self.args.distributed_world_size > 1 and (
            (not self.args.use_bmuf)
            or (
                self.args.use_bmuf
                and (self.get_num_updates() + 1) % self.args.global_sync_iter == 0
            )
        )

    def _log_oom(self, exc):
        msg = "| OOM: Ran out of memory with exception: {}".format(exc)
        # TODO: print should really go to logger, this print goes
        # to stderr, which is buffered, which in many cases is not
        # printed out if another exception happens.
        # NB(jerry): added a flush to mitigate this
        print(msg, file=sys.stderr)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                print(torch.cuda.memory_summary(device=device_idx), file=sys.stderr)
        sys.stderr.flush()
