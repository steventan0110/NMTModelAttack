#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch
import warnings
from mosestokenizer import *
from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from comet.models import load_checkpoint
warnings.filterwarnings('ignore')

def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    if args.adv_gen:
        # load zh-en model and use its src side embedding to generate adversarial samples
        aux_state = checkpoint_utils.load_checkpoint_to_cpu(args.adv_model_path)
        adv_embed = aux_state['model']['encoder.embed_tokens.weight']
        if use_cuda:
            adv_embed = adv_embed.cuda()

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True

    if args.adv_gen and not args.adv_test:
        src_file = open(args.src_file, 'w')
        tgt_file = open(args.tgt_file, 'w')

    if args.comet_score:
        # load comet model file to compute the score
        comet_route = args.comet_route
        comet_model = load_checkpoint(comet_route)
        src_sent = []
        tgt_sent = []
        hypo_sent = []

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        if args.adv_gen:
            model_embed = model.encoder.embed_tokens.weight
            model_embed = model_embed / model_embed.norm(dim=1, keepdim=True)

        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            if args.adv_gen:
                import random
                random.seed(41)
                src_token = sample['net_input']['src_tokens']
                pad_mask = src_token.eq(src_dict.pad())

                # print(src_str)
                # retrieve the updated embedding
                embed = adv_embed[src_token]
                embed = embed / embed.norm(dim=2, keepdim=True)
                adv_sample_prob = embed  @ torch.transpose(model_embed, 0, 1)

                ##########################
                prob = args.adv_percent * 0.01
                _, adv_sample_tokens = adv_sample_prob.topk(3, dim=2)
                # adv_sample_tokens = adv_sample_prob.argmax(dim=2)
                temp = src_token
                row, col = src_token.size(0), src_token.size(1)
                # for i in range(row):
                #     print(src_token[i, :])
                #     adv_token = src_token[i, 0:1]
                #     print(src_dict.string(adv_token, args.remove_bpe))
                #     print(adv_sample_tokens[i, :, 2])
                #     print(src_dict.string(adv_sample_tokens[i, 0:1, 2], args.remove_bpe))

                for i in range(row):
                    for j in range(col-1):
                        if pad_mask[i, j]:
                            continue
                        else:
                            # if adv_sample_tokens[i, j, 1] != src_token[i, j]:
                            #     print('different from itself')
                            if random.random() < prob:
                                # perturbe the word
                                temp[i, j] = adv_sample_tokens[i, j, 0]

                ########################
                if args.adv_test:
                    sample['net_input']['src_tokens'] = temp
                else:
                    target_token = sample['target']
                    for i in range(row):
                        token = utils.strip_pad(target_token[i, :], tgt_dict.pad())
                        tgt_str = tgt_dict.string(token, args.remove_bpe)

                        adv_token = utils.strip_pad(temp[i, :], src_dict.pad())
                        adv_str = src_dict.string(adv_token, args.remove_bpe)
                        # print(adv_str, file=adv_output)
                        print(tgt_str, file=tgt_file)
                        print(adv_str, file=src_file)

                    continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()

            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                if args.detokenize_moses:
                    src_l = args.source_lang
                    ref_l = args.target_lang
                    with MosesDetokenizer(ref_l) as detokenize:
                        target_str = detokenize(target_str.split(' '))
                    with MosesDetokenizer(src_l) as detokenize:
                        src_str = detokenize(src_str.split(' '))

                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))


                # Process top predictions
                for j, hypo in enumerate(hypos[i][:args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    if args.detokenize_moses:
                        with MosesDetokenizer(ref_l) as detokenize:
                            hypo_str = detokenize(hypo_str.split(' '))

                    if args.comet_score:
                        src_sent.append(src_str)
                        tgt_sent.append(target_str)
                        hypo_sent.append(hypo_str)

                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))

                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                            ))

                        if args.print_step:
                            print('I-{}\t{}'.format(sample_id, hypo['steps']))

                        if getattr(args, 'retain_iter_history', False):
                            print("\n".join([
                                    'E-{}_{}\t{}'.format(
                                        sample_id, step,
                                        utils.post_process_prediction(
                                            h['tokens'].int().cpu(),
                                            src_str, None, None, tgt_dict, None)[1])
                                        for step, h in enumerate(hypo['history'])]))

                    # Score only the top hypothesis
                    if has_target and j == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        if hasattr(scorer, 'add_string'):
                            if args.target_lang == 'zh':
                                # tokenize chinese sentence
                                import sacrebleu
                                tok = sacrebleu.tokenizers.TokenizerZh()
                                target_str = tok(target_str)
                                hypo_str = tok(hypo_str)
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.score()))
    if args.comet_score:
        data = {"src": src_sent, "mt": hypo_sent, "ref": tgt_sent}
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        prediction = comet_model.predict(data, cuda=True, show_progress=True)
        total_score = 0
        size = 0
        for idx, item in enumerate(prediction[0]):
            score = item['predicted_score']
            # print("Score for {}th sentence is {}".format(idx, score))
            total_score += float(score)
            size += 1
        print('| Generate {} with beam={}, comet score: {}'.format(args.gen_subset, args.beam, total_score/size))
    return scorer


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
