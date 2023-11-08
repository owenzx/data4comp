import os
import math
import random
import re
import datetime
import json
import torch
from torch import nn, optim
import torch.utils.data as torch_data
import numpy as np
import pickle as pkl

from tqdm import tqdm

from collections import Counter
from nltk.translate.bleu_score import corpus_bleu


from absl import app, flags

from mutex import Vocab,  RecordLoss, MultiIter
from meta_wrapper import MetaWrapper
from data import encode_io, collate, eval_format, collate_with_both_lens, encode_io_with_idx, collate_with_both_lens_without_sort
from src import NoamLR
from functools import partial
import hlog

from curriculum_setter import CurriculumSetter

import wandb

FLAGS = flags.FLAGS
flags.DEFINE_integer("dim", 512, "trasnformer dimension")
flags.DEFINE_integer("n_layers", 2, "number of rnn layers")
flags.DEFINE_integer("n_decoder_layers", 0, "number of rnn decoder layers")
flags.DEFINE_integer("n_batch", 512, "batch size")
flags.DEFINE_float("gclip", 0.5, "gradient clip")
flags.DEFINE_integer("n_epochs", 100, "number of training epochs")
flags.DEFINE_integer("beam_size", 5, "beam search size")
flags.DEFINE_float("lr", 1.0, "learning rate")
flags.DEFINE_float("temp", 1.0, "temperature for samplings")
flags.DEFINE_float("dropout", 0.4, "dropout")
flags.DEFINE_string("load_model", "", "load pretrained model")
flags.DEFINE_integer("seed", 0, "random seed")
flags.DEFINE_bool("debug", False, "debug mode")
flags.DEFINE_bool("full_data", True, "full figure 2 experiments or simple col")
flags.DEFINE_bool("COGS", False, "COGS experiments")
flags.DEFINE_bool("COGS_GENVAL", False, "COGS experiments")
flags.DEFINE_bool("COGS_GOODDEV", False, "COGS experiments")
flags.DEFINE_bool("regularize", False, "regularization")
flags.DEFINE_bool("SCAN", False, "SCAN experiments")
flags.DEFINE_bool("GEO", False, "GEOQUERY experiments")
flags.DEFINE_bool("ATIS", False, "ATIS experiments")
flags.DEFINE_bool("SMCAL", False, "SMCAL")
flags.DEFINE_bool("SCAN_GOODDEV", False, "SCAN experiments")
flags.DEFINE_bool("bidirectional", False, "bidirectional encoders")
flags.DEFINE_bool("attention", True, "Source Attention")
flags.DEFINE_integer("warmup_steps", 4000, "noam warmup_steps")
flags.DEFINE_integer("valid_steps", 500, "validation steps")
flags.DEFINE_integer("max_step", 8000, "maximum number of steps")
flags.DEFINE_integer("tolarance", 5, "early stopping tolarance")
flags.DEFINE_integer("accum_count", 4, "grad accumulation count")
flags.DEFINE_bool("shuffle", True, "shuffle training set")
flags.DEFINE_bool("lr_schedule", True, "noam lr scheduler")
flags.DEFINE_string("hf_lr_schedule", "none", "if not none, use hf_scheduler")
flags.DEFINE_string("scan_split", "", "around_right or jump or mcd1 or mcd2 or mcd3")
flags.DEFINE_string("geo_split", "", "different split/output format for the geoquery dataset")
flags.DEFINE_string("smcal_split", "", "different split format for the smcalflow dataset")
flags.DEFINE_bool("highdrop", False, "high drop mechanism")
flags.DEFINE_bool("highdroptest", False, "high drop at test")
flags.DEFINE_float("highdropvalue", 0.5, "high drop value")
flags.DEFINE_string("attention_type", "simple", "attention type used in the model")
flags.DEFINE_string("model_type", "lstm", "use lstm or transformer")
flags.DEFINE_string("transformer_config", "6layer", "transformer config")
flags.DEFINE_string("proj_name", "compcomp", "project name for wandb")
flags.DEFINE_string("exp_name", "compcomp", "exp name prefix for wandb")
flags.DEFINE_bool("train", False, "training mode")
flags.DEFINE_bool("cogs_perturbation", False, "perturbation for cogs")
flags.DEFINE_bool("scan_perturbation", False, "perturbation for scan")
flags.DEFINE_string("dataset_special_id", "", "identifier for speical datasets")
flags.DEFINE_string("sim_stats_path", "", "path for the similarity stats path")
flags.DEFINE_bool("sample_sim_first", False, "if set, sample all the sim idx first to speed up the experiment")

# additional arguments for meta training
flags.DEFINE_bool("meta", False, "use meta training")
flags.DEFINE_bool("unlikelihood", False, "use unlikelihood training")
flags.DEFINE_integer("n_inner_iter", 5, "number of update steps for the inner optimizer")
flags.DEFINE_float("mlm_prob", 0.15, "probability for masking")
flags.DEFINE_float("meta_loss_weight", 1.0, "mtl weight for the meta loss")
flags.DEFINE_float("ul_loss_weight", 1.0, "ul weight for the meta loss")
flags.DEFINE_string("meta_loss_type", None, "meta loss type")
flags.DEFINE_float("inner_lr", None, "learning rate for the inner opt, set to None to use the same as the outer lr")
flags.DEFINE_integer("multi_permutation", 1, "number of permutations in mx dataset")
flags.DEFINE_integer("multi_permutation_split", 1, "number of permutations in mx dataset, all in split files")
flags.DEFINE_string("special_train_data", None, "set to the name of special train split")
flags.DEFINE_string("special_dev_data", None, "set to the name of special dev split")
flags.DEFINE_string("special_test_data", None, "set to the name of special test split")
flags.DEFINE_string("special_vocab_data", None, "set to the name of special vocab split")

flags.DEFINE_bool("m2m", False, "if True, multiple src maps to multiple tgt")


flags.DEFINE_bool("save_memory", False, "if True, save memory by not saving original text")
flags.DEFINE_bool("dynamic_curriculum", False, "if True, for each epoch, the order and content of batches will be controlled on-the-fly")
flags.DEFINE_string("curriculum_type", None, "the curriculum type for curriculum learning experiments")
flags.DEFINE_bool("augment_in_order", False, "set to True if the augmented training data have the augmented examples in-order right after the corresponding original examples ")
flags.DEFINE_integer("augment_times", 0, "set to the number of augmentation times for the training data (including the original data)")
flags.DEFINE_string("zipf_sampling", None, "choose from [example, prim, src_length, tgt_length]")
flags.DEFINE_string("zipf_dist_path", None, "the path that saves zipf sampling weights for every example")
flags.DEFINE_float("zipf_exponent", 1.0, "zipf exponent, used when the zipf_samling flag is set to non-None")
flags.DEFINE_float("init_training_portion", 1.0, "init training portion novel curriculum experiments")
flags.DEFINE_float("curriculum_ending_time", 0.9, "when applicable (not applicable to all curricula), this argument decides when does the entire curriculum ends (in terms of the total training process) 0.9 means that the curriculum ends when it reaches 0.9*max_step")


ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
DEVICE = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))



def convert_logit_array_to_prob_array(logits_array):
    probs = np.exp(logits_array)
    probs = probs/probs.sum(axis=1, keepdims=True)
    return probs




def prepare_sim_data(FLAGS):
    if os.path.exists(f"{FLAGS.sim_stats_path}.sim.npy"):
        with open(f"{FLAGS.sim_stats_path}.sim.npy", 'rb') as fr:
            sim_prob_array = np.load(fr)
        with open(f"{FLAGS.sim_stats_path}.idx.npy", 'rb') as fr:
            sim_idx_array = np.load(fr)
    else:
        with open(f"{FLAGS.sim_stats_path}.pack.npy", 'rb') as fr:
            sim_logit_array, sim_idx_array = np.load(fr)
            sim_prob_array = convert_logit_array_to_prob_array(sim_logit_array)

    return sim_prob_array, sim_idx_array


def sample_everything(sim_prob_array, sim_idx_array):
    sampled_idxs = []
    max_epoch_num = FLAGS.max_step // (len(sim_idx_array) // (FLAGS.n_batch * FLAGS.accum_count)) + 1
    for idx in range(len(sim_idx_array)):
        sampled_idxs.append(np.random.choice(sim_idx_array[idx], size=max_epoch_num, replace=True, p=sim_prob_array[idx]).tolist())
    return sampled_idxs





def get_zipf_distribution(zipf_sampling, train_items, origin_train_text, FLAGS):
    if zipf_sampling == 'example':
        zipf_example_dist = [1 / np.power(idx + 1, FLAGS.zipf_exponent) for idx in
                     range(len(train_items))]
        denom =  sum(zipf_example_dist)
        zipf_example_dist = [x / denom for x in zipf_example_dist]

    elif zipf_sampling == 'prim':
        example_to_prims = {}
        prim_count = Counter()
        prim_prob = {}
        for i, ogn_text in enumerate(origin_train_text):
            if FLAGS.GEO:
                prim_list = []
                tgt = ogn_text[1]
                open_prim = ''
                for tok_idx, tok in enumerate(tgt):
                    if tok[0].islower():
                        open_prim = open_prim + ' ' + tok
                    else:
                        if open_prim != '':
                            prim_list.append(open_prim.strip())
                            open_prim = ''
                if open_prim != '':
                    prim_list.append(open_prim.strip())
                example_to_prims[i] = prim_list
                for prim in prim_list:
                    prim_count[prim] += 1
            else:
                raise NotImplementedError

        denom = 0
        for i, prim_with_freq in enumerate(prim_count.most_common()):
            prim = prim_with_freq[0]
            prim_prob[prim] = 1 / np.power(i + 1, FLAGS.zipf_exponent)
            denom += prim_prob[prim]
        zipf_example_dist = [0 for _ in train_items]
        for i in range(len(train_items)):
            prims_i = example_to_prims[i]
            for p in prims_i:
                prim_prob_i = (prim_prob[p] / denom) / prim_count[p]
                zipf_example_dist[i] += prim_prob_i


    elif (zipf_sampling == 'src_length') or (zipf_sampling == 'tgt_length'):
        example_to_length = {}
        length_count = Counter()
        length_prob = {}
        for i, ogn_text in enumerate(origin_train_text):
            if zipf_sampling == 'src_length':
                ex_len  =  len(ogn_text[0])
            elif zipf_sampling == 'tgt_length':
                ex_len = len(ogn_text[1])
            else:
                raise ValueError
            example_to_length[i] = ex_len
            length_count[ex_len] += 1

        denom = 0
        for i, length_with_freq in enumerate(length_count.most_common()):
            length = length_with_freq[0]
            length_prob[length] = 1 / np.power(i + 1, FLAGS.zipf_exponent)
            denom += length_prob[length]
        zipf_example_dist = [0 for _ in train_items]
        for i in range(len(train_items)):
            len_i = example_to_length[i]
            len_prob_i = (length_prob[len_i] / denom) / length_count[len_i]
            zipf_example_dist[i] = len_prob_i

    elif (zipf_sampling == 'src_length') or (zipf_sampling == 'tgt_length'):
        example_to_length = {}
        length_count = Counter()
        length_prob = {}
        for i, ogn_text in enumerate(origin_train_text):
            if zipf_sampling == 'src_length':
                ex_len = len(ogn_text[0])
            elif zipf_sampling == 'tgt_length':
                ex_len = len(ogn_text[1])
            else:
                raise ValueError
            example_to_length[i] = ex_len
            length_count[ex_len] += 1
        sorted_length_count = sorted(length_count.most_common(), key=lambda x: x[0])

        denom = 0
        for i, length_with_freq in enumerate(sorted_length_count):
            length = length_with_freq[0]
            length_prob[length] = 1 / np.power(i + 1, FLAGS.zipf_exponent)
            denom += length_prob[length]
        zipf_example_dist = [0 for _ in train_items]
        for i in range(len(train_items)):
            len_i = example_to_length[i]
            len_prob_i = (length_prob[len_i] / denom) / length_count[len_i]
            zipf_example_dist[i] = len_prob_i


    elif (zipf_sampling == 'sorted_num_connector'):
        example_to_length = {}
        length_count = Counter()
        length_prob = {}
        for i, ogn_text in enumerate(origin_train_text):
            num_andafter = sum([1 if x in ['and', 'after'] else 0 for x in ogn_text[0]])
            example_to_length[i] = num_andafter
            length_count[num_andafter] += 1

        denom = 0
        sorted_num_count = sorted(length_count.most_common(), key=lambda x: x[0])
        for i, length_with_freq in enumerate(sorted_num_count):
            length = length_with_freq[0]
            length_prob[length] = 1 / np.power(i + 1, FLAGS.zipf_exponent)
            denom += length_prob[length]
        zipf_example_dist = [0 for _ in train_items]
        for i in range(len(train_items)):
            len_i = example_to_length[i]
            len_prob_i = (length_prob[len_i] / denom) / length_count[len_i]
            zipf_example_dist[i] = len_prob_i


    else:
        raise ValueError
    del origin_train_text # just to save memory
    return zipf_example_dist



def prepare_data(FLAGS):

    vocab_x = Vocab()
    vocab_y = Vocab()
    references = None

    origin_train_text = []

    if FLAGS.SMCAL:
        data = {}
        max_len_x, max_len_y = 0, 0
        smcal_folder = f"smcalflow/clean/{FLAGS.smcal_split}/"
        smcal_file = smcal_folder + "{}.canonical"
        test_split_name = 'test'
        dev_split_name = 'valid'
        train_split_name = 'train'

        if FLAGS.special_train_data is not None:
            train_split_name = FLAGS.special_train_data

        if FLAGS.special_dev_data is not None:
            dev_split_name = FLAGS.special_dev_data

        if FLAGS.special_test_data is not None:
            test_split_name = FLAGS.special_test_data

        splits = [train_split_name, dev_split_name, test_split_name]

        if FLAGS.special_vocab_data is not None:
            splits = [FLAGS.special_vocab_data, train_split_name, dev_split_name, test_split_name]

        for split in splits:
            split_data = []
            with open(f"{ROOT_FOLDER}/"+smcal_file.format(split)+'.src', 'r') as fr_src, open(f"{ROOT_FOLDER}/"+smcal_file.format(split)+'.tgt', 'r') as fr_tgt:

                src_lines = fr_src.readlines()
                tgt_lines = fr_tgt.readlines()
                line_idx = 0
                for src, tgt in zip(src_lines, tgt_lines):
                    src = src.strip().split()
                    tgt = tgt.strip().split()
                    if vocab_x is not None:
                        for t in src:
                            vocab_x.add(t)
                        for t in tgt:
                            vocab_y.add(t)
                    split_data.append(encode_io_with_idx((src, tgt), vocab_x, vocab_y, line_idx))
                    if split == train_split_name and not FLAGS.save_memory:
                        origin_train_text.append((src, tgt))
                    max_len_x = max(max_len_x, len(split_data[-1][0]))
                    max_len_y = max(max_len_y, len(split_data[-1][1]))
                    line_idx += 1
            data[split] = split_data

        max_len_x += 1
        max_len_y += 1
        if vocab_x is not None:
            hlog.value("vocab_x len: ", len(vocab_x))
            hlog.value("vocab_y len: ", len(vocab_y))
        hlog.value("split lengts: ", [(k, len(v)) for (k, v) in data.items()])
        train_items = data[train_split_name]
        val_items = data[dev_split_name]
        test_items = data[test_split_name]


    elif FLAGS.ATIS:
        data = {}
        max_len_x, max_len_y = 0, 0
        atis_file = "atis/{}.txt"

        train_split_name = 'train'
        dev_split_name = 'dev'
        test_split_name = 'test'

        if FLAGS.special_train_data is not None:
            train_split_name = FLAGS.special_train_data

        if FLAGS.special_dev_data is not None:
            dev_split_name = FLAGS.special_dev_data

        if FLAGS.special_test_data is not None:
            test_split_name = FLAGS.special_test_data

        splits = [train_split_name, dev_split_name, test_split_name]

        for split in splits:
            split_data = []
            line_idx = 0
            for l in open(f"{ROOT_FOLDER}/" + atis_file.format(split), "r").readlines():
                inp, out = l.strip().split(' ||| ')
                inp = inp.split(' ')
                out = out.split(' ')
                if vocab_x is not None:
                    for t in inp:
                        vocab_x.add(t)
                    for t in out:
                        vocab_y.add(t)
                split_data.append(encode_io_with_idx((inp, out), vocab_x, vocab_y, line_idx))
                if split == train_split_name and not FLAGS.save_memory:
                    origin_train_text.append((inp, out))
                max_len_x = max(max_len_x, len(split_data[-1][0]))
                max_len_y = max(max_len_y, len(split_data[-1][1]))
                line_idx += 1
            data[split] = split_data

        train_items = data[train_split_name]
        val_items = data[dev_split_name]
        test_items = data[test_split_name]

        max_len_x += 1
        max_len_y += 1
        if vocab_x is not None:
            hlog.value("vocab_x len: ", len(vocab_x))
            hlog.value("vocab_y len: ", len(vocab_y))
        hlog.value("split lengths: ", [(k, len(v)) for (k, v) in data.items()])

    elif FLAGS.GEO:
        data = {}
        max_len_x, max_len_y = 0, 0
        if FLAGS.geo_split == 'logic_query':
            geo_file = "geoquery/logic/query/{}.txt"
        elif FLAGS.geo_split == 'logic_question':
            geo_file = "geoquery/logic/question/{}.txt"
        elif FLAGS.geo_split == 'sql_query':
            geo_file = "geoquery/sql/query/{}.txt"
        elif FLAGS.geo_split == 'sql_question':
            geo_file = "geoquery/sql/question/{}.txt"

        train_split_name = 'train'
        dev_split_name = 'dev'
        test_split_name = 'test'

        if FLAGS.special_train_data is not None:
            train_split_name = FLAGS.special_train_data

        if FLAGS.special_dev_data is not None:
            dev_split_name = FLAGS.special_dev_data

        if FLAGS.special_test_data is not None:
            test_split_name = FLAGS.special_test_data

        splits = [train_split_name, dev_split_name, test_split_name]

        for split in splits:
            split_data = []
            line_idx = 0
            for l in open(f"{ROOT_FOLDER}/" + geo_file.format(split), "r").readlines():
                inp, out = l.strip().split('\t')
                inp = inp.split(' ')
                out = out.split(' ')
                if vocab_x is not None:
                    for t in inp:
                        vocab_x.add(t)
                    for t in out:
                        vocab_y.add(t)
                split_data.append(encode_io_with_idx((inp, out), vocab_x, vocab_y, line_idx))
                if split == train_split_name and not FLAGS.save_memory:
                    origin_train_text.append((inp, out))
                max_len_x = max(max_len_x, len(split_data[-1][0]))
                max_len_y = max(max_len_y, len(split_data[-1][1]))
                line_idx += 1
            data[split] = split_data

        train_items = data[train_split_name]
        val_items = data[dev_split_name]
        test_items = data[test_split_name]

        max_len_x += 1
        max_len_y += 1
        if vocab_x is not None:
            hlog.value("vocab_x len: ", len(vocab_x))
            hlog.value("vocab_y len: ", len(vocab_y))
        hlog.value("split lengths: ", [(k, len(v)) for (k, v) in data.items()])

    elif FLAGS.SCAN_GOODDEV:
        data = {}
        max_len_x, max_len_y = 0, 0
        reg = re.compile('^IN\:\s(.*?)\sOUT\: (.*?)$')
        if FLAGS.scan_split == "around_right":
            scan_file = "SCAN/template_split/tasks_{}_template_around_right.txt"
        elif FLAGS.scan_split == "jump":
            scan_file = "SCAN/add_prim_split/tasks_{}_addprim_jump.txt"
        elif FLAGS.scan_split == "length":
            scan_file = "SCAN/length_split/tasks_{}_length.txt"
        elif FLAGS.scan_split == "mcd1":
            scan_file = "SCAN/mcd_split/tasks_{}_mcd1.txt"
        elif FLAGS.scan_split == "mcd2":
            scan_file = "SCAN/mcd_split/tasks_{}_mcd2.txt"
        elif FLAGS.scan_split == "mcd3":
            scan_file = "SCAN/mcd_split/tasks_{}_mcd3.txt"
        elif FLAGS.scan_split == "mcdx":
            scan_file = "SCAN/mcd_split/tasks_{}_mcdx.txt"
        elif FLAGS.scan_split == "mcdxtmp":
            scan_file = "SCAN/mcd_split/tasks_{}_mcdxtmp.txt"
        elif FLAGS.scan_split == "hugelength":
            scan_file = "SCAN/length_split/tasks_{}.txt"
        elif FLAGS.scan_split == "hugejump":
            scan_file = "SCAN/add_prim_split/tasks_{}.txt"
        elif FLAGS.scan_split == "hugearound_right":
            scan_file = "SCAN/template_split/tasks_{}.txt"
        elif FLAGS.scan_split == "hugesimple":
            scan_file = "SCAN/simple_split/tasks_{}.txt"
        else:
            raise ValueError
        test_split_name = 'newtest'
        dev_split_name = 'gooddev'
        train_split_name = 'train'
        if FLAGS.multi_permutation > 1:
            train_split_name = "trainm{}".format(FLAGS.multi_permutation)
        elif FLAGS.special_train_data is not None:
            train_split_name = FLAGS.special_train_data

        if FLAGS.special_dev_data is not None:
            dev_split_name = FLAGS.special_dev_data

        if FLAGS.special_test_data is not None:
            test_split_name = FLAGS.special_test_data

        splits = [train_split_name, dev_split_name, test_split_name]
        if FLAGS.multi_permutation_split > 1:
            for i in range(FLAGS.multi_permutation_split):
                splits.append('trainm{}s{}'.format(FLAGS.multi_permutation_split, i))

        for split in splits:
            split_data = []
            line_idx = 0
            for l in open(f"{ROOT_FOLDER}/" + scan_file.format(split), "r").readlines():
                m = reg.match(l)
                inp, out = m.groups(1)
                inp, out = (inp.split(" "), out.split(" "))
                if vocab_x is not None:
                    for t in inp:
                        vocab_x.add(t)
                    for t in out:
                        vocab_y.add(t)
                split_data.append(encode_io_with_idx((inp, out), vocab_x, vocab_y, line_idx))
                if split == train_split_name and not FLAGS.save_memory:
                    origin_train_text.append((inp, out))
                max_len_x = max(max_len_x, len(split_data[-1][0]))
                max_len_y = max(max_len_y, len(split_data[-1][1]))
                line_idx += 1
            data[split] = split_data

        train_items = data[train_split_name]
        val_items = data[dev_split_name]
        test_items = data[test_split_name]

        max_len_x += 1
        max_len_y += 1
        if vocab_x is not None:
            hlog.value("vocab_x len: ", len(vocab_x))
            hlog.value("vocab_y len: ", len(vocab_y))
        hlog.value("split lengths: ", [(k, len(v)) for (k, v) in data.items()])
    elif FLAGS.COGS_GOODDEV:
        data = {}
        max_len_x, max_len_y = 0, 0

        old_gentest_split_name = 'gen'
        old_dev_split_name = 'dev'
        test_split_name = 'new_test'
        dev_split_name = 'dev_gen'
        train_split_name = 'train'
        if FLAGS.multi_permutation > 1:
            train_split_name = "trainm{}".format(FLAGS.multi_permutation)
        elif FLAGS.special_train_data is not None:
            train_split_name = FLAGS.special_train_data
        splits = [train_split_name, dev_split_name, test_split_name, old_dev_split_name, old_gentest_split_name]
        if FLAGS.multi_permutation_split > 1:
            for i in range(FLAGS.multi_permutation_split):
                splits.append('trainm{}s{}'.format(FLAGS.multi_permutation_split, i))
        for split in splits:
            split_data = []
            line_idx = 0
            for l in open(f"{ROOT_FOLDER}/COGS/cogs/{split}.tsv", "r").readlines():
                text, sparse, _ = l.split("\t")
                text, sparse = (text.split(" "), sparse.split(" "))
                if vocab_x is not None:
                    for t in text:
                        vocab_x.add(t)
                    for t in sparse:
                        vocab_y.add(t)
                split_data.append(encode_io_with_idx((text, sparse), vocab_x, vocab_y, line_idx))
                if split == train_split_name and not FLAGS.save_memory:
                    origin_train_text.append((inp, out))
                max_len_x = max(max_len_x, len(split_data[-1][0]))
                max_len_y = max(max_len_y, len(split_data[-1][1]))
                line_idx += 1
            data[split] = split_data


        max_len_x += 1
        max_len_y += 1
        if vocab_x is not None:
            hlog.value("vocab_x len: ", len(vocab_x))
            hlog.value("vocab_y len: ", len(vocab_y))
        hlog.value("split lengts: ", [(k, len(v)) for (k, v) in data.items()])
        train_items = data[train_split_name]
        val_items = data[dev_split_name]
        test_items = data[test_split_name]

    else:
        raise ValueError

    if FLAGS.meta:
        if vocab_x is not None:
            vocab_x.add_mask()
            vocab_y.add_mask()

    return vocab_x, vocab_y, max_len_x, max_len_y, train_items, val_items, test_items, references, data, origin_train_text


def train(opt, model, train_dataset, val_dataset, references=None, full_exp_name=None, double_batch_training=False, additional_data=None, dynamic_curriculum=False, curriculum_setter=None, zipf_sampling_dist=None):

    if FLAGS.lr_schedule:
        if FLAGS.hf_lr_schedule=='none':
            scheduler = NoamLR(opt, FLAGS.dim, warmup_steps=FLAGS.warmup_steps)
        elif FLAGS.hf_lr_schedule!='none':
            from transformers import get_scheduler
            scheduler = get_scheduler(name=FLAGS.hf_lr_schedule, optimizer=opt, num_warmup_steps=FLAGS.warmup_steps, num_training_steps=FLAGS.max_step)
    else:
        scheduler = None


    if zipf_sampling_dist is None:
        train_loader = torch_data.DataLoader(
            train_dataset,
            batch_size=FLAGS.n_batch,
            shuffle=FLAGS.shuffle,
            collate_fn= collate_with_both_lens
        )
    else:

        zipf_sampler = torch_data.WeightedRandomSampler(
            weights=zipf_sampling_dist,
            num_samples=len(train_dataset)
        )
        train_loader = torch_data.DataLoader(
            train_dataset,
            batch_size=FLAGS.n_batch,
            sampler = zipf_sampler,
            collate_fn= collate_with_both_lens,
        )

    if double_batch_training:
        aux_train_loader = torch_data.DataLoader(
            train_dataset,
            batch_size=FLAGS.n_batch,
            shuffle=FLAGS.shuffle,
            collate_fn= collate_with_both_lens
        )
        aux_data_iter = iter(aux_train_loader)
    else:
        aux_batch = None



    if additional_data is not None:
        raise NotImplementedError


    def get_next_batch(data_loader, data_iter):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
        return batch, data_iter


    def update_ema(val, new, alpha=0.99):
        if val is None:
            return new
        else:
            return alpha * val + (1 - alpha) * new


    tolarance = FLAGS.tolarance
    best_f1 = best_acc = -np.inf
    best_loss = np.inf
    best_bleu = steps = accum_steps = 0
    got_nan = False
    is_running = lambda: not got_nan and accum_steps < FLAGS.max_step and tolarance > 0

    train_loss = None
    train_batches = 0
    train_only_loss = train_other_loss = None
    train_meta_loss_delta = None

    epoch_num = 0
    while is_running():
        opt.zero_grad()
        recorder = RecordLoss()

        if dynamic_curriculum:
            train_loader = curriculum_setter.get_train_loader_with_curriculum(epoch_num=epoch_num, cur_step=accum_steps, max_step=FLAGS.max_step)

        for train_sample in tqdm(train_loader):
            inp, out, lens, y_lens, index = train_sample
            if not is_running():
                break
            #get other data used in training
            if double_batch_training or (FLAGS.meta_loss_type in ['maml_lev', 'unlikelihood_lev', 'levmaml_unlikelihood', 'levmaml_unmaml']):
                if FLAGS.meta_loss_type in ['maml_lev', 'unlikelihood_lev', 'levmaml_unlikelihood', 'levmaml_unmaml']:
                    raise NotImplementedError
                else:
                    aux_batch, aux_data_iter = get_next_batch(aux_train_loader, aux_data_iter)
                    aux_batch = [x.to(DEVICE) if x is not None else None for x in aux_batch]

            if FLAGS.meta or FLAGS.unlikelihood:
                all_train_losses = model.train_forward(inp.to(DEVICE), out.to(DEVICE), lens=lens.to(DEVICE), y_lens=y_lens.to(DEVICE), recorder=recorder, accum_count = FLAGS.accum_count, current_lr = scheduler.get_lr(), aux_batch=aux_batch)
                train_batch_loss, batch_train_only_loss, batch_other_loss = all_train_losses
            else:
                train_batch_loss = nll = model(inp.to(DEVICE), out.to(DEVICE), lens=lens.to(DEVICE), recorder=recorder)
                batch_train_only_loss = train_batch_loss
                batch_other_loss= 0


            steps += 1
            loss = train_batch_loss / FLAGS.accum_count
            accum_train_only_loss = batch_train_only_loss / FLAGS.accum_count
            accum_other_loss = batch_other_loss / FLAGS.accum_count
            loss.backward()

            train_loss = update_ema(train_loss,  (loss.detach().item() * FLAGS.accum_count))
            if type(accum_train_only_loss) is not float:
                train_only_loss = update_ema(train_only_loss, (accum_train_only_loss.detach().item() * FLAGS.accum_count))
            else:
                train_only_loss = update_ema(train_only_loss, (accum_train_only_loss * FLAGS.accum_count))
            if type(accum_other_loss) is not float:
                train_other_loss = update_ema(train_other_loss, (accum_other_loss.detach().item() * FLAGS.accum_count))
            else:
                train_other_loss = update_ema(train_other_loss, accum_other_loss * FLAGS.accum_count)



            train_batches += 1
            if steps % FLAGS.accum_count == 0:
                accum_steps += 1
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), FLAGS.gclip)
                if not np.isfinite(gnorm.cpu()):
                    got_nan = True
                    print("=====GOT NAN=====")
                    break
                opt.step()
                opt.zero_grad()

                if scheduler is not None:
                    scheduler.step()


                if accum_steps % FLAGS.valid_steps == 0:
                    with hlog.task(accum_steps):
                        wandb.log({"train" + "/loss": train_loss })
                        hlog.value("curr loss", train_loss )
                        wandb.log({"train" + "/train_only_loss": train_only_loss })
                        hlog.value("train only loss", train_only_loss)
                        wandb.log({"train" + "/other_loss": train_other_loss })
                        hlog.value("other loss", train_other_loss )
                        acc, f1, val_loss, bscore = validate(model, val_dataset, references=references, split_name='val', step_num=accum_steps)
                        val_acc = acc
                        model.train()
                        hlog.value("acc", acc)
                        hlog.value("f1", f1)
                        hlog.value("bscore", bscore)
                        hlog.value("val_loss", val_loss)
                        if val_acc > best_acc:
                            best_acc = val_acc
                            tolarance = FLAGS.tolarance
                            torch.save(model, f"{full_exp_name}.best.model")
                        else:
                            tolarance -= 1
                        if val_loss < best_loss:
                            torch.save(model, f"{full_exp_name}.bestloss.model")
                        best_loss = min(best_loss, val_loss)
                        best_f1 = max(best_f1, f1)
                        best_acc = max(best_acc, acc)
                        best_bleu = max(best_bleu, bscore)
                        hlog.value("best_loss", best_loss)
                        hlog.value("best_acc", best_acc)
                        hlog.value("best_f1", best_f1)
                        hlog.value("best_bleu", best_bleu)

    wandb.log({"final_val" + "/acc": acc,
               "final_val" + "/f1": f1,
               "final_val" + "/bleu": bscore})
    wandb.log({"best_val" + "/acc": best_acc,
               "best_val" + "/f1": best_f1,
               "best_val" + "/loss": best_loss,
               "best_val" + "/bleu": best_bleu})
    hlog.value("final_acc", acc)
    hlog.value("final_f1", f1)
    hlog.value("final_bleu", bscore)
    hlog.value("best_acc", best_acc)
    hlog.value("best_f1", best_f1)
    hlog.value("best_loss", best_loss)
    hlog.value("best_bleu", best_bleu)
    return acc, f1, bscore



def validate(model, val_dataset, vis=False, beam=False, references=None, split_name=None, step_num=None):
    model.eval()
    val_loader = torch_data.DataLoader(
        val_dataset,
        batch_size=FLAGS.n_batch,
        shuffle=False,
        collate_fn=collate_with_both_lens_without_sort
    )
    total = correct = loss = tp = fp = fn = 0
    cur_references = []
    candidates = []
    with torch.no_grad():
        for inp, out, lens, y_lens, index in tqdm(val_loader):
            input = inp.to(DEVICE)
            lengths = lens.to(DEVICE)
            y_lengths = y_lens.to(DEVICE)
            pred, _ = model.sample(input,
                                   lens=lengths,
                                   temp=1.0,
                                   max_len=model.MAXLEN_Y,
                                   greedy=True,
                                   beam_size=FLAGS.beam_size * beam,
                                   calc_score=False)

            loss += model.pyx(input, out.to(DEVICE), lens=lengths, y_lens=y_lengths).item() * input.shape[1]
            for i, seq in enumerate(pred):
                ref = out[:, i].numpy().tolist()
                ref = eval_format(model.vocab_y, ref)
                pred_here = eval_format(model.vocab_y, pred[i])
                if references is None:
                    cur_references.append([ref])
                else:
                    inpref = " ".join(model.vocab_x.decode(inp[0:lens[i], i].numpy().tolist()))
                    cur_references.append(references[inpref])

                candidates.append(pred_here)
                if FLAGS.m2m:
                    new_pred_here = ''.join([c for c in pred_here if not c.isdigit()])
                    new_ref = ''.join([c for c in ref if not c.isdigit()])
                    correct_here = new_pred_here == new_ref
                else:
                    correct_here = pred_here == ref
                correct += correct_here
                tp_here = len([p for p in pred_here if p in ref])
                tp += tp_here
                fp_here = len([p for p in pred_here if p not in ref])
                fp += fp_here
                fn_here = len([p for p in ref if p not in pred_here])
                fn += fn_here
                total += 1
                if vis:
                    with hlog.task(total):
                        hlog.value("label", correct_here)
                        hlog.value("tp", tp_here)
                        hlog.value("fp", fp_here)
                        hlog.value("fn", fn_here)
                        inp_lst = inp[:, i].detach().cpu().numpy().tolist()
                        hlog.value("input", eval_format(model.vocab_x, inp_lst))
                        hlog.value("gold", ref)
                        hlog.value("pred", pred_here)

    wandb.log({split_name + "/loss": loss / total,
               split_name + "/acc": correct / total})

    bleu_score = corpus_bleu(cur_references, candidates)
    acc = correct / total
    loss = loss / total
    if tp+fp > 0:
        prec = tp / (tp + fp)
    else:
        prec = 0
    rec = tp / (tp + fn)
    if prec == 0 or rec == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    hlog.value("acc", acc)
    hlog.value("f1", f1)
    hlog.value("bleu", bleu_score)
    wandb.log({split_name + "/f1": f1,
               split_name + "/bleu": bleu_score})

    return acc, f1, loss, bleu_score


def swap_io(items):
    return [(y, x) for (x, y) in items]


def main(argv):
    hlog.flags()
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)

    vocab_x, vocab_y, max_len_x, max_len_y, train_items, val_items, test_items, references, all_data, origin_train_text = prepare_data(FLAGS)

    if FLAGS.meta_loss_type in ['maml_lev', 'unlikelihood_lev', 'levmaml_unlikelihood', 'levmaml_unmaml']:
        sim_prob_array, sim_idx_array = prepare_sim_data(FLAGS)
        additional_data = {'sim_prob_array': sim_prob_array,
                           'sim_idx_array': sim_idx_array}
        if FLAGS.sample_sim_first:
            sampled_idx = sample_everything(sim_prob_array, sim_idx_array)
            additional_data['sim_sampled_idx'] = sampled_idx

    else:
        additional_data = None




    args_dict = FLAGS.flag_values_dict()
    full_exp_name = FLAGS.exp_name + '___' + 's{0}_b{1}_ly{2}_d{3}_lr{4}_dp{5}_b{6}_w{7}'.format(FLAGS.seed, FLAGS.n_batch * FLAGS.accum_count, FLAGS.n_layers, FLAGS.dim, FLAGS.lr, FLAGS.dropout, FLAGS.beam_size, FLAGS.meta_loss_weight)

    wandb.init(name=full_exp_name, project=FLAGS.proj_name, config=args_dict)

    if FLAGS.load_model == "":
        model = MetaWrapper(vocab_x,
                      vocab_y,
                      FLAGS.dim,
                      FLAGS.dim,
                      max_len_x=max_len_x,
                      max_len_y=max_len_y,
                      copy=False,
                      n_layers=FLAGS.n_layers,
                      self_att=False,
                      attention=FLAGS.attention,
                      dropout=FLAGS.dropout,
                      temp=FLAGS.temp,
                      bidirectional=FLAGS.bidirectional, 
                      model_type = FLAGS.model_type,
                      transformer_config = FLAGS.transformer_config,
                      attention_type = FLAGS.attention_type,
                      n_decoder_layers = FLAGS.n_decoder_layers,
                      meta = FLAGS.meta,
                      unlikelihood = FLAGS.unlikelihood,
                      n_inner_iter = FLAGS.n_inner_iter,
                      mlm_prob= FLAGS.mlm_prob,
                      meta_loss_weight = FLAGS.meta_loss_weight,
                      ul_loss_weight = FLAGS.ul_loss_weight,
                      meta_loss_type = FLAGS.meta_loss_type,
                      cogs_perturbation = FLAGS.cogs_perturbation,
                      scan_perturbation = FLAGS.scan_perturbation, 
                      inner_lr = FLAGS.inner_lr).to(DEVICE) 
        double_batch_training = model.use_double_batch
    else:
        model = torch.load(FLAGS.load_model).to(DEVICE)
        double_batch_training = model.use_double_batch

    if FLAGS.model_type == 'transformer' and FLAGS.hf_lr_schedule=='none':
        FLAGS.dim = model.pyx.output_dim # overwriting needed for noam scheduler

    if FLAGS.train:
        with hlog.task("train model"):
            opt = optim.Adam(model.pyx.parameters(), lr=FLAGS.lr, betas=(0.9, 0.998))

            if FLAGS.dynamic_curriculum:
                curriculum_setter = CurriculumSetter(curriculum_type=FLAGS.curriculum_type,
                                                     train_data = train_items,
                                                     origin_train_text=origin_train_text,
                                                     augment_in_order=FLAGS.augment_in_order,
                                                     augment_times = FLAGS.augment_times,
                                                     n_batch = FLAGS.n_batch,
                                                     init_training_portion=FLAGS.init_training_portion,
                                                     curriculum_ending_time=FLAGS.curriculum_ending_time)

            else:
                curriculum_setter = None

            if FLAGS.zipf_dist_path is not None:
                with open(FLAGS.zipf_dist_path, 'rb') as fr:
                    zipf_sampling_dist = pkl.load(fr)
            else:
                if FLAGS.zipf_sampling is not None:
                    zipf_sampling_dist = get_zipf_distribution(FLAGS.zipf_sampling, train_items, origin_train_text, FLAGS)
                else:
                    zipf_sampling_dist = None

            acc, f1, bscore = train(opt,
                                    model,
                                    train_items,
                                    val_items,
                                    references=references,
                                    full_exp_name=full_exp_name,
                                    double_batch_training=double_batch_training,
                                    additional_data=additional_data,
                                    dynamic_curriculum=FLAGS.dynamic_curriculum,
                                    curriculum_setter=curriculum_setter,
                                    zipf_sampling_dist=zipf_sampling_dist)
        torch.save(model, f"{full_exp_name}.final.model")


    if not FLAGS.train:
        with hlog.task("val evaluation"):
            validate(model, val_items, vis=True, references=references, split_name="val")


    with hlog.task("test evaluation (greedy)"):
        validate(model, test_items, vis=True, beam=False, references=references, split_name="test(greedy)")


    if FLAGS.train:
        model = torch.load(f"{full_exp_name}.best.model")


        with hlog.task("test evaluation (greedy)"):
            validate(model, test_items, vis=True, beam=False, references=references, split_name="best_test(greedy)")




if __name__ == "__main__":
    app.run(main)
