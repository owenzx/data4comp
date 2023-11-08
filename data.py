import torch
from torch import nn, optim
import torch.utils.data as torch_data
import torch.nn.functional as F
import numpy as np
import random
import sys
from src import batch_seqs
EPS = 1e-7

def encode(data, vocab_x, vocab_y):
    encoded = []
    for datum in data:
        encoded.append(encode_io(datum, vocab_x, vocab_y))
    return encoded

def encode_io(datum, vocab_x, vocab_y):
    inp, out = datum
    return ([vocab_x.sos()] + vocab_x.encode(inp) + [vocab_x.eos()], [vocab_y.sos()] + vocab_y.encode(out) + [vocab_y.eos()])

def encode_io_with_idx(datum, vocab_x, vocab_y, index):
    inp, out = datum
    return ([vocab_x.sos()] + vocab_x.encode(inp) + [vocab_x.eos()], [vocab_y.sos()] + vocab_y.encode(out) + [vocab_y.eos()], index)

def eval_format(vocab, seq):
    if vocab.eos() in seq:
        seq = seq[:seq.index(vocab.eos())+1]
    if seq[0] == vocab.sos():
        seq = seq[1:]
    if seq[-1] == vocab.eos():
        seq = seq[:-1]
    return vocab.decode(seq)


def eval_format_probs(prob_seq, vocab, seq):
    prob_seq = prob_seq[:len(seq)+1]

    token_prob_lst = []

    for tok, prob in zip(seq, prob_seq):
        token_prob_lst.append(tok+' '+str(prob))

    token_prob_lst.append('eos' + ' ' + str(prob_seq[-1]))

    token_with_prob_str = '\t'.join(token_prob_lst)
    return token_with_prob_str



def pred_to_ref_format(pred, vocab):
    if pred[0] != vocab.sos():
        pred = [vocab.sos()] + pred

    if pred[-1] != vocab.eos():
        pred = pred + [vocab.eos()]

    return pred



def preds_to_refs_batch(preds, vocab, return_len=False):
    preds = [pred_to_ref_format(p, vocab) for p in preds]
    max_len = max([len(p) for p in preds])
    all_lens = torch.LongTensor(list(map(len, preds)))
    ref_batch = torch.zeros((len(preds),max_len) , dtype=torch.long)
    for i, p in enumerate(preds):
        ref_batch[i, :len(p)] = torch.LongTensor(p)

    ref_batch = ref_batch.transpose(0,1)
    if return_len:
        return ref_batch, all_lens
    else:
        return ref_batch



def collate(batch):
    batch = sorted(batch,
                   key=lambda x: len(x[0]),
                   reverse=True)
    if len(batch[0]) == 2:
        inp, out = zip(*batch)
        index = None
    else:
        inp, out, index = zip(*batch)
        index = torch.LongTensor(list(index))
    lens = torch.LongTensor(list(map(len,inp)))
    inp = batch_seqs(inp)
    out = batch_seqs(out)
    return inp, out, lens, index


def collate_with_both_lens(batch):
    batch = sorted(batch,
                   key=lambda x: len(x[0]),
                   reverse=True)
    if len(batch[0]) == 2:
        inp, out = zip(*batch)
        index = None
    else:
        inp, out, index = zip(*batch)
        index = torch.LongTensor(list(index))
    inp_lens = torch.LongTensor(list(map(len,inp)))
    out_lens = torch.LongTensor(list(map(len,out)))
    inp = batch_seqs(inp)
    out = batch_seqs(out)
    return inp, out, inp_lens, out_lens, index


def collate_with_both_lens_without_sort(batch):
    if len(batch[0]) == 2:
        inp, out = zip(*batch)
        index = None
    else:
        inp, out, index = zip(*batch)
        index = torch.LongTensor(list(index))
    inp_lens = torch.LongTensor(list(map(len,inp)))
    out_lens = torch.LongTensor(list(map(len,out)))
    inp = batch_seqs(inp)
    out = batch_seqs(out)
    return inp, out, inp_lens, out_lens, index


def collage_targets_only_without_sort(batch):
    out = batch
    out = batch_seqs(out)
    return out



def collate_without_sort(batch):
    if len(batch[0]) == 2:
        inp, out = zip(*batch)
        index = None
    else:
        inp, out, index = zip(*batch)
        index = torch.LongTensor(list(index))
    inp_lens = torch.LongTensor(list(map(len,inp)))
    inp = batch_seqs(inp)
    out = batch_seqs(out)
    return inp, out, inp_lens, index




def get_lens_of_text(text):
    out_lens = torch.LongTensor(list(map(len,text)))
    return out_lens

