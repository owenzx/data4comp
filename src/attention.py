import torch
from torch import nn
import torch.nn.functional as F
import pdb

class SimpleAttention(nn.Module):
    def __init__(self, n_features, n_hidden, key=False, copy=False, query=True, memory=False, attention_type='simple'):
        super().__init__()
        self.key = key
        self.query = query
        self.memory = memory
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.copy = copy
        self.attention_type = attention_type

        if self.attention_type == 'bahdanau':
            self.attention_scorer = BahdanauPointer(n_hidden, n_hidden, n_hidden)

        if self.copy: assert self.query

        if self.key:
            self.make_key = nn.Linear(n_features, n_hidden)
        if self.query:
            self.make_query = nn.Linear(n_features, (1+copy) * n_hidden)
        if self.memory:
            self.make_memory = nn.Linear(n_features, n_hidden)
        self.n_out = n_hidden

    def forward(self, features, hidden, mask=None):
        if self.key:
            key = self.make_key(features)
        else:
            key = features

        if self.memory:
            memory = self.make_memory(features)
        else:
            memory = features

        if self.query:
            query = self.make_query(hidden)
        else:
            query = hidden

        # attention
        #query = query.expand_as(key) # T X B X H
        if self.copy:
            query = query.view(1, -1, query.shape[-1] // 2, 2)
            key   = key.unsqueeze(-1)
            mask  = mask.unsqueeze(-1)
        elif len(query.shape) < 3:
            query = query.unsqueeze(0)

        if self.attention_type == 'simple':
            scores = (key * query).sum(dim=2)
        elif self.attention_type == 'bahdanau':
            scores = self.attention_scorer(query.squeeze(0), key.transpose(0, 1))
            scores = scores.transpose(0, 1)
        else:
            raise KeyError

        if mask is not None:
            scores += mask * -99999

        if self.copy:
            scores, copy_scores = torch.chunk(scores,2,dim=-1)
            copy_distribution = F.softmax(copy_scores.squeeze(-1), dim=0)
            distribution = F.softmax(scores.squeeze(-1), dim=0)
        else:
            distribution = F.softmax(scores, dim=0)
            copy_distribution = distribution


        weighted = (memory * distribution.unsqueeze(2).expand_as(memory))
        summary = weighted.sum(dim=0, keepdim=True)

        # value
        return summary, distribution, copy_distribution


class BahdanauPointer(torch.nn.Module):
    def __init__(self, query_size, key_size, proj_size):
        super().__init__()
        self.compute_scores = torch.nn.Sequential(
            torch.nn.Linear(query_size + key_size, proj_size),
            torch.nn.Tanh(),
            torch.nn.Linear(proj_size, 1),
        )

    def forward(self, query: torch.Tensor, keys: torch.Tensor, attn_mask=None):
        # query shape: batch x query_size
        # keys shape: batch x num keys x key_size

        # query_expanded shape: batch x num keys x query_size
        query_expanded = query.unsqueeze(1).expand(-1, keys.shape[1], -1)

        # scores shape: batch x num keys x 1
        attn_logits = self.compute_scores(
            # shape: batch x num keys x query_size + key_size
            torch.cat((query_expanded, keys), dim=2)
        )
        # scores shape: batch x num keys
        attn_logits = attn_logits.squeeze(2)
        maybe_mask(attn_logits, attn_mask)
        return attn_logits



def maybe_mask(attn, attn_mask):
    if attn_mask is not None:
        assert all(
            a == 1 or b == 1 or a == b
            for a, b in zip(attn.shape[::-1], attn_mask.shape[::-1])
        ), "Attention mask shape {} should be broadcastable with attention shape {}".format(
            attn_mask.shape, attn.shape
        )

        attn.data.masked_fill_(attn_mask, -float("inf"))

class Attention(torch.nn.Module):
    def __init__(self, pointer):
        super().__init__()
        self.pointer = pointer
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query, values, attn_mask=None):
        # query shape: batch x query_size
        # values shape: batch x num values x value_size

        # attn_logits shape: batch x num values
        attn_logits = self.pointer(query, values, attn_mask)
        # attn_logits shape: batch x num values
        attn = self.softmax(attn_logits)
        # output shape: batch x 1 x value_size
        output = torch.bmm(attn.unsqueeze(1), values)
        output = output.squeeze(1)
        return output, attn

class BahdanauAttention(Attention):
    def __init__(self, query_size, value_size, proj_size):
        super().__init__(BahdanauPointer(query_size, value_size, proj_size))
