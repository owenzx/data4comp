import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(
            self,
            vocab,
            n_embed,
            n_hidden,
            n_layers,
            bidirectional=True,
            dropout=0,
            rnntype=nn.LSTM,
    ):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(len(vocab), n_embed, vocab.pad())
        self.embed_dropout = nn.Dropout(dropout)
        self.rnn = rnntype(
            n_embed, n_hidden, n_layers, bidirectional=bidirectional, dropout=dropout
        )

    def get_embed(self, data, detach_mask=None):
        if len(data.shape) == 3:
            emb = torch.matmul(data, self.embed.weight)
            tokens = torch.argmax(data.detach(), dim=-1)
            emb = emb * (tokens != self.vocab.pad()).unsqueeze(2).float()
        else:
            emb = self.embed(data)
        if detach_mask is not None:
            # detach all locations where the mask is True
            #TODO test whether this is effective
            emb[detach_mask] = emb[detach_mask].detach()
        return emb



    def forward(self, data, lens=None, additional_output=False, use_updated_emb=False, detach_mask=None):
        if use_updated_emb:
            emb = self.tmp_emb_for_update
            # TODO debug code!!!!

            # return emb.sum()
            # return self.rnn.weight_ih_l0.sum()
            # return self.tmp_emb_for_update.sum()
        else:
            if len(data.shape) == 3:
                emb    = torch.matmul(data, self.embed.weight)
                tokens = torch.argmax(data.detach(),dim=-1)
                emb    = emb * (tokens != self.vocab.pad()).unsqueeze(2).float()
            else:
                emb   = self.embed(data)
        if detach_mask is not None:
            # pass
            # print("HEY!!!")
            with torch.no_grad():
                self.tmp_emb_for_update[:] = emb
            emb = self.tmp_emb_for_update
            #TODO before or after, this is a question
            detached_emb = emb.detach()
            # detached_emb = emb
            detach_mask = detach_mask.int().unsqueeze(-1)
            # print(emb.shape)
            # print(detach_mask.shape)
            # TODO check if mask is reversed
            combine_emb = emb * (1-detach_mask) + detached_emb * detach_mask
            # emb[detach_mask] = emb[detach_mask].detach()
            emb = combine_emb
        additional_dict = {"enc_emb": emb.clone()}
        if lens is not None:
            padded_sequence = self.embed_dropout(emb)
            total_length = padded_sequence.shape[0]
            packed_sequence = nn.utils.rnn.pack_padded_sequence(padded_sequence, lens.cpu())
            # with torch.backends.cudnn.flags(enabled=False):
            packed_output, hidden = self.rnn(packed_sequence)
            output_padded,_ = nn.utils.rnn.pad_packed_sequence(packed_output,
                                                               total_length=total_length,
                                                               padding_value=self.vocab.pad())
            if not additional_output:
                return output_padded, hidden
            else:
                return output_padded, hidden, additional_dict
        else:
            # with torch.backends.cudnn.flags(enabled=False):
            output_padded, hidden = self.rnn(self.embed_dropout(emb))
            if not additional_output:
                # return self.rnn(self.embed_dropout(emb))
                return output_padded, hidden
            else:
                # output_padded, hidden = self.rnn(self.embed_dropout(emb))
                return output_padded, hidden, additional_dict
