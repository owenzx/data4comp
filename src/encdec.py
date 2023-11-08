import torch
from torch import nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .attention import SimpleAttention
class EncDec(nn.Module):
    def __init__(self,
                 vocab_x,
                 vocab_y,
                 emb,
                 dim,
                 copy=False,
                 n_layers=1,
                 self_att=False,
                 dropout=0.,
                 bidirectional=True,
                 rnntype=nn.LSTM,
                 MAXLEN=45,
                 source_att=False,
                 attention_type = 'simple',
                 n_decoder_layers=0,
                ):

        super().__init__()

        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.rnntype = rnntype
        self.bidirectional = bidirectional
        self.nll = nn.CrossEntropyLoss(ignore_index=vocab_y.pad(), reduction='sum') #TODO: why mean better?
        self.nll_wr = nn.CrossEntropyLoss(ignore_index=vocab_y.pad(), reduction='none')
        self.dim = dim
        self.n_layers = n_layers
        self.MAXLEN = MAXLEN
        self.source_att = source_att
        self.self_att = self_att
        self.concat_feed = self_att or source_att
        if n_decoder_layers == 0:
            self.n_decoder_layers = n_layers
        else:
            self.n_decoder_layers = n_decoder_layers

        if self.bidirectional:
            self.proj = nn.Linear(dim * 2, dim)
        else:
            self.proj = nn.Identity()



        self.encoder = Encoder(vocab_x,
                               emb,
                               dim,
                               n_layers,
                               dropout=dropout,
                               bidirectional=bidirectional,
                               rnntype=rnntype)

        if self.source_att:
            attention = (SimpleAttention(dim,dim, attention_type=attention_type),)
        else:
            attention = None

        self.decoder = Decoder(vocab_y,
                               emb,
                               dim,
                               self.n_decoder_layers,
                               attention=attention,
                               self_attention=self_att,
                               copy=copy,
                               dropout=dropout,
                               rnntype=rnntype,
                               concat_feed=self.concat_feed,
                               MAXLEN=self.MAXLEN,
                              )


    def pass_hiddens(self, rnnstate):
        if self.rnntype == nn.LSTM:
            state = [
                s.view(self.n_layers, -1, rnnstate[0].shape[1], self.dim).sum(dim=1)
                for s in rnnstate
            ]
        else:
            state = rnnstate.view(self.n_layers, -1, rnnstate.shape[1], self.dim).sum(dim=1)

        return state

    def forward(self, inp, out, lens=None, per_instance=False, additional_output=False, use_updated_emb=False, detach_mask=None):
        #TODO DEBUG code!!!
        # if fix_emb is not None:
        #     encoder_output = self.encoder(inp, lens=lens, additional_output=additional_output, fix_emb=fix_emb, detach_mask=detach_mask)
        #     return encoder_output, {'enc_emb':None}
        encoder_output = self.encoder(inp, lens=lens, additional_output=additional_output, use_updated_emb=use_updated_emb, detach_mask=detach_mask)
        hid, state = encoder_output[0], encoder_output[1]
        if additional_output:
            enc_add_dict = encoder_output[2]


        state = self.pass_hiddens(state)
        out_src = out[:-1, :]

        if self.source_att:
            att_features = [self.proj(hid)]
            att_tokens   = [inp]
        else:
            att_features = None
            att_tokens   = None

        decoder_output = self.decoder(state,
                                         out_src.shape[0],
                                         ref_tokens=out_src,
                                         att_features=att_features,
                                         att_tokens=att_tokens,
                                         additional_output=additional_output)
        dec, extras = decoder_output[0], decoder_output[3]
        if additional_output:
            dec_add_dict = decoder_output[4]

        if per_instance:
            out_tgt = out[1:, :].transpose(0,1)
            output = dec.permute(1,2,0)
            loss = self.nll_wr(output,out_tgt).sum(dim=-1)
        else:
            out_tgt = out[1:, :].view(-1)
            dec = dec.view(-1, len(self.vocab_y))
            loss = self.nll(dec, out_tgt) / inp.shape[1]

        if additional_output:
            additional_dict = {**enc_add_dict, **dec_add_dict}
            return loss, additional_dict
        else:
            return loss

    def logprob(self, inp, out, lens=None):
        hid, state = self.encoder(inp, lens=lens)
        if self.source_att:
            att_features = [self.proj(hid)]
            att_tokens   = [inp]
        else:
            att_features = None
            att_tokens   = None

        return self.decoder.logprob(out,
                                    rnn_state=self.pass_hiddens(state),
                                    att_features=att_features,
                                    att_tokens=att_tokens)

    def logprob_interleaved(self, inp, out, lens=None):
        hid, state = self.encoder(inp, lens=lens)
        sbatch     = [s.repeat_interleave(out.shape[1],dim=1) for s in self.pass_hiddens(state)]
        outbatch     = out.repeat(1, inp.shape[1])
        if self.source_att:
            att_features = [self.proj(hid).repeat_interleave(out.shape[1],dim=1)]
            att_tokens   = [inp.repeat_interleave(out.shape[1],dim=1)]
        else:
            att_features = None
            att_tokens   = None
        return self.decoder.logprob(outbatch,
                                    rnn_state=sbatch,
                                    att_features=att_features,
                                    att_tokens=att_tokens
                                    ).view(inp.shape[1],out.shape[1])
            #xbatch = xps.repeat_interleav(ys.shape[1],1)

    def sample(
            self,
            inp,
            max_len,
            lens=None,
            prompt=None,
            greedy=False,
            top_p=None,
            temp=1.0,
            custom_sampler=None,
            beam_size=1,
            calc_score=False,
            **kwargs):

        if beam_size > 1:
            preds = []
            scores = []
            for i in range(inp.shape[1]):
                p = self.beam(inp[:, i:i+1], beam_size)
                preds.append(p[0])
                scores.append(0)
            return preds, scores

        hid, state = self.encoder(inp, lens=lens)
        state = self.pass_hiddens(state)

        if self.source_att:
            att_features = [self.proj(hid)]
            att_tokens   = [inp]
        else:
            att_features = None
            att_tokens   = None

        return self.decoder.sample(state,
                                   max_len,
                                   att_features=att_features,
                                   att_tokens=att_tokens,
                                   temp=temp,
                                   greedy=greedy,
                                   top_p=top_p,
                                   custom_sampler=custom_sampler,
                                   calc_score=calc_score,
                                   )

    def beam(self, inp, beam_size, lens=None):
        hid, state = self.encoder(inp, lens=lens)
        state = self.pass_hiddens(state)

        if self.source_att:
            att_features = [self.proj(hid)]
            att_tokens   = [inp]
        else:
            att_features = None
            att_tokens   = None

        return self.decoder.beam(state, beam_size, 150, att_features=att_features, att_tokens=att_tokens)

    def sample_with_gumbel(self, inp, max_len, lens=None, temp=1.0, **kwargs):
        hid, state = self.encoder(inp, lens=lens)
        state = self.pass_hiddens(state)

        if self.source_att:
            att_features = [self.proj(hid)]
            att_tokens   = [inp]
        else:
            att_features = None
            att_tokens   = None

        return self.decoder.sample_with_gumbel(state,
                                               max_len,
                                               att_features=att_features,
                                               att_tokens=att_tokens,
                                               temp=temp)
