import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src import EncDec, RecordLoss, TransformerEncDec
from data import eval_format
import collections
import wandb
import higher
from src import AugBatchGenerator

LossTrack = collections.namedtuple('LossTrack', 'nll mlogpyx pointkl')


def nll_to_prob(nll):
    logprob = -nll
    prob = torch.exp(logprob)
    return prob

class MetaSeq2SeqWrapper4Unlikelihood(nn.Module):
    def __init__(self,
                 seq2seq):
        super().__init__()
        self.seq2seq = seq2seq

    def forward(self, inp, out, lens=None, recorder=None, unlikelihood=False):
        if unlikelihood:
            perinstance_loss = self.seq2seq(inp, out, lens = lens, additional_output=False, per_instance=True)
            perinstance_logp = -  perinstance_loss
            unlikelihood = 1 - torch.exp(perinstance_logp)
            log_unlikelihood = (-torch.log(unlikelihood)).mean()

            return log_unlikelihood
        else:
            loss = self.seq2seq(inp, out, lens=lens, additional_output=False)
            return loss


class MetaSeq2SeqWrapper(nn.Module):
    def __init__(self,
                 seq2seq):
        super().__init__()
        self.seq2seq = seq2seq



    def forward(self, inp, out, lens=None, recorder=None, use_updated_emb=False, detach_mask=None, additional_output=False):
        outputs = self.seq2seq(inp, out, lens=lens, additional_output=additional_output, use_updated_emb=use_updated_emb, detach_mask=detach_mask)
        if additional_output:
            loss, additional_dict = outputs
            embs = additional_dict['enc_emb']
            return loss, embs
        else:
            loss = outputs
            return loss




class MetaWrapper(nn.Module):
    def __init__(self,
                 vocab_x,
                 vocab_y,
                 emb,
                 dim,
                 copy=False,
                 temp=1.0,
                 max_len_x=8,
                 max_len_y=8,
                 n_layers=1,
                 attention_type = 'simple',
                 model_type = 'lstm',
                 transformer_config = '2layer',
                 self_att=False,
                 attention=True,
                 dropout=0.,
                 bidirectional=True,
                 n_decoder_layers=0,
                 rnntype=nn.LSTM,
                 recorder=RecordLoss(),
                 meta=None,
                 unlikelihood=None,
                 n_inner_iter=None,
                 mlm_prob=None,
                 meta_loss_weight=None,
                 ul_loss_weight=None,
                 meta_loss_type=None,
                 inner_lr = None,
                 cogs_perturbation=False,
                 scan_perturbation=False,
                 ):

        super().__init__()


        self.meta = meta
        self.unlikelihood = unlikelihood
        self.n_inner_iter = n_inner_iter
        # TODO add n_inner_iter
        assert(self.n_inner_iter > 0)
        self.mlm_prob = mlm_prob
        self.meta_loss_weight = meta_loss_weight
        self.ul_loss_weight = ul_loss_weight
        self.meta_loss_type = meta_loss_type

        if self.meta_loss_type not in ['maml_unlikelihood']:
            assert(self.meta_loss_weight == self.ul_loss_weight)

        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.rnntype = rnntype
        self.bidirectional = bidirectional
        self.dim = dim
        self.n_layers = n_layers
        self.temp = temp
        self.MAXLEN_X = max_len_x
        self.MAXLEN_Y = max_len_y
        self.model_type = model_type



        if self.model_type == 'lstm':
            self.pyx = EncDec(vocab_x,
                              vocab_y,
                              emb,
                              dim,
                              copy=copy,
                              n_layers=n_layers,
                              self_att=self_att,
                              source_att=attention,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              attention_type=attention_type,
                              rnntype=rnntype,
                              MAXLEN=self.MAXLEN_Y,
                              n_decoder_layers=n_decoder_layers)
        elif self.model_type == 'transformer':
            self.pyx = TransformerEncDec(vocab_x, vocab_y, MAXLEN=self.MAXLEN_Y, transformer_config=transformer_config)
        else:
            raise ValueError

        aug_batch_kwards = {"use_cogs_substitutor": cogs_perturbation,
                            "use_scan_substitutor": scan_perturbation}



        if self.meta:
            if self.meta_loss_type == 'unmaml':
                self.aux_batch_source = 'perturb'
                self._calc_loss = self._calc_loss_unmaml
                self.use_double_batch = False
                self.aug_batch_generator = AugBatchGenerator(vocab_x, vocab_y, **aug_batch_kwards)
                self.pyx4meta = MetaSeq2SeqWrapper4Unlikelihood(self.pyx)
            else:
                raise ValueError
        elif self.unlikelihood:
            if self.meta_loss_type == 'unlikelihood':
                self.aux_batch_source = 'perturb'
                self._calc_loss = self._calc_loss_unlikelihood
                self.use_double_batch = False
                self.aug_batch_generator = AugBatchGenerator(vocab_x, vocab_y, **aug_batch_kwards)
            elif self.meta_loss_type == 'unlikelihood2':
                self.aux_batch_source = 'perturb'
                self._calc_loss = self._calc_loss_unlikelihood_impl2
                self.use_double_batch = False
                self.aug_batch_generator = AugBatchGenerator(vocab_x, vocab_y, **aug_batch_kwards)
            elif self.meta_loss_type == 'unlikelihood3':
                self.aux_batch_source = 'perturb'
                self._calc_loss = self._calc_loss_unlikelihood_impl3
                self.use_double_batch = False
                self.aug_batch_generator = AugBatchGenerator(vocab_x, vocab_y, **aug_batch_kwards)

            else:
                raise ValueError
        else:
            # baseline model, just do supervised training
            self.aux_batch_source = None
            self._calc_loss = self._calc_loss_mle
            self.use_double_batch = False

        self.inner_lr = inner_lr

        self.recorder = recorder




    def forward(self, inp, out, lens=None, recorder=None, additional_output=False, per_instance=False):
        return self.pyx(inp, out, lens=lens, additional_output=additional_output, per_instance=per_instance)





    def _calc_loss_unmaml(self, inp, out, lens, aux_inp, aux_out, aux_lens, accum_count, current_lr):
        with torch.backends.cudnn.flags(enabled=False):
            normal_loss = self.forward(inp, out, lens=lens, additional_output=False)

            (normal_loss / accum_count).backward()

            if self.inner_lr is not None:
                inner_opt = torch.optim.SGD(self.pyx.parameters(), lr=self.inner_lr)
            else:
                assert(current_lr is not None)
                inner_opt = torch.optim.SGD(self.pyx.parameters(), lr=current_lr[0])

            total_meta_loss = 0

            with higher.innerloop_ctx(self.pyx4meta, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                loss = fnet(inp, out, lens=lens, unlikelihood=False)
                diffopt.step(loss)

                meta_loss = fnet(aux_inp, aux_out, lens=aux_lens, unlikelihood=True)

                (meta_loss * self.meta_loss_weight / accum_count).backward()

                total_meta_loss += meta_loss

        total_loss = self.meta_loss_weight * total_meta_loss + normal_loss
        return total_loss, normal_loss, total_meta_loss


    def _calc_loss_unlikelihood(self, inp, out, lens, aux_inp, aux_out, aux_lens, accum_count, current_lr):
        normal_loss = self.forward(inp, out, lens=lens, additional_output=False)
        perinstance_loss = self.forward(aux_inp, aux_out, lens = aux_lens, additional_output=False, per_instance=True)
        perinstance_logp = -  perinstance_loss
        unlikelihood = 1 - torch.exp(perinstance_logp)
        log_unlikelihood = (-torch.log(unlikelihood)).mean()


        total_loss = self.meta_loss_weight * log_unlikelihood + normal_loss

        (total_loss / accum_count).backward()

        return total_loss, normal_loss, log_unlikelihood


    def _calc_loss_unlikelihood_impl2(self, inp, out, lens, aux_inp, aux_out, aux_lens, accum_count, current_lr):
        output_mask = (out==self.vocab_y.pad()).to(out.device)
        aux_output_mask = (aux_out==self.vocab_y.pad()).to(aux_out.device)
        aux_output_loss_mask = aux_output_mask[1:, :].transpose(0,1)


        normal_loss = self.forward(inp, out, lens=lens, additional_output=False)
        perinstance_loss, add_dict = self.forward(aux_inp, aux_out, lens = aux_lens, additional_output=True, per_instance=True)
        perstep_loss = add_dict['per_step_loss']
        perstep_logp = -perstep_loss
        unlikelihood = 1 - torch.exp(perstep_logp)
        unlikelihood = unlikelihood + aux_output_loss_mask*0.0000001
        log_unlikelihood = -torch.log(unlikelihood) * (~aux_output_loss_mask)
        sum_loss = log_unlikelihood.sum(dim=-1).mean()

        total_loss = self.meta_loss_weight * sum_loss + normal_loss

        (total_loss / accum_count).backward()

        return total_loss, normal_loss, sum_loss


    def _calc_loss_unlikelihood_impl3(self, inp, out,lens, aux_inp, aux_out, aux_lens, accum_count, current_lr):
        output_mask = (out==self.vocab_y.pad()).to(out.device)
        aux_output_mask = (aux_out==self.vocab_y.pad()).to(aux_out.device)
        aux_output_loss_mask = aux_output_mask[1:, :].transpose(0,1)


        normal_loss = self.forward(inp, out, lens=lens, additional_output=False)
        perinstance_loss, add_dict = self.forward(aux_inp, aux_out, lens = aux_lens, additional_output=True, per_instance=True)
        perstep_loss = add_dict['per_step_loss']
        perstep_logp = -perstep_loss
        min_perstep_logp, _ = perstep_logp.min(dim=-1)
        unlikelihood = 1 - torch.exp(min_perstep_logp)
        log_unlikelihood = -torch.log(unlikelihood)
        sum_loss = log_unlikelihood.mean()

        total_loss = self.meta_loss_weight * sum_loss + normal_loss

        (total_loss / accum_count).backward()

        return total_loss, normal_loss, sum_loss





    def _calc_loss_mle_position(self, inp, out, lens):
        _, add_dict = self.forward(inp, out, lens=lens, per_instance=True, additional_output=True)
        normal_loss = add_dict['per_step_loss']
        return normal_loss

    def _calc_loss_mle(self, inp, out, lens, aux_inp, aux_out, aux_lens, accum_count, current_lr):
        loss_pyx = self.forward(inp, out, lens=lens, additional_output=False)
        return loss_pyx, loss_pyx, 0

    def _pair_current_tgt_with_negative_src(self, cur_inp, cur_out, cur_lens, aux_batch):
        aux_inp, aux_out, aux_lens = aux_batch
        return aux_inp, cur_out, aux_lens





    def train_forward(self, inp, out, lens=None, y_lens=None, recorder=None, accum_count=1, current_lr=None, aux_batch=None, return_aux_batch=False, return_position_prob=False, sort_aux_batch=True):
        additional_dict = {}
        if self.aux_batch_source is None:
            aux_inp, aux_out, aux_lens = None, None, None
        elif self.aux_batch_source in ['both']:
            aux_inp, aux_out, aux_lens, index = aux_batch
            ul_inp, ul_out, ul_lens = self.aug_batch_generator.generate_batch_with_perturbed_input(inp, out, lens, sort_results=sort_aux_batch)
            additional_dict = {'ul_inp': ul_inp,
                               'ul_out': ul_out,
                               'ul_lens': ul_lens}
        elif self.aux_batch_source in ['dataset', 'loadidx']:
            aux_inp, aux_out, aux_lens, index = aux_batch
        elif self.aux_batch_source in ['multi_dataset']:
            aux_inp, aux_out, aux_lens, index = aux_batch['outer_batch']
            aux_inner_inp, aux_inner_out, aux_inner_lens, inner_index = aux_batch['inner_batch']
            additional_dict = {'aux_inner_inp': aux_inner_inp,
                               'aux_inner_out': aux_inner_out,
                               'aux_inner_lens': aux_inner_lens}
        elif self.aux_batch_source in ['dataset_only_source']:
            aux_inp, aux_out, aux_lens = self._pair_current_tgt_with_negative_src(inp, out, lens, aux_batch)
        elif self.aux_batch_source in ['mask']:
            aux_inp, aux_out, aux_lens, additional_dict = self.aug_batch_generator.generate_masked_batch(inp, out, lens, vocab_x=self.vocab_x, mlm_prob=self.mlm_prob)
        elif self.aux_batch_source in ['perturb']:
            aux_inp, aux_out, aux_lens = self.aug_batch_generator.generate_batch_with_perturbed_input(inp, out, lens, sort_results=sort_aux_batch)
        else:
            raise NotImplementedError

        if return_position_prob:
            # return position-level loss for analysis
            if return_aux_batch:
                aux_batch = {'inp': aux_inp,
                             'out': aux_out,
                             'lens': aux_lens}
                normal_loss = self._calc_loss_mle_position(inp, out, lens)
                aux_loss = self._calc_loss_mle_position(aux_inp, aux_out, aux_lens)
                normal_prob = nll_to_prob(normal_loss)
                aux_prob = nll_to_prob(aux_loss)
                return normal_prob, aux_prob, aux_batch
            else:
                normal_loss = self._calc_loss_mle_position(inp, out, lens)
                normal_prob = nll_to_prob(normal_loss)
                return normal_prob

        else:
            total_loss, train_only_loss, other_loss = self._calc_loss(inp, out, lens, aux_inp, aux_out, aux_lens, accum_count, current_lr, **additional_dict)

            if return_aux_batch:
                aux_batch = {'inp': aux_inp,
                             'out': aux_out,
                             'lens': aux_lens}
                return total_loss, train_only_loss, other_loss, aux_batch
            else:
                return total_loss, train_only_loss, other_loss


    def print_tokens(self, vocab, tokens):
        return [" ".join(eval_format(vocab, tokens[i])) for i in range(len(tokens))]

    def sample(self, *args, **kwargs):
        return self.pyx.sample(*args, **kwargs)
