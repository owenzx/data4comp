import torch
from typing import Tuple
import numpy as np
import pickle as pkl
import os


ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))



def get_root_and_suffix(s):
    j = len(s) - 1
    while str.isnumeric(s[j]):
        j -= 1
    return s[:j+1], s[j+1:]



def update_substitutor(token2tag, tag2token, vocab_x, vocab_y):
    for x in vocab_x._contents.keys():
        if x in {vocab_x.PAD, vocab_x.SOS, vocab_x.EOS, vocab_x.COPY, vocab_x.UNK, vocab_x.MASK}:
            continue
        if x not in token2tag:
            print(x)
            ogn_token, suffix = get_root_and_suffix(x)
            assert(ogn_token in token2tag)
            ogn_tag = token2tag[ogn_token]
            if (ogn_tag + suffix) in vocab_y._contents.keys():
                new_tag = ogn_tag + suffix
            else:
                new_tag = ogn_tag

            token2tag[x] = new_tag
            if new_tag  not in tag2token:
                tag2token[new_tag] = []
            tag2token[new_tag].append(x)
    return token2tag, tag2token



class AugBatchGenerator(object):
    def __init__(self, vocab_x, vocab_y, perturbations=None, use_cogs_substitutor=False, use_scan_substitutor=False):
        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.PAD_x = vocab_x.pad()

        if perturbations is None:
            self.perturbations = ['edit', 'add', 'delete']
        else:
            self.perturbations = perturbations

        self.use_cogs_substitutor = use_cogs_substitutor
        self.use_scan_substitutor = use_scan_substitutor

        if use_cogs_substitutor:
            self.perturbations = ['edit']
            with open(f'{ROOT_FOLDER}/../comp-data/tags/cogs_word2tag.pkl', 'rb') as fr:
                self.cogs_token2tag = pkl.load(fr)
            with open(f'{ROOT_FOLDER}/../comp-data/tags/cogs_tag2word.pkl', 'rb') as fr:
                self.cogs_tag2word = pkl.load(fr)
                for tag in self.cogs_tag2word.keys():
                    self.cogs_tag2word[tag] = list(self.cogs_tag2word[tag])

            self.cogs_token2tag, self.cogs_tag2word = update_substitutor(self.cogs_token2tag, self.cogs_tag2word, self.vocab_x, self.vocab_y)


        if use_scan_substitutor:
            self.perturbations = ['edit']
            if 'dax' in self.vocab_x:
                pkl_type = 'dax'
            else:
                pkl_type = 'normal'
            with open(f'{ROOT_FOLDER}/../comp-data/tags/scan_{pkl_type}_word2tag.pkl', 'rb') as fr:
                self.scan_token2tag = pkl.load(fr)
            with open(f'{ROOT_FOLDER}/../comp-data/tags/scan_{pkl_type}_tag2word.pkl', 'rb') as fr:
                self.scan_tag2word = pkl.load(fr)
                for tag in self.scan_tag2word.keys():
                    self.scan_tag2word[tag] = list(self.scan_tag2word[tag])
            self.scan_token2tag, self.scan_tag2word = update_substitutor(self.scan_token2tag, self.scan_tag2word, self.vocab_x, self.vocab_y)



    def get_random_real_token_x_cogs(self, origin_tok_id):
        tok = self.vocab_x.get(int(origin_tok_id))
        tag = self.cogs_token2tag[tok]
        random_tok_num = len(self.cogs_tag2word[tag])
        if random_tok_num == 1:
            random_new_token_id = self.get_random_real_token_x()
        else:
            random_new_token = np.random.choice(self.cogs_tag2word[tag])
            random_new_token_id = self.vocab_x[random_new_token]
        while random_new_token_id == int(origin_tok_id):
            if random_tok_num == 1:
                random_new_token_id = self.get_random_real_token_x()
            else:
                random_new_token = np.random.choice(self.cogs_tag2word[tag])
                random_new_token_id = self.vocab_x[random_new_token]
        return random_new_token_id

    def get_random_real_token_x_scan(self, origin_tok_id):
        tok = self.vocab_x.get(int(origin_tok_id))
        tag = self.scan_token2tag[tok]
        random_tok_num = len(self.scan_tag2word[tag])
        if random_tok_num == 1:
            random_new_token_id = self.get_random_real_token_x()
        else:
            random_new_token = np.random.choice(self.scan_tag2word[tag])
            random_new_token_id = self.vocab_x[random_new_token]
        while random_new_token_id == int(origin_tok_id):
            if random_tok_num == 1:
                random_new_token_id = self.get_random_real_token_x()
            else:
                random_new_token = np.random.choice(self.scan_tag2word[tag])
                random_new_token_id = self.vocab_x[random_new_token]
        print(random_new_token_id)
        return random_new_token_id


    def get_random_real_token_x(self):
        vocab_size = len(self.vocab_x)
        return np.random.randint(5, vocab_size)

    def get_random_real_token_y(self):
        vocab_size = len(self.vocab_y)
        return np.random.randint(5, vocab_size)



    def _insert_tok_to_vec(self, vec, tok, position):
        new_vec = torch.cat([vec[:position], torch.Tensor([tok]).to(vec), vec[position:]])
        return new_vec

    def _delete_tok_from_vec(self, vec, position):
        new_vec = torch.cat([vec[:position], vec[position+1:]])
        return new_vec


    def generate_batch_with_perturbed_input(self, inp, out, lens, sort_results=True):


        perturbations = self.perturbations


        batch_size = len(lens)

        batch_first_inp = inp.clone().transpose(0, 1).cpu()
        batch_first_out = out.clone().transpose(0, 1).cpu()
        aug_lens = lens.clone().cpu()


        rand_perturbation = np.random.randint(0,len(perturbations), batch_size)
        rand_position = np.random.randint(1, aug_lens-1) #skip bos and eos
        rand_position_for_add = np.random.randint(1, aug_lens)

        batch_first_inp_list = [inp for inp in batch_first_inp]
        batch_first_out_list = [out for out in batch_first_out]



        for i in range(batch_size):
            example_len = aug_lens[i]
            if example_len > 3: # 1 + 1 (bos) + 1 (eos)
                selected_perturbation = perturbations[rand_perturbation[i]]
            elif example_len == 3:
                selected_perturbation = perturbations[rand_perturbation[i]%2]
            else:
                raise ValueError


            if selected_perturbation == 'edit':
                if self.use_cogs_substitutor:
                    new_tok = self.get_random_real_token_x_cogs(batch_first_inp_list[i][rand_position[i]].cpu().numpy())
                elif self.use_scan_substitutor:
                    new_tok = self.get_random_real_token_x_scan(batch_first_inp_list[i][rand_position[i]].cpu().numpy())
                else:
                    new_tok = self.get_random_real_token_x()
                while new_tok == batch_first_inp_list[i][rand_position[i]]:
                    if self.use_cogs_substitutor:
                        new_tok = self.get_random_real_token_x_cogs(batch_first_inp_list[i][rand_position[i]].cpu().numpy())
                    elif self.use_scan_substitutor:
                        new_tok = self.get_random_real_token_x_scan(batch_first_inp_list[i][rand_position[i]].cpu().numpy())
                    else:
                        new_tok = self.get_random_real_token_x()
                batch_first_inp_list[i][rand_position[i]] =  new_tok
            elif selected_perturbation == 'add':
                new_tok = self.get_random_real_token_x()
                batch_first_inp_list[i] = self._insert_tok_to_vec(batch_first_inp_list[i], new_tok, rand_position_for_add[i])
                aug_lens[i] += 1
            elif selected_perturbation == 'delete':
                batch_first_inp_list[i] = self._delete_tok_from_vec(batch_first_inp_list[i], rand_position[i])
                aug_lens[i] -= 1

        max_len = max(aug_lens)
        for i in range(batch_size):
            if len(batch_first_inp_list[i]) > max_len:
                batch_first_inp_list[i] = batch_first_inp_list[i][:max_len]
            elif len(batch_first_inp_list[i]) < max_len:
                batch_first_inp_list[i] = torch.cat([batch_first_inp_list[i], torch.Tensor([self.PAD_x]*(max_len - len(batch_first_inp_list[i]))).to(batch_first_inp_list[i])])


        lens_list = [l for l in aug_lens]

        all_batch_list = list(zip(batch_first_inp_list, batch_first_out_list, lens_list))

        if sort_results:
            all_batch_list = sorted(all_batch_list, key=lambda x:x[2], reverse=True)
            batch_first_inp_list = [x[0] for x in all_batch_list]
            batch_first_out_list = [x[1] for x in all_batch_list]
            lens_list = [x[2] for x in all_batch_list]



        new_batch_first_inp = torch.stack(batch_first_inp_list)
        new_inp = new_batch_first_inp.transpose(0,1).to(inp).contiguous()

        new_batch_first_out = torch.stack(batch_first_out_list)
        new_out = new_batch_first_out.transpose(0,1).to(out).contiguous()

        new_lens = torch.stack(lens_list)
        new_lens = new_lens.to(lens).contiguous()


        return new_inp, new_out, new_lens



    def generate_masked_batch(self, inp, out,lens, vocab_x, mlm_prob):
        padding_masks = (inp == vocab_x.pad()).to(inp.device)
        masked_inp, masked_labels, masked_indices = self._mask_tokens(inp, padding_masks, mask_id=vocab_x.mask(),
                                                                vocab_size=len(vocab_x) - 1,
                                                                mlm_probability=mlm_prob)  # small hacking: vocab size - 1 to remove the mask token
        return masked_inp, out, lens, {'masked_indices': masked_indices}




    def _mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: torch.Tensor, mask_id, vocab_size, mlm_probability=0.15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
        special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8).to(inputs.device)).bool() & masked_indices
        inputs[indices_replaced] = mask_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5).to(inputs.device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long).to(inputs.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, masked_indices
