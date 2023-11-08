import torch
import torch.utils.data as torch_data
from data import collate_with_both_lens
import numpy as np



class CurriculumSetter(object):

    def __init__(self,
                 curriculum_type,
                 train_data,
                 origin_train_text,
                 augment_in_order,
                 augment_times,
                 n_batch,
                 init_training_portion=1.0,
                 curriculum_ending_time=0.9):
        self.curriculum_type = curriculum_type
        self.augment_in_order = augment_in_order
        self.augment_times = augment_times
        self.n_batch = n_batch

        self.init_training_portion = init_training_portion
        self.curriculum_ending_time = curriculum_ending_time


        self.origin_train_text = origin_train_text
        self.train_data = train_data

        assert(len(self.origin_train_text) == len(self.train_data))


        self.annotated_dataset = self.annotate_dataset(self.origin_train_text, self.train_data)

        self.set_init_trainining_set(self.init_training_portion)







    def set_init_trainining_set(self, init_training_portion):
        if self.curriculum_type == 'static_aug':
            return
        if self.curriculum_type == 'always_novel':
            np.random.shuffle(self.annotated_dataset)
            total_size = len(self.annotated_dataset)
            init_size = int(total_size * init_training_portion)
            self.dynamic_train_subset = [self.annotated_dataset[i] for i in range(init_size)]
            self.remaining_example_pointer = init_size
            return
        if self.curriculum_type == 'always_novel_prim':
            np.random.shuffle(self.annotated_dataset)

            self.prim2example = {}
            for i, ex in enumerate(self.annotated_dataset):
                for p in ex['prim_list']:
                    if p not in self.prim2example:
                        self.prim2example[p] = []
                    self.prim2example[p].append(i)
            self.total_prims = list(self.prim2example.keys())
            np.random.shuffle(self.total_prims)
            total_prim_size = len(self.total_prims)
            init_prim_size = int(total_prim_size * init_training_portion)

            self.dynamic_train_subset = []
            self.added_indices = set()
            for i in range(init_prim_size):
                for ex_i in self.prim2example[self.total_prims[i]]:
                    if ex_i not in self.added_indices:
                        self.added_indices.add(ex_i)
                        self.dynamic_train_subset.append(self.annotated_dataset[ex_i])

            self.remaining_prim_pointer = init_prim_size
            return


    def annotate_dataset(self, text_dataset, preprocessed):
        dataset_with_prim_annotation = []

        dataset_with_embs = list(zip(text_dataset, preprocessed))

        if self.augment_in_order:
            assert(self.augment_times is not None)
            augment_times = self.augment_times

            total_size = len(text_dataset)
            assert(total_size % augment_times == 0)
            origin_size = total_size // augment_times

            for i in range(origin_size):
                origin_example_idx =  i * augment_times

                augs = [dataset_with_embs[origin_example_idx + j] for j in range(1, augment_times)]

                ex_dict = {'origin': dataset_with_embs[origin_example_idx],
                           'augmentations': augs}
                dataset_with_prim_annotation.append(ex_dict)

        else:
            # otherwise, do not distrinuish between augmented examples and original examples
            total_size = len(text_dataset)
            for i in range(total_size):
                #this only applies to GEO
                text = dataset_with_embs[i][0]
                tgt = text[1]
                prim_list = []
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

                dataset_with_prim_annotation.append({'origin': dataset_with_embs[i],
                                                     'prim_list': prim_list})

        return dataset_with_prim_annotation


    def get_train_loader_with_curriculum(self, epoch_num, cur_step, max_step):
        if self.curriculum_type == 'static_aug':
            train_dataset= []

            #First add the original examples
            np.random.shuffle(self.annotated_dataset)

            for ex in self.annotated_dataset:
                train_dataset.append(ex['origin'][1])

            for i in range(1, self.augment_times):
                for ex in self.annotated_dataset:
                    train_dataset.append(ex['augmentations'][i-1][1])

            train_loader = torch_data.DataLoader(
                train_dataset,
                batch_size=self.n_batch,
                shuffle=False,
                collate_fn=collate_with_both_lens
            )

        elif self.curriculum_type == 'always_novel':
            # The main idea here is to gradually add the novel examples into the training, so the model always have something new to learn from
            past_step_portion = min(1, cur_step / (self.curriculum_ending_time * max_step))
            if past_step_portion < self.init_training_portion:
                pass
            else:
                for i in range(self.remaining_example_pointer,
                                   int(len(self.annotated_dataset) * past_step_portion)):
                    self.dynamic_train_subset.append(self.annotated_dataset[i])
                    self.remaining_example_pointer += 1

            train_dataset = [ex['origin'][1] for ex in self.dynamic_train_subset]

            train_loader = torch_data.DataLoader(
                train_dataset,
                batch_size=self.n_batch,
                shuffle=True,
                collate_fn=collate_with_both_lens
            )
        elif self.curriculum_type == 'always_novel_prim':
            past_step_portion = min(1, cur_step / (self.curriculum_ending_time * max_step))
            if past_step_portion < self.init_training_portion:
                pass
            else:
                for i in range(self.remaining_prim_pointer, int(len(self.total_prims) * past_step_portion)):
                    self.remaining_prim_pointer += 1
                    for ex_i in self.prim2example[self.total_prims[i]]:
                        if ex_i not in self.added_indices:
                            self.added_indices.add(ex_i)
                            self.dynamic_train_subset.append(self.annotated_dataset[ex_i])

            train_dataset = [ex['origin'][1] for ex in self.dynamic_train_subset]

            train_loader = torch_data.DataLoader(
                train_dataset,
                batch_size=self.n_batch,
                shuffle=True,
                collate_fn=collate_with_both_lens
            )



        else:
            raise ValueError


        return train_loader


