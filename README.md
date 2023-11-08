# Data Factors for Better Compositional Generalization

This repository contains the official code for the paper:

[Data Factors for Better Compositional Generalization](https://arxiv.org/abs/XXXX)

Xiang Zhou, Yichen Jiang and Mohit Bansal

EMNLP 2023

### Dependencies

The code is tested on Python 3.7.9 and PyTorch 1.8.1.

Other dependencies are listed in `requirements.txt` and can be installed by running `pip install -r requirements.txt`

This repository uses [wandb](https://github.com/wandb/client) for logging experiments. So before running your experiments, you need to log in you wandb account.


### Datasets and Preprocessing
The preprocessed datasets and resources used in our experiments can be downloaded at [here](https://drive.google.com/drive/folders/1XipV6sL2EpSs7RYIoUG_ldse5aSefIJq?usp=sharing).

To replicate the experiments below, please download the datasets and extract it under the root folder.

### Experiments 

We provide scripts to run experiments in our paper. All the scripts are under the `scripts` folder.

For the experiments on increasing the lexical or length complexity of the dataset in Sec. 3, the corresponding scripts are
* Increasing primitive complexity in SCAN `SCAN_transformer_huge.sh`. The corresponding baseline for SCAN can be replicated from the instructions at [owenzx/met-primaug](https://github.com/owenzx/met-primaug)
* Increasing primitive complexity in GeoQuery `GEO_transformer_prim.sh` The corresponding baseline is at `GEO_transformer_baseline.sh`
* Increasing length complexity in SCAN `SCAN_length.sh`

For the experiments changing the curriculum in Table 4 in Sec. 4, the corresponding scripts are
* On the original dataset: `GEO_curriculum_transformer.sh` and `GEO_curriculum_transformer_prim.sh`
* On the 20x augmentation dataset: `GEO_large_curriculum_transformer.sh` and `GEO_large_curriculum_transformer_prim.sh`

For the AugZero experiments in Sec. 4, the corresponding script is
* On SCAN: `SCAN_transformer_zero.sh`
* On Geoquery: `GEO_transformer_zero.sh`

For the difficulty-related experiments in Sec. 5, the corresponding scripts are
* Prototype-based difficulty experiments (Fig. 4): `ATIS_transformer_difficulty.sh`, `SMCAL_transformer_all_difficulty.sh` and `SMCAL_cs_transformer_all_difficulty.sh`
* Increasing example difficulty in SCAN (Fig. 5): `SCAN_transformer_jump_complete_len.sh` and `SCAN_tranformer_jump_complete_pe.sh`


### Acknowledgement
The code in this repository is based on [https://github.com/ekinakyurek/lexical](https://github.com/ekinakyurek/lexical)

### Reference
```
@inproceedings{zhou2023data,
  title={Data Factors for Better Compositional Generalization},
  author={Xiang Zhou and Yichen Jiang and Mohit Bansal},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2023}
}
```