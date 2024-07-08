<div align="center">

# GLOSS2TEXT
## Sign Language Gloss translation using LLMs and Semantically Aware Label Smoothing

[![arXiv](https://img.shields.io/badge/arXiv-GLOSS2TEXT-A10717.svg?logo=arXiv)](https://arxiv.org/abs/2407.01394)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

</div>

## Description
Official PyTorch implementation of the paper:
<div align="center">

[GLOSS2TEXT: Sign Language Gloss translation using LLMs and Semantically Aware Label Smoothing](https://arxiv.org/abs/2407.01394).

</div>


### Bibtex
If you find this code useful in your research, please cite:

```bibtex
@article{fayyazsanavi2024gloss2text,
  title={Gloss2Text: Sign Language Gloss translation using LLMs and Semantically Aware Label Smoothing},
  author={Fayyazsanavi, Pooya and Anastasopoulos, Antonios and Ko{\v{s}}eck{\'a}, Jana},
  journal={arXiv preprint arXiv:2407.01394},
  year={2024}
}
```

## Installation :construction_worker: 
To set up the environment, run:

```
conda create -n slt python=3.8.4
```

## Dataset :closed_book: 
Please follow the link to download the [Phoenix-2014T dataset](
https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/), the dataset is a german sign lanugae consisting the gloss and translation pairs:


## Training :rocket:
To start training, run the following command. Modify any arguments as needed:

```
python train_gls2text_nllb_lora.py
```

## Test :bar_chart:
The pre-trained model is located [here](https://drive.google.com/drive/folders/1aoiBWg0-_iQ9JaWG4uscJuTMGJyvpnSL?usp=drive_link), download it and put it in the 'pretrained' folder, 

## License :books:
Note that the code depends on other libraries, including PyTorch, HugginFace, Two-Stram Network, and use the Phonix-2014 dataset which each have their own respective licenses that must also be followed.
