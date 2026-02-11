# Federated Label Unlearning

### Official pytorch implementation of the paper: "Towards Privacy-Guaranteed Label Unlearning in Vertical Federated Learning: Few-Shot Forgetting Without Disclosure"

#### ICLR 2026 [(OpenReview)](https://openreview.net/forum?id=G1JdmhkicJ&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)) | [(ArXiv)](https://arxiv.org/abs/2410.10922) | 

#### (Released on February 08, 2026)

## Introduction
We tackle label unlearning in Vertical Federated Learning (VFL), where labels are both necessary inputs and sensitive information. We introduce a representation-level manifold mixup to synthesize embeddings for unlearned and retained samples, providing stronger signals for efficient gradient-based forgetting and recovery. Our method removes label information while preserving utility via a lightweight recovery optimization, and scales across diverse datasets (e.g., MNIST, CIFAR-10/100, ModelNet, medical imaging, and Yahoo Answers).

## Getting Start

### Datasets
Download the following datasets from the link provided.
Place the datasets in the .\data directory.

Brain Tumor MRI: https://drive.google.com/drive/folders/1gFVOAGlUh-sCl-wbDzzrM9G_2UwtMCHB?usp=sharing

Yahoo Answer : https://drive.google.com/drive/folders/1Frwb-ozdsDCSwUbGKuXsj5bCbd3hIp8K?usp=sharing


### Commands to train VFL model:
Train Full Model:
``` 
python main.py
```

Train a retrain model in 1 label unlearning scenario:
``` 
python main.py --mode=retrain
```

You may specify different data with --data=`<data name>`


### Before running the command for unlearning, change the saved model path directory in the torch.load() code from the unlearn python file.
### Command for unlearning
Before running the unlearning Python files, ensure you update the model path in the `torch.load()` code to point to your saved directory in the following files: `unlearn.py`, `unlearn_2labels.py`, and `unlearn_4labels.py`.

## Citation

```bibtex
@InProceedings{Hong_2026_ICLR,
    author    = {Gu, Hanlin and Tae, Hongxi and and Fan, Lixin and Chan, Chee Seng},
    title     = {Towards Privacy-Guaranteed Label Unlearning in Vertical Federated Learning: Few-Shot Forgetting Without Disclosure},
    booktitle = {The Fourteenth International Conference on Learning Representations (ICLR)},
    year      = {2026}
}
```
## Feedback
Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the author by sending an email to
`cs.chan at um.edu.my`

# License and Copyright

The project is open source under BSD-3 license (see the `LICENSE` file).

©2026 Universiti Malaya.








