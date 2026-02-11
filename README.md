# This is an official implementation of the paper [Towards-Privacy-Guaranteed-Label-Unlearning-in-Vertical-Federated-Learning](https://openreview.net/forum?id=G1JdmhkicJ&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions))

## Getting Start


### Datasets
Download the following datasets from the link provided.
Place the datasets in the .\data directory.

<!--- ModelNet: https://drive.google.com/drive/folders/14WZ7oaobP4STJkhHDWLHo6LDd9U994FX?usp=sharing -->

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
Before running the unlearning Python files, ensure you update the model path in the `torch.load()` code to point to your saved directory in the following files: `unlearn.py`, `unlearn_modelnet.py`, `unlearn_2labels.py`, and `unlearn_4labels.py`.

## Citation

```bibtex
@InProceedings{Hong_2026_ICLR,
    author    = {Gu, Hanlin and Tae, Hongxi and and Fan, Lixin and Chan, Chee Seng},
    title     = {Towards Privacy-Guaranteed Label Unlearning in Vertical Federated Learning: Few-Shot Forgetting Without Disclosure},
    booktitle = {The Fourteenth International Conference on Learning Representations (ICLR)},
    year      = {2026}
}
```



