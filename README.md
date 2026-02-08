# This is an official implementation of the paper [Towards-Privacy-Guaranteed-Label-Unlearning-in-Vertical-Federated-Learning](https://openreview.net/forum?id=G1JdmhkicJ&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions))

## Getting Start


### Datasets
Download the following datasets from the link provided.
Place the datasets in the .\data directory.

<!--- ModelNet: https://drive.google.com/drive/folders/14WZ7oaobP4STJkhHDWLHo6LDd9U994FX?usp=sharing -->

Brain Tumor MRI: https://drive.google.com/drive/folders/1gFVOAGlUh-sCl-wbDzzrM9G_2UwtMCHB?usp=sharing

Yahoo Answer : https://drive.google.com/drive/folders/1Frwb-ozdsDCSwUbGKuXsj5bCbd3hIp8K?usp=sharing


### Commands to train VFL model:
#### CIFAR10 Resnet18
Train Full Model:
``` 
python main.py
```

Train a retrain model in 1 label unlearning scenario:
``` 
python main.py --mode=retrain
```

Train a retrain model in 2 labels unlearning scenario:
``` 
python main.py --mode=retrain --unlearn_class_num=2
```

Train a retrain model in 4 labels unlearning scenario:
```
python main.py --mode=retrain --unlearn_class_num=4
```

#### MNIST Resnet18:
Train Full Model:
```
python main.py --data=mnist
```

Train a retrain model in 1 label unlearning scenario:
```
python main.py --data=mnist --mode=retrain
```

Train a retrain model in 2 labels unlearning scenario:
```
python main.py --data=mnist --mode=retrain --unlearn_class_num=2
```

Train a retrain model in 4 labels unlearning scenario:
```
python main.py --data=mnist --mode=retrain --unlearn_class_num=4
```


#### CIFAR100 Resnet18:
Train Full Model:
```
python main.py --data=cifar100 --num_classes=100
```

Train a retrain model in 1 label unlearning scenario:
```
python main.py --data=cifar100 --num_classes=100 --mode=retrain
```

Train a retrain model in 2 labels unlearning scenario:
```
python main.py --data=cifar100 --num_classes=100 --mode=retrain --unlearn_class_num=2
```


Train a retrain model in 4 labels unlearning scenario:
```
python main.py --data=cifar100 --num_classes=100 --mode=retrain --unlearn_class_num=4
```

#### Yahoo Answer MixText
Train Full Model:
```
python main.py --data=yahoo --model_type=mixtext --epochs=30
```

Train a retrain model:
```
python main.py --data=yahoo --model_type=mixtext --epochs=20 --mode=retrain --unlearn_class=6
```



#### Brain MRI Resnet18:
Train Full Model:
```
python main.py --data=mri --num_classes=4
```

Train a retrain model in 1 label unlearning scenario:
```
python main.py --data=mri --num_classes=4 --mode=retrain --unlearn_class=2
```

#### CIFAR10 VGG16
Train Full Model:
```
python main.py --model_type=vgg16
```

Train a retrain model in 1 label unlearning scenario:
```
python main.py --mode=retrain --model_type=vgg16
```

Train a retrain model in 2 labels unlearning scenario:
```
python main.py --mode=retrain --unlearn_class_num=2 --model_type=vgg16
```


Train a retrain model in 4 labels unlearning scenario:
```
python main.py --mode=retrain --unlearn_class_num=4 --model_type=vgg16
```


#### CIFAR100 VGG16:
Train Full Model:
```
python main.py --data=cifar100 --num_classes=100 --model_type=vgg16
```

Train a retrain model in 1 label unlearning scenario:
```
python main.py --data=cifar100 --num_classes=100 --mode=retrain --model_type=vgg16
```

Train a retrain model in 2 labels unlearning scenario:
```
python main.py --data=cifar100 --num_classes=100 --mode=retrain --unlearn_class_num=2 --model_type=vgg16
```

Train a retrain model in 4 labels unlearning scenario:
```
python main.py --data=cifar100 --num_classes=100 --mode=retrain --unlearn_class_num=4 --model_type=vgg16
```

### Before running the command for unlearning, change the saved model path directory in the torch.load() code from the unlearn python file.
### Command for unlearning
Before running the unlearning Python files, ensure you update the model path in the `torch.load()` code to point to your saved directory in the following files: `unlearn.py`, `unlearn_modelnet.py`, `unlearn_2labels.py`, and `unlearn_4labels.py`.

