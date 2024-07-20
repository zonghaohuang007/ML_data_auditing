# This is official code repo for the CCS 2024 paper: ``A General Framework for Data-Use Auditing of ML Models''

### Environment
(Please feel free to reach out if you find any missing package in the below list)
```
python==3.9.15
datasets==2.16.1
numpy==1.23.5
openai==1.10.0
peft==0.10.0
pillow==9.3.0
scikit-learn==1.1.2
scipy==1.8.0
tokenizers==0.15.2
torch==1.13.0
torchvision==0.16.2
transformers==4.35.2
```

### To download CIFAR10, CIFAR100, TinyImageNet dataset

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz 
```

### Prepare raw CIFAR10, CIFAR100, TinyImageNet

To prepare CIFAR10 and export it as a dataset dir in './experiments/data/cifar10', please run:
```
python3 ./experiments/export_raw_data.py --dataset 'CIFAR10' --data_path <path where cifar-10-python.tar.gz is downloaded and unzipped> --saved_path './experiments/data/cifar10'
```

To prepare CIFAR100 and export it as a dataset dir in './experiments/data/cifar100', please run:
```
python3 ./experiments/export_raw_data.py --dataset 'CIFAR100' --data_path <path where cifar-100-python.tar.gz is downloaded and unzipped> --saved_path './experiments/data/cifar100'
```

To prepare TinyImageNet and export it as a dataset dir in './experiments/data/tinyimagenet', please run:
```
python3 ./experiments/export_raw_data.py --dataset 'TinyImageNet' --data_path <path where tiny-imagenet-200.zip is downloaded and unzipped> --saved_path './experiments/data/tinyimagenet'
```

### To implement and evaluate our proposed framework:
```
bash run.sh
```
