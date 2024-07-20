## This is official code repo for the CCS 2024 paper: ``A General Framework for Data-Use Auditing of ML Models''

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

### To download CIFAR10, CIFAR100, TinyImageNet datasets

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz 
```

### To prepare raw CIFAR10, CIFAR100, TinyImageNet

Export CIFAR10 as a dataset dir in './experiments/data/cifar10':
```
python3 ./experiments/export_raw_data.py --dataset 'CIFAR10' --data_path <path where cifar-10-python.tar.gz is downloaded and unzipped> --saved_path './experiments/data/cifar10'
```

Export CIFAR100 as a dataset dir in './experiments/data/cifar100':
```
python3 ./experiments/export_raw_data.py --dataset 'CIFAR100' --data_path <path where cifar-100-python.tar.gz is downloaded and unzipped> --saved_path './experiments/data/cifar100'
```

Export TinyImageNet as a dataset dir in './experiments/data/tinyimagenet':
```
python3 ./experiments/export_raw_data.py --dataset 'TinyImageNet' --data_path <path where tiny-imagenet-200.zip is downloaded and unzipped> --saved_path './experiments/data/tinyimagenet'
```

### To reproduce the main experimental results of our proposed framework in the paper:
(In "./experiments/run.sh", it is needed to set the data path where the dataset is downloaded and unzipped)
```
bash ./experiments/run.sh
```


If you have any question on our work or this repo, please feel free to email the author.


If you find this git repo is helpful for your research, please consider to cite:
```
@inproceedings{huang2024:auditdata,
  title={A general framework for data-use auditing of ML models},
  author={Huang, Z. and Gong, N. Z. and Reiter M. K.},
  booktitle={31\textsuperscript{st}} ACM Conference on Computer and Communications Security,
  year={2024}
}