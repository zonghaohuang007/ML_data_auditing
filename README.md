## This is official code repo for the CCS 2024 paper: ``[A General Framework for Data-Use Auditing of ML Models](https://arxiv.org/pdf/2407.15100)''

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

### To apply the marking algorithm and detection algorithm of our auditing framework as a plug-in tool:

Mark your image (for marking other type of data, e.g., text, please contact the author for help):
```
from data_marking import data_marking

class Args:
    radius = 10 # epsilon: utility bound
    mepoch = 90 # number of iteration to update the mark
    raw_image_path = <the path of raw image>
    published_path = <the path of saving your published image>
    unpublished_path = <the path of saving your unpublished image>
args=Args()
data_marking(args)
```

Detect the use of your data in a target ML model:
```
from data_use_detection import data_use_detection

class Args:
    p = 0.05 # the upper bound on false-detection rate
    published_data_dir = <the dir path where the set of your published data is saved>
    unpublished_data_dir = <the dir path where the set of your unpublished data is saved>
args=Args()

model = # load the target model or provide its API
black_box_membership_inference = # define the black box membership inference metric as the score function whose return is a value, indicating the likelihood of the input
published_data = # load the list of your published data
unpublished_data = # load the list of your unpublished data

detected_result = data_use_detection(model, black_box_membership_inference, published_data, unpublished_data)
if detected_result:
    print('The target model uses your published data, under a false-detection rate less than {}'.format(args.p))
else:
    print('The target model does not use your published data')
```


If you have any question on our work or this repo, please feel free to email the author. 
If you find this git repo is helpful for your research, please consider to cite:
```
@inproceedings{huang2024:auditdata,
  title={A general framework for data-use auditing of ML models},
  author={Huang, Z. and Gong, N. Z. and Reiter, M. K.},
  booktitle={31\textsuperscript{st}} ACM Conference on Computer and Communications Security,
  year={2024}
}
