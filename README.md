# The official code repo for CCS 2024 paper: ``[A general framework for data-use auditing of ML models](https://arxiv.org/pdf/2407.15100)''

## Environment
(Please feel free to reach out if you find any missing package in the below list)
```
python==3.9.15
datasets==2.16.1
numpy==1.23.5
peft==0.10.0
pillow==9.3.0
scikit-learn==1.1.2
scipy==1.8.0
tokenizers==0.15.2
torch==2.1.2
torchvision==0.16.2
transformers==4.35.2
bitsandbytes==0.43.2
clip==1.0
```
To install the package of "clip==1.0", please follow [this git repo](https://github.com/openai/CLIP):
```
pip install git+https://github.com/openai/CLIP.git
```


## To download and unzip TinyImageNet datasets

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

## To prepare raw CIFAR10, CIFAR100, TinyImageNet

Export CIFAR10 as a dataset dir in './experiments/data/cifar10':
```
python3 ./experiments/export_raw_data.py --dataset 'CIFAR10' --saved_path './experiments/data/cifar10'
```

Export CIFAR100 as a dataset dir in './experiments/data/cifar100':
```
python3 ./experiments/export_raw_data.py --dataset 'CIFAR100' --saved_path './experiments/data/cifar100'
```

Export TinyImageNet as a dataset dir in './experiments/data/tinyimagenet':
```
python3 ./experiments/export_raw_data.py --dataset 'TinyImageNet' --data_path './tiny-imagenet-200' --saved_path './experiments/data/tinyimagenet'
```

## To get access to Llama models

1. register an account in huggingface (https://huggingface.co/);
2. make a request to Meta AI (https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) (it might take one or two business days for get approved from Meta AI);
3. create an access token (https://huggingface.co/settings/tokens);
4. in your terminal, run `huggingface-cli login` and provide your access token.

## To reproduce the main experimental results of our proposed framework in the paper
Please check the notes or comments in "./experiments/run.sh" for the information on the mapping between main results in the paper and the corresponding codes used to reproduce them. 
```
bash ./experiments/run.sh
```

## To apply the marking algorithm and detection algorithm of our auditing framework as a plug-in tool

Mark your image (for marking other type of data, e.g., text, please contact the author for help):
```
from data_marking import data_marking

class Args:
    radius = 10 # epsilon: utility bound
    mepochs = 90 # number of iteration to update the mark
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

black_box_membership_inference = # define the black box membership inference metric as the score function whose return is a value, indicating the likelihood of the input being a member of the training set of the target model

published_data = # load the list of your published data

unpublished_data = # load the list of your unpublished data

detected_result = data_use_detection(model, black_box_membership_inference, published_data, unpublished_data, args)
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
