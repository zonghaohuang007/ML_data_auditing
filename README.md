# A General Framework for Data-Use Auditing of ML Models

### Environment
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

```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz 
```

### Prepare CIFAR10, CIFAR100, TinyImageNet as dataset dir

Move your data of cifar10, cifar100, tinyimagenet to './data/cifar10', './data/cifar100', './data/tinyimagenet', respectively. Their dir should include a 'train' dir and a 'test' dir.

### To implement and evaluate our proposed framework:
```bash
bash run.sh
```
