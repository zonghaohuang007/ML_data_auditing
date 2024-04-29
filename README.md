# A General Framework for Data-Use Auditing of ML Models (For Review Purpose)

### License

This software is provided solely for the purpose of anonymous review by reviewers of the 31th ACM CCS (Conference on Computer and Communications Security) who review the paper named "A General Framework for Data-Use Auditing of ML Models". By accessing, using, or reviewing this software, you agree to the following terms:

1. Review Purpose: This software is provided solely for the purpose of anonymous review by reviewers of the ACM CCS. You agree not to use, distribute, or disclose this software for any other purpose without explicit permission from the author.

2. Confidentiality: You acknowledge and agree that this software, including its source code, design, algorithms, and any associated documentation, is confidential information. You agree to keep this information confidential and not to disclose it to any third party without explicit permission from the author.

3. Restrictions: You may not copy, modify, distribute, or sublicense this software. You may not use this software for any commercial purpose or in any commercial product without explicit permission from the author.

4. No Warranty: This software is provided "as is," without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. The author shall not be liable for any damages arising from the use of this software.


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

### prepare CIFAR10, CIFAR100, TinyImageNet

Move your data dir of cifar10, cifar100, tinyimagenet to ./data/cifar10, ./data/cifar100, ./data/tinyimagenet, respectively. Their dir should include a train dir and a test dir.

### To implement and evaluate our proposed framework:
```bash
bash run.sh
```
