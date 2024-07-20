# Please note that it will cost weeks or even months to finish all the experiments. It is better to run them in parallel to save time.

# auditing cifar10 in image classifier
python3 ./experiments/classifier_cifar10.py --data_path <path where cifar-10-python.tar.gz is downloaded and unzipped> --mark_budget 0.1 --radius 10 --K 16 --p 0.05 --num_experiments 20

# auditing cifar100 in image classifier
python3 ./experiments/classifier_cifar100.py --data_path <path where cifar-100-python.tar.gz is downloaded and unzipped> --mark_budget 0.1 --radius 10 --K 16 --p 0.05 --num_experiments 20

# auditing tinyimagenet in image classifier
python3 ./experiments/classifier_tinyimagenet.py --data_path <path where tiny-imagenet-200.zip is downloaded and unzipped> --mark_budget 0.1 --radius 10 --K 16 --p 0.05 --num_experiments 20

# auditing cifar10 in visual encoder trained by simclr
python3 ./experiments/simclr_cifar10.py --mark_budget 0.1 --radius 10 --K 64 --p 0.05 --num_experiments 20

# auditing cifar100 in visual encoder trained by simclr
python3 ./experiments/simclr_cifar100.py --mark_budget 0.1 --radius 10 --K 64 --p 0.05 --num_experiments 20

# auditing tinyimagenet in visual encoder trained by simclr
python3 ./experiments/simclr_tinyimagenet.py --mark_budget 0.1 --radius 10 --K 64 --p 0.05 --num_experiments 20

# auditing sst2 in llama 2
python3 ./experiments/llama2_sst2.py --p 0.05 --num_experiments 20

# auditing agnews in llama 2
python3 ./experiments/llama2_agnews.py --p 0.05 --num_experiments 20

# auditing tweet in llama 2
python3 ./experiments/llama2_tweet.py --p 0.05 --num_experiments 20

# auditing flickr30k in clip
python3 ./experiments/clip_flickr.py --radius 10 --p 0.05 --num_experiments 20