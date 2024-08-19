# Please note that it will cost weeks or even months to finish all the experiments. It is better to run them in parallel to save time.

# To reproduce results in table 2: overall performance of our proposed method applied in image classifiers on different image benchmarks
## auditing cifar10 in image classifier
python3 ./experiments/classifier_cifar10.py --mark_budget 0.1 --radius 10 --K 16 --p 0.05 --num_experiments 20
## auditing cifar100 in image classifier
python3 ./experiments/classifier_cifar100.py --mark_budget 0.1 --radius 10 --K 16 --p 0.05 --num_experiments 20
## auditing tinyimagenet in image classifier
python3 ./experiments/classifier_tinyimagenet.py --data_path './tiny-imagenet-200' --mark_budget 0.1 --radius 10 --K 16 --p 0.05 --num_experiments 20


# To reproduce results in table 6: Results on auditing data in visual encoder trained by SimCLR
## auditing cifar10 in visual encoder trained by simclr
python3 ./experiments/simclr_cifar10.py --mark_budget 0.1 --radius 10 --K 64 --p 0.05 --num_experiments 20
## auditing cifar100 in visual encoder trained by simclr
python3 ./experiments/simclr_cifar100.py --mark_budget 0.1 --radius 10 --K 64 --p 0.05 --num_experiments 20
## auditing tinyimagenet in visual encoder trained by simclr
python3 ./experiments/simclr_tinyimagenet.py --mark_budget 0.1 --radius 10 --K 64 --p 0.05 --num_experiments 20


# To reproduce results in table 7: Overall performance of our proposed method on Llama 2 fine-tuned on marked text datasets
## auditing sst2 in llama 2
python3 ./experiments/llama2_sst2.py --mark_budget 0.1 --p 0.05 --num_experiments 20
## auditing agnews in llama 2
python3 ./experiments/llama2_agnews.py --mark_budget 0.1 --p 0.05 --num_experiments 20
## auditing tweet in llama 2
python3 ./experiments/llama2_tweet.py --mark_budget 0.1 --p 0.05 --num_experiments 20


# To reproduce results in table 8:  Overall performance of our proposed method on CLIP fine-tuned on marked Flickr30k
## auditing flickr30k in clip
python3 ./experiments/clip_flickr.py --mark_budget 0.1 --radius 10 --p 0.05 --num_experiments 20