
import torch
from torchvision import transforms
from torch import nn
import torchvision

import argparse
import numpy as np
import os
import shutil
from PIL import Image
import warnings

import forest
from src.datasets.folder import default_loader
import random

from PIL import Image
from scipy import stats
import math

import models
from mark import generate_mark_data

# torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)


torch.set_num_threads(1)

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "Metadata Warning, tag [0-9]+ had too many entries", UserWarning)


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


from cswor import BBHG_confseq

def tight_chernoff_bound(tau, N):
    return (math.exp(tau*2/N-1) / ((tau*2/N)**(tau*2/N)))**(N/2)

def find_tau(p, N):
    tau_a = N // 2
    tau_b = N

    while tau_b - tau_a > 1:
        if tight_chernoff_bound((tau_a+tau_b)//2, N) > p:
            tau_a = (tau_a+tau_b)//2
        elif tight_chernoff_bound((tau_a+tau_b)//2, N) < p:
            tau_b = (tau_a+tau_b)//2
        else:
            tau_b = (tau_a+tau_b)//2
            break
    assert tight_chernoff_bound(tau_b, N) <= p
    return tau_b

def detection(transform, augmentation, sample_list, num_classes, classes, args):

    # target model
    print('Loading the target model...')
    ckpt = torch.load(args.published_path + 'target_model.pth')
    target_model = models.get_model(args.net[0], args.dataset, args.pretrained)
    target_model.cuda()
    target_model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()}, strict=True)
    target_model.eval()

    augmentation = transforms.Compose([transforms.RandomResizedCrop(64,(0.8, 1.0))])

    loader = default_loader
    member_img_list = {j[0]: transform(loader(args.published_path + j[1] + '/' + j[0])).unsqueeze(0).numpy() for j in sample_list}
    nonmember_img_list = {j[0]: transform(loader(args.unpublished_path + j[1] + '/' + j[0])).unsqueeze(0).numpy() for j in sample_list}

    sequences1_full = []
    sequences2_full = []
    sequences1_label = []
    sequences2_label = []
    cost_full_me = len(sample_list)
    cost_full_en = len(sample_list)
    cost_label_me = len(sample_list)
    cost_label_en = len(sample_list)
    detected_full_me = False
    detected_full_en = False
    detected_label_me = False
    detected_label_en = False
    acc_full_me = 0
    acc_full_en = 0
    acc_label_me = 0
    acc_label_en = 0
    alpha1 = args.p / 2
    alpha2 = args.p / 2
    tau =  find_tau(alpha2, len(sample_list))
    for j in range(len(sample_list)):

        test_class = sample_list[j][1]

        output1_full = torch.zeros(1, num_classes).cuda(non_blocking=True)
        output2_full = torch.zeros(1, num_classes).cuda(non_blocking=True)
        output1_label = torch.zeros(1, num_classes).cuda(non_blocking=True)
        output2_label = torch.zeros(1, num_classes).cuda(non_blocking=True)
        for _ in range(args.K):
            with torch.no_grad():
                img1 = augmentation(torch.Tensor(member_img_list[sample_list[j][0]]).cuda(non_blocking=True))
                img2 = augmentation(torch.Tensor(nonmember_img_list[sample_list[j][0]]).cuda(non_blocking=True))

                logits1 = target_model(img1)
                logits2 = target_model(img2)

                # output label
                prediction1 = torch.argmax(logits1)
                prediction2 = torch.argmax(logits2)
            
            # accumulate the output confidence score vector
            output1_full += nn.Softmax(dim=1)(logits1)
            output2_full += nn.Softmax(dim=1)(logits2)

            # accumulate the prediction
            output1_label[0][prediction1] += 1
            output2_label[0][prediction2] += 1

        # average full vectors over several augmentation
        output1_full /= args.K   
        output2_full /= args.K

        output1_label /= args.K   
        output2_label /= args.K

        # smooth the vector
        for i in range(num_classes):
            if output1_label[0][i] == 0:
                output1_label[0][i] = 1e-5
            if output2_label[0][i] == 0:
                output2_label[0][i] = 1e-5
        output1_label /= torch.sum(output1_label)
        output2_label /= torch.sum(output2_label)

        # ==================== confidence scores =======================
        pro1 = output1_full.detach().cpu().numpy()[0]
        pro2 = output2_full.detach().cpu().numpy()[0]
        # modified entropy
        score1_full_me = (1-pro1[classes.index(test_class)]) * np.log(pro1[classes.index(test_class)])
        score2_full_me = (1-pro2[classes.index(test_class)]) * np.log(pro2[classes.index(test_class)])
        for i in range(len(classes)):
            if classes[i] != test_class:
                score1_full_me += pro1[i] * np.log(1 - pro1[i])
                score2_full_me += pro2[i] * np.log(1 - pro2[i])

        # entropy
        score1_full_en = -stats.entropy(pro1, base=2)
        score2_full_en = -stats.entropy(pro2, base=2)

        # ==================== label only ============================
        pro1 = output1_label.detach().cpu().numpy()[0]
        pro2 = output2_label.detach().cpu().numpy()[0]
        # modified entropy
        score1_label_me = (1-pro1[classes.index(test_class)]) * np.log(pro1[classes.index(test_class)])
        score2_label_me = (1-pro2[classes.index(test_class)]) * np.log(pro2[classes.index(test_class)])
        for i in range(len(classes)):
            if classes[i] != test_class:
                score1_label_me += pro1[i] * np.log(1 - pro1[i])
                score2_label_me += pro2[i] * np.log(1 - pro2[i])

        # entropy
        score1_label_en = -stats.entropy(pro1, base=2)
        score2_label_en = -stats.entropy(pro2, base=2)
        
        if not detected_full_me:
            if score1_full_me > score2_full_me:
                acc_full_me += 1
                success = 1
            elif score1_full_me == score2_full_me:  # if equal, toss a coin
                if random.sample([True, False], k=1)[0]:
                    acc_full_me += 1
                    success = 1
                else:
                    success = 0
            else:
                success = 0
            sequences1_full.append(success)
            y1, y2 = BBHG_confseq(sequences1_full, len(sample_list), BB_alpha=1, BB_beta=1, alpha=alpha1)
            assert len(y1) == len(sequences1_full)
            if y1[-1] >= tau:
                cost_full_me = len(sequences1_full)
                detected_full_me = True
        
        if not detected_full_en:
            if score1_full_en > score2_full_en:
                acc_full_en += 1
                success = 1
            elif score1_full_en == score2_full_en:  # if equal, toss a coin
                if random.sample([True, False], k=1)[0]:
                    acc_full_en += 1
                    success = 1
                else:
                    success = 0
            else:
                success = 0
            sequences2_full.append(success)
            y1, y2 = BBHG_confseq(sequences2_full, len(sample_list), BB_alpha=1, BB_beta=1, alpha=alpha1)
            assert len(y1) == len(sequences2_full)
            if y1[-1] >= tau:
                cost_full_en = len(sequences2_full)
                detected_full_en = True

        if not detected_label_me:
            if score1_label_me > score2_label_me:
                acc_label_me += 1
                success = 1
            elif score1_label_me == score2_label_me:  # if equal, toss a coin
                if random.sample([True, False], k=1)[0]:
                    acc_label_me += 1
                    success = 1
                else:
                    success = 0
            else:
                success = 0
            sequences1_label.append(success)
            y1, y2 = BBHG_confseq(sequences1_label, len(sample_list), BB_alpha=1, BB_beta=1, alpha=alpha1)
            assert len(y1) == len(sequences1_label)
            if y1[-1] >= tau:
                cost_label_me = len(sequences1_label)
                detected_label_me = True
        
        if not detected_label_en:
            if score1_label_en > score2_label_en:
                acc_label_en += 1
                success = 1
            elif score1_label_en == score2_label_en:  # if equal, toss a coin
                if random.sample([True, False], k=1)[0]:
                    acc_label_en += 1
                    success = 1
                else:
                    success = 0
            else:
                success = 0
            sequences2_label.append(success)
            y1, y2 = BBHG_confseq(sequences2_label, len(sample_list), BB_alpha=1, BB_beta=1, alpha=alpha1)
            assert len(y1) == len(sequences2_label)
            if y1[-1] >= tau:
                cost_label_en = len(sequences2_label)
                detected_label_en = True
    
        if detected_label_en and detected_label_me and detected_full_en and detected_full_me:
            break

    print('==>con1 | cost: {} | membership acc: {}'.format(cost_full_me, acc_full_me / cost_full_me))
    if detected_full_me:
        detected_full_me = 1
    else:
        detected_full_me = 0

    print('==>con2 | cost: {} | membership acc: {}'.format(cost_full_en, acc_full_en / cost_full_en))
    if detected_full_en:
        detected_full_en = 1
    else:
        detected_full_en = 0

    print('==>con3 | cost: {} | membership acc: {}'.format(cost_label_me, acc_label_me / cost_label_me))
    if detected_label_me:
        detected_label_me = 1
    else:
        detected_label_me = 0

    print('==>con4 | cost: {} | membership acc: {}'.format(cost_label_en, acc_label_en / cost_label_en))
    if detected_label_en:
        detected_label_en = 1
    else:
        detected_label_en = 0
    
    return cost_full_me, detected_full_me, cost_full_en, detected_full_en, cost_label_me, detected_label_me, cost_label_en, detected_label_en
    

def prepare_data_loader(path, data, model, data_mean, data_std, args):

    # replace the trainset with the target one
    trainset = torchvision.datasets.ImageFolder(root=path, transform=transforms.ToTensor())

    trainset.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(data_mean, data_std)])
    data.trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(model.defs.batch_size, len(trainset)),
                                                    shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    print('OK')


def generate_twins_data(args):
    listing = os.listdir(args.img_path)
    for i in listing:
        if os.path.isfile(args.img_path + i):
            listing.remove(i)

        if os.path.exists(args.published_path + i):
            shutil.rmtree(args.published_path + i)
        if not os.path.exists(args.published_path + i):
            os.mkdir(args.published_path + i)
        if os.path.exists(args.unpublished_path + i):
            shutil.rmtree(args.unpublished_path + i)
        if not os.path.exists(args.unpublished_path + i):
            os.mkdir(args.unpublished_path + i)

    for i in listing:
        file_list1 = os.listdir(args.img_path + i)
        file_list2 = random.sample(file_list1, int(len(file_list1)*args.mark_budget))
        args.marked_file[i] = file_list2

        for j in range(len(file_list1)):
            image = Image.open(args.img_path + i + '/' + file_list1[j])

            if file_list1[j] in file_list2:
                image1, image2 = generate_mark_data(image, args)
                if random.choice([True,False]):
                    image1.save(args.published_path + '/' + i + '/' + file_list1[j])
                    image2.save(args.unpublished_path + '/' + i + '/' + file_list1[j])
                else:
                    image2.save(args.published_path + '/' + i + '/' + file_list1[j])
                    image1.save(args.unpublished_path + '/' + i + '/' + file_list1[j])
            else:
                image.save(args.published_path + '/' + i + '/' + file_list1[j])

    print('finished generating marked images.')


def get_parser():
    """
    We build up the code using the framework from https://github.com/JonasGeiping/data-poisoning. Some below hyperparameters are unused in our project.
    """
    parser = argparse.ArgumentParser(description='Data auditing')

    ###########################################################################
    # Central:
    parser.add_argument('--net', default='ResNet18', type=lambda s: [str(item) for item in s.split(',')])
    parser.add_argument('--dataset', default='TinyImageNet', type=str, choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'ImageNet1k', 'MNIST', 'TinyImageNet'])
    parser.add_argument('--recipe', default='gradient-matching', type=str, choices=['gradient-matching', 'gradient-matching-private',
                                                                                    'watermarking', 'poison-frogs', 'metapoison', 'bullseye'])
    parser.add_argument('--threatmodel', default='single-class', type=str, choices=['single-class', 'third-party', 'random-subset'])

    # Reproducibility management:
    parser.add_argument('--poisonkey', default=None, type=str, help='Initialize poison setup with this key.')  # Also takes a triplet 0-3-1
    parser.add_argument('--modelkey', default=None, type=int, help='Initialize the model with this key.')
    parser.add_argument('--deterministic', action='store_true', help='Disable CUDNN non-determinism.')

    parser.add_argument('--eps', default=16, type=float)
    parser.add_argument('--budget', default=0.01, type=float, help='Fraction of training data that is poisoned')
    parser.add_argument('--targets', default=1, type=int, help='Number of targets')

    # Files and folders
    parser.add_argument('--name', default='', type=str, help='Name tag for the result table and possibly for export folders.')
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--img_path', type=str, default='./experiments/data/tinyimagenet/train/')
    parser.add_argument('--published_path', type=str, default='./experiments/data/tinyimagenet/')
    parser.add_argument('--unpublished_path', type=str, default='./experiments/data/tinyimagenet/')
    ###########################################################################

    parser.add_argument('--attackoptim', default='signAdam', type=str)
    parser.add_argument('--attackiter', default=250, type=int)
    parser.add_argument('--init', default='randn', type=str)  # randn / rand
    parser.add_argument('--tau', default=0.1, type=float)
    parser.add_argument('--scheduling', action='store_false', help='Disable step size decay.')
    parser.add_argument('--target_criterion', default='cross-entropy', type=str, help='Loss criterion for target loss')
    parser.add_argument('--restarts', default=8, type=int, help='How often to restart the attack.')

    parser.add_argument('--pbatch', default=512, type=int, help='Poison batch size during optimization')
    parser.add_argument('--pshuffle', action='store_true', help='Shuffle poison batch during optimization')
    parser.add_argument('--paugment', action='store_false', help='Do not augment poison batch during optimization')
    parser.add_argument('--data_aug', type=str, default='default', help='Mode of diff. data augmentation.')

    parser.add_argument('--full_data', action='store_true', help='Use full train data (instead of just the poison images)')
    parser.add_argument('--adversarial', default=0, type=float, help='Adversarial PGD for poisoning.')
    parser.add_argument('--ensemble', default=1, type=int, help='Ensemble of networks to brew the poison on')
    parser.add_argument('--stagger', action='store_true', help='Stagger the network ensemble if it exists')
    parser.add_argument('--step', action='store_true', help='Optimize the model for one epoch.')
    parser.add_argument('--max_epoch', default=None, type=int, help='Train only up to this epoch before poisoning.')

    # Use only a subset of the dataset:
    parser.add_argument('--ablation', default=1.0, type=float, help='What percent of data (including poisons) to use for validation')

    # Gradient Matching - Specific Options
    parser.add_argument('--loss', default='similarity', type=str)  # similarity is stronger in  difficult situations

    # These are additional regularization terms for gradient matching. We do not use them, but it is possible
    # that scenarios exist in which additional regularization of the poisoned data is useful.
    parser.add_argument('--centreg', default=0, type=float)
    parser.add_argument('--normreg', default=0, type=float)
    parser.add_argument('--repel', default=0, type=float)

    # Specific Options for a metalearning recipe
    parser.add_argument('--nadapt', default=2, type=int, help='Meta unrolling steps')
    parser.add_argument('--clean_grad', action='store_true', help='Compute the first-order poison gradient.')

    # Validation behavior
    parser.add_argument('--vruns', default=1, type=int, help='How often to re-initialize and check target after retraining')
    parser.add_argument('--vnet', default=None, type=lambda s: [str(item) for item in s.split(',')], help='Evaluate poison on this victim model. Defaults to --net')
    parser.add_argument('--retrain_from_init', action='store_true', help='Additionally evaluate by retraining on the same model initialization.')

    # Optimization setup
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained models from torchvision, if possible [only valid for ImageNet].')
    parser.add_argument('--optimization', default='conservative', type=str, help='Optimization Strategy')
    # Strategy overrides:
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--noaugment', action='store_true', help='Do not use data augmentation during training.')
    parser.add_argument('--gradient_noise', default=None, type=float, help='Add custom gradient noise during training.')
    parser.add_argument('--gradient_clip', default=None, type=float, help='Add custom gradient clip during training.')

    # Optionally, datasets can be stored as LMDB or within RAM:
    parser.add_argument('--lmdb_path', default=None, type=str)
    parser.add_argument('--cache_dataset', action='store_true', help='Cache the entire thing :>')

    # These options allow for testing against the toxicity benchmark found at
    # https://github.com/aks2203/poisoning-benchmark
    parser.add_argument('--benchmark', default='', type=str, help='Path to benchmarking setup (pickle file)')
    parser.add_argument('--benchmark_idx', default=0, type=int, help='Index of benchmark test')

    # Debugging:
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--save', default='full', help='Export poisons into a given format. Options are full/limited/automl/numpy.')

    # Distributed Computations
    parser.add_argument("--local_rank", default=None, type=int, help='Distributed rank. This is an INTERNAL ARGUMENT! '
                                                                     'Only the launch utility should set this argument!')
    
    # parameters of marking algorithm:
    parser.add_argument("--mepochs", type=int, default=90)
    parser.add_argument("--lambda_ft_l2", type=float, default=0.01)
    parser.add_argument("--lambda_l2_img", type=float, default=0.0005)
    parser.add_argument("--moptimizer", type=str, default="sgd,lr=1.0")

    # main parameters
    parser.add_argument('--mark_budget', default=0.1, type=float, help='ratio of marked data or percentage of training data contributed from a data owner')
    parser.add_argument("--radius", type=int, default=10, help='epsilon: utility bound')
    parser.add_argument("--K", type=int, default=16, help='K: number of perturbations per sample in detection')
    parser.add_argument("--p", type=float, default=0.05, help='p: upper bound on false-detection rate')
    parser.add_argument("--num_experiments", type=int, default=20, help='number of experiments to run')

    return parser


if __name__ == "__main__":

    # Parse input arguments
    args = get_parser().parse_args()

    setup = forest.utils.system_startup(args)
    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
    data_mean, data_std = data.trainset.data_mean, data.trainset.data_std
    print(data.trainset.classes)

    args.image_mean = data_mean
    args.image_std = data_std
    args.classes = data.trainset.classes
    args.data_transform = data.trainset.transform
    args.data_augmentation = data.augment
    
    '''
    con1: output is a confidence vector and the ground-truth is known
    con2: output is a confidence vector but the ground-truth is unknown
    con3: output is a prediction but the ground-truth if known
    con4: output is a prediction and the ground-truth is unknown
    '''
    outputs = ['con1', 'con2', 'con3', 'con4']  

    results = {}
    for i in outputs:
        results[i] = {'detected': 0, 'cost': 0, 'Q/M': 0, 'test_acc': 0}

    published_path = args.published_path
    unpublished_path = args.unpublished_path

    for exp_index in range(args.num_experiments):
        
        print('=================================================================================')
        print('Running {}-th experiment'.format(exp_index))

        # data
        args.published_path = published_path + 'published({})/'.format(exp_index)
        args.unpublished_path = unpublished_path + 'unpublished({})/'.format(exp_index)

        if os.path.exists(args.published_path):
            shutil.rmtree(args.published_path)
        if os.path.exists(args.unpublished_path):
            shutil.rmtree(args.unpublished_path)
        if not os.path.exists(args.published_path):
            os.mkdir(args.published_path)
        if not os.path.exists(args.unpublished_path):
            os.mkdir(args.unpublished_path)

        # generate twins data and published one uniformly at random
        print('Generate marked data and published one uniformly at random...')
        args.marked_file = {}
        generate_twins_data(args)

        # train target model
        model = forest.Victim(args, setup=setup)
        data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
        data_mean, data_std = data.trainset.data_mean, data.trainset.data_std

        print('Training a {} model on published data...'.format(args.net[0]))
        prepare_data_loader(args.published_path, data, model, data_mean, data_std, args)
        stats_results = model.validate(data, 1)
        print('Saving target model to {}...'.format(args.published_path + 'target_model.pth'))
        torch.save(model.model.state_dict(), args.published_path + 'target_model.pth')
                
        for output in outputs:
            results[output]['test_acc'] += stats_results['valid_accs'][-1] / args.num_experiments

        print('Detect the data use in target model...')
        # random shuffle the pairs of published data and unpublished data
        listing = os.listdir(args.published_path)
        sample_list = []
        for i in listing:
            if os.path.isdir(args.published_path + i):
                # file_list = os.listdir(args.published_path + i)
                samples = args.marked_file[i]
                # samples = file_list
                for j in samples:
                    sample_list.append([j,i])
        random.shuffle(sample_list)

        # membership inference
        cost1, detected1, cost2, detected2, cost3, detected3, cost4, detected4 = detection(data.trainset.transform, data.augment, sample_list, len(data.trainset.classes), data.trainset.classes, args)
        results['con1']['cost'] += cost1 * 2 * args.K / args.num_experiments
        results['con1']['Q/M'] += cost1 / 100000 / args.num_experiments
        results['con1']['detected'] += detected1 / args.num_experiments
        results['con2']['cost'] += cost2 * 2 * args.K / args.num_experiments
        results['con2']['Q/M'] += cost2 / 100000 / args.num_experiments
        results['con2']['detected'] += detected2 / args.num_experiments
        results['con3']['cost'] += cost3 * 2 * args.K / args.num_experiments
        results['con3']['Q/M'] += cost3 / 100000 / args.num_experiments
        results['con3']['detected'] += detected3 / args.num_experiments
        results['con4']['cost'] += cost4 * 2 * args.K / args.num_experiments
        results['con4']['Q/M'] += cost4 / 100000 / args.num_experiments
        results['con4']['detected'] += detected4 / args.num_experiments

    print('print out results averaged over {} experiments...'.format(args.num_experiments))
    print(results)

    print('-------------Job finished.-------------------------')
