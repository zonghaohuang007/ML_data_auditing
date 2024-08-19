
import logging

import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import transforms
import torchvision

import forest
from tqdm import tqdm
import argparse
import os
import random
import shutil
import math
from sklearn.metrics.pairwise import cosine_similarity
from src.datasets.folder import default_loader
from mark import generate_mark_data


logger = logging.getLogger(__name__)


class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.enc = base_encoder(pretrained=False)  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.enc.fc.in_features

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.enc.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()  # remove final fully connected layer.

        # Add MLP projection.
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, projection_dim))

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CIFAR10Pair(CIFAR10):
    """Generate mini-batche pairs on CIFAR10 training set."""
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # stack a positive pair


class C10DataGen(Dataset):
    def __init__(self,transforms,img):
        self.img = img
        self.transforms = transforms

    def __len__(self):
        return len(self.img)

    def __getitem__(self,idx):
        
        x = self.img[idx][0]
        
        x1 = self.transforms(x)
        x2 = self.transforms(x)
        
        return torch.stack([x1, x2]), self.img[idx][1]


def nt_xent(x, t=0.5):
    x = F.normalize(x, dim=1)
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


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


class LinModel(nn.Module):
    """Linear wrapper of encoder."""
    def __init__(self, encoder: nn.Module, feature_dim: int, n_classes: int):
        super().__init__()
        self.enc = encoder
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(self.feature_dim, self.n_classes)

    def forward(self, x):
        return self.lin(self.enc(x))


def run_epoch(model, dataloader, epoch, optimizer=None, scheduler=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    loader_bar = tqdm(dataloader)
    for x, y in loader_bar:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        acc = (logits.argmax(dim=1) == y).float().mean()
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))
        if optimizer:
            loader_bar.set_description("Train epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))
        else:
            loader_bar.set_description("Test epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))

    return loss_meter.avg, acc_meter.avg


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def finetune(args):

    train_transform = transforms.Compose([transforms.RandomResizedCrop(64),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=args.image_mean, std=args.image_std)])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=args.image_mean, std=args.image_std)])

    train_set = torchvision.datasets.ImageFolder(root='./experiments/data/tinyimagenet/train/', transform=train_transform)
    sample = random.sample(range(len(train_set)), int(0.1 * len(train_set)))
    train_set = forest.data.datasets.Subset(train_set, sample)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )

    test_set = torchvision.datasets.ImageFolder(root='./experiments/data/tinyimagenet/test/', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )

    # Prepare model
    base_encoder = eval(args.backbone)
    pre_model = SimCLR(base_encoder, projection_dim=args.projection_dim).cuda()
    pre_model.load_state_dict(torch.load(args.published_path + 'simclr_model_resnet18(1000).pth'))
    model = LinModel(pre_model.enc, feature_dim=pre_model.feature_dim, n_classes=len(train_set.targets))
    model = model.cuda()

    # Fix encoder
    model.enc.requires_grad = False
    parameters = [param for param in model.parameters() if param.requires_grad is True]  # trainable parameters.

    optimizer = torch.optim.SGD(
        parameters,
        0.2,   # lr = 0.1 * batch_size / 256, see section B.6 and B.7 of SimCLR paper.
        momentum=args.momentum,
        weight_decay=0.,
        nesterov=True)

    # cosine annealing lr
    # scheduler = LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
    #         step,
    #         args.finetune_epochs * len(train_loader),
    #         args.learning_rate,  # lr_lambda computes multiplicative factor
    #         1e-3))
    scheduler = None

    optimal_loss, optimal_acc = 1e5, 0.
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, epoch, optimizer, scheduler)
        test_loss, test_acc = run_epoch(model, test_loader, epoch)

        if train_loss < optimal_loss:
            optimal_loss = train_loss
            optimal_acc = test_acc

    return optimal_acc


def get_parser():
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--img_path', type=str, default='./experiments/data/tinyimagenet/train/')
    parser.add_argument('--published_path', type=str, default='./experiments/data/tinyimagenet/')
    parser.add_argument('--unpublished_path', type=str, default='./experiments/data/tinyimagenet/')
    parser.add_argument("--mepochs", type=int, default=90)
    parser.add_argument("--lambda_ft_l2", type=float, default=0.01)
    parser.add_argument("--lambda_l2_img", type=float, default=0.0005)
    parser.add_argument("--moptimizer", type=str, default="sgd,lr=1.0")

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--backbone", type=str, default='resnet18')
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=512)

    parser.add_argument('--finetune_epochs', type=int, default=90)
    parser.add_argument('--learning_rate', type=float, default=0.6)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--mark_budget', default=0.1, type=float, help='ratio of marked data or percentage of training data contributed from a data owner')
    parser.add_argument("--radius", type=int, default=10, help='epsilon: utility bound')
    parser.add_argument("--K", type=int, default=64, help='K: number of perturbations per sample in detection')
    parser.add_argument("--p", type=float, default=0.05, help='p: upper bound on false-detection rate')
    parser.add_argument("--num_experiments", type=int, default=20, help='number of experiments to run')


    return parser


def train(args):

    args.train_path = args.published_path

    assert torch.cuda.is_available()
    cudnn.benchmark = True

    train_transform = transforms.Compose([transforms.RandomResizedCrop(64),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          get_color_distortion(s=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.4789886474609375, 0.4457630515098572, 0.3944724500179291], std=[0.27698642015457153, 0.2690644860267639, 0.2820819020271301])
                                          ])

    train_dataset = torchvision.datasets.ImageFolder(root=args.train_path, transform=None)
    train_data = C10DataGen(train_transform, train_dataset)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True
    )

    # Prepare model
    assert args.backbone in ['resnet18', 'resnet34']
    base_encoder = eval(args.backbone)
    model = SimCLR(base_encoder, projection_dim=args.projection_dim).cuda()
    logger.info('Base model: {}'.format(args.backbone))
    logger.info('feature dim: {}, projection dim: {}'.format(model.feature_dim, args.projection_dim))

    optimizer = torch.optim.SGD(
        model.parameters(),
        0.3 * args.batch_size / 256,
        momentum=0.9,
        weight_decay=1.0e-6,
        nesterov=True)

    # cosine annealing lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            0.3 * args.batch_size / 256,  # lr_lambda computes multiplicative factor
            1e-3))

    # SimCLR training
    model.train()
    for epoch in range(1, args.epochs + 1):
        loss_meter = AverageMeter("SimCLR_loss")
        train_bar = tqdm(train_loader)
        for x, y in train_bar:
            sizes = x.size()
            x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4]).cuda(non_blocking=True)

            optimizer.zero_grad()
            feature, rep = model(x)
            loss = nt_xent(rep, 0.5)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), x.size(0))
            train_bar.set_description("Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))

        # save checkpoint very log_interval epochs
        if epoch >= 100 and epoch % 100 == 0:
            logger.info("==> Save checkpoint. Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))
            torch.save(model.state_dict(), args.train_path + 'simclr_model_resnet18({}).pth'.format(epoch))


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

def detection(transform, sample_list, args):

    # target model
    ckpt = torch.load(args.published_path + 'simclr_model_resnet18(1000).pth')
    assert args.backbone in ['resnet18', 'resnet34']
    base_encoder = eval(args.backbone)
    target_model = SimCLR(base_encoder, projection_dim=args.projection_dim).cuda()
    target_model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()}, strict=True)
    target_model.eval()

    augmentation = transforms.Compose([transforms.RandomResizedCrop(64,(0.8, 1.0)),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        ])

    loader = default_loader
    member_img_list = {j[0]: transform(loader(args.published_path + j[1] + '/' + j[0])).unsqueeze(0).numpy() for j in sample_list}
    nonmember_img_list = {j[0]: transform(loader(args.unpublished_path + j[1] + '/' + j[0])).unsqueeze(0).numpy() for j in sample_list}

    sequences = []
    cost = len(sample_list)
    detected = False
    acc = 0
    alpha1 = args.p / 2
    alpha2 = args.p / 2
    tau =  find_tau(alpha2, len(sample_list))
    for j in range(len(sample_list)):
        
        fv1_set = []
        fv2_set = []

        for _ in range(args.K):
            with torch.no_grad():
                img1 = augmentation(torch.Tensor(member_img_list[sample_list[j][0]]).cuda(non_blocking=True))
                img2 = augmentation(torch.Tensor(nonmember_img_list[sample_list[j][0]]).cuda(non_blocking=True))

                fv1, _ = target_model(img1)
                fv2, _ = target_model(img2)

            fv1_set.append(fv1.detach().cpu().numpy()[0])
            fv2_set.append(fv2.detach().cpu().numpy()[0])
        
        fv1_set = np.array(fv1_set)
        fv2_set = np.array(fv2_set)

        # average full vectors over several augmentation
        score1 = np.sum(cosine_similarity(fv1_set))
        score2 = np.sum(cosine_similarity(fv2_set))

        if not detected:
            if score1 > score2:
                acc += 1
                success = 1
            elif score1 == score2:  # if equal, toss a coin
                if random.sample([True, False], k=1)[0]:
                    acc += 1
                    success = 1
                else:
                    success = 0
            else:
                success = 0
            sequences.append(success)
            y1, y2 = BBHG_confseq(sequences, len(sample_list), BB_alpha=1, BB_beta=1, alpha=alpha1)
            assert len(y1) == len(sequences)
            if y1[-1] >= tau:
                cost = len(sequences)
                detected = True

        if detected:
            break

    print('==>cost: {} | membership acc: {}'.format(cost, acc / cost))
    if detected:
        detected = 1
    else:
        detected = 0
    
    return cost, detected


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    published_path = args.published_path
    unpublished_path = args.unpublished_path

    results = {'detected': 0, 'cost': 0, 'Q/M': 0, 'test_acc': 0}

    for exp_index in range(args.num_experiments):
        
        print('=================================================================================')
        print('Running {}-th experiment'.format(exp_index))

        # data
        args.published_path = published_path + 'encoder_published({})/'.format(exp_index)
        args.unpublished_path = unpublished_path + 'encoder_unpublished({})/'.format(exp_index)

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

        # train encoder by simclr
        print('Train encoder by SIMCLR')
        train(args)

        # detect the encoder
        args.image_mean = [0.4789886474609375, 0.4457630515098572, 0.3944724500179291]
        args.image_std = [0.27698642015457153, 0.2690644860267639, 0.2820819020271301]

        # train downstream model
        print('Evaluate the trained encoder by measuring the test accuracy of the downstream classifier')
        optimal_acc = finetune(args)
        results['test_acc'] += optimal_acc / args.num_experiments
                
        # random shuffle
        listing = os.listdir(args.published_path)
        sample_list = []
        for i in listing:
            if os.path.isdir(args.published_path + i):
                samples = args.marked_file[i]
                for j in samples:
                    sample_list.append([j,i])
        random.shuffle(sample_list)
            
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=args.image_mean, std=args.image_std)])
            
        # detect data use
        print('Detect data use in the visual encoder')
        cost, detected = detection(transform, sample_list, args)
        results['cost'] += cost * 2 * args.K / args.num_experiments
        results['Q/M'] += cost / 100000 / args.num_experiments
        results['detected'] += detected / args.num_experiments

    print('print out results averaged over {} experiments...'.format(args.num_experiments))
    print(results)
