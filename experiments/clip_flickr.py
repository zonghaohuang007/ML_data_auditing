
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torch.nn as nn
import torch.nn.functional as F
import torch

import clip
import os
import argparse
import random
import math
import numpy as np

from mark import generate_mark_data

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def detection(target_model, published, unpublished, preprocess, args):

    sequences = []
    cost = len(published)
    detected = False
    acc = 0
    alpha1 = args.p / 2
    alpha2 = args.p / 2
    tau =  find_tau(alpha2, len(published))
    shuffled_list = list(range(len(published)))
    random.shuffle(shuffled_list)

    transform = preprocess

    for j in shuffled_list:
        
        with torch.no_grad():
            img1 = transform(published[j]['image'].convert("RGB")).unsqueeze(0).cuda(non_blocking=True)
            img2 = transform(unpublished[j]['image'].convert("RGB")).unsqueeze(0).cuda(non_blocking=True)
            
            fv1 = target_model.encode_image(img1).cpu().numpy()[0]
            fv2 = target_model.encode_image(img2).cpu().numpy()[0]

            txt1 = clip.tokenize([published[j]['caption'][1]]).cuda(non_blocking=True)
            txt2 = clip.tokenize([unpublished[j]['caption'][1]]).cuda(non_blocking=True)

            txt_fv1 = target_model.encode_text(txt1).cpu().numpy()[0]
            txt_fv2 = target_model.encode_text(txt2).cpu().numpy()[0]
        
        # average full vectors over several augmentation
        score1 = np.dot(fv1, txt_fv1)/(np.linalg.norm(fv1)*np.linalg.norm(txt_fv1))
        score2 = np.dot(fv2, txt_fv2)/(np.linalg.norm(fv2)*np.linalg.norm(txt_fv2))

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
            y1, y2 = BBHG_confseq(sequences, len(published), BB_alpha=1, BB_beta=1, alpha=alpha1)
            assert len(y1) == len(sequences)
            if y1[-1] >= tau:
                cost = len(sequences)
                detected = True

        if detected:
            break

    print('==>cost: {} | membership acc: {}'.format(len(sequences), acc / len(sequences)))
    if detected:
        detected = 1
    else:
        detected = 0
    
    return cost, detected


def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


class Flickr30kDataset(Dataset):
    def __init__(self, dataset, preprocess):
        # self.dataset = load_dataset("nlphuji/flickr30k", cache_dir="./huggingface_data")
        self.dataset = dataset
        self.transform = preprocess
        # self.cap_per_image = 2

    def __len__(self):
        # return self.dataset.num_rows["test"] * self.cap_per_image
        return len(self.dataset)

    def __getitem__(self, idx):
        # original_idx = idx // self.cap_per_image
        # image_path = self.dataset[idx]["image_path"]
        image = self.dataset[idx]["image"].convert("RGB")
        image = self.transform(image)

        # You might need to adjust the labels based on your task
        caption = clip.tokenize([self.dataset[idx]["caption"][1]])[0]

        return {"image": image, "caption": caption}


def metrics(similarity: torch.Tensor):
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc


def generate_twins_data(dataset, args):

    published = []
    unpublished = []
    train = []
    sampled_idx = random.sample(list(range(len(dataset))), int(len(dataset)*0.1))  # assume 10% is contributed from a data owner
    for i in range(len(dataset)):
        
        image = dataset[i]['image']
        image = image.resize((224, 224))
        if i in sampled_idx:
            image1, image2 = generate_mark_data(image, args)
            published_ = {}
            unpublished_ = {}
            if random.choice([True,False]):
                published_['image'] = image1
                unpublished_['image'] = image2
                train.append({'image': image1, 'caption': dataset[i]['caption']})
            else:
                published_['image'] = image2
                unpublished_['image'] = image1
                train.append({'image': image2, 'caption': dataset[i]['caption']})
            published_['caption'] = dataset[i]['caption']
            unpublished_['caption'] = dataset[i]['caption']
            published.append(published_)
            unpublished.append(unpublished_)
        else:
            train.append({'image': image, 'caption': dataset[i]['caption']})
    
    assert len(train) == len(dataset)

    assert len(published) == len(unpublished)

    return train, published, unpublished


def get_parser():
    """Construct the central argument parser, filled with useful defaults.

    The first block is essential to test poisoning in different scenarios.
    The options following afterwards change the algorithm in various ways and are set to reasonable defaults.
    """
    parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')

    # mark:
    parser.add_argument("--mepochs", type=int, default=90)
    parser.add_argument("--lambda_ft_l2", type=float, default=0.01)
    parser.add_argument("--lambda_l2_img", type=float, default=0.0005)
    parser.add_argument("--moptimizer", type=str, default="sgd,lr=1.0")

    # CLIP model
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--transformer_embed_dim", type=int, default=768)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)

    # Distributed Computations
    parser.add_argument("--local_rank", default=None, type=int, help='Distributed rank. This is an INTERNAL ARGUMENT! '
                                                                     'Only the launch utility should set this argument!')

    parser.add_argument("--radius", type=int, default=10, help='epsilon: utility bound')
    parser.add_argument("--p", type=float, default=0.05, help='p: upper bound on false-detection rate')
    parser.add_argument("--num_experiments", type=int, default=20, help='number of experiments to run')

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()

    num_epochs = 3
    results = {}
    for i in range(num_epochs+1):
        results[str(i)] = {'cost':0, 'detected':0, 'Q/M': 0, 'acc': 0}

    for exp_index in range(args.num_experiments):
        
        print('=================================================================================')
        print('Running {}-th experiment'.format(exp_index))

        dataset = load_dataset("nlphuji/flickr30k")['test']
        testset = dataset.select([i for i in list(range(25000,31014))])

        trainset, publishedset, unpublishedset = generate_twins_data(dataset, args)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

        clip_dataset = Flickr30kDataset(trainset, preprocess)
        testset = Flickr30kDataset(testset, preprocess)

        # Create the DataLoader
        clip_dataloader = DataLoader(
            clip_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

        test_dataloader = DataLoader(
            testset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

        # initial test
        cost, detected = detection(model, publishedset, unpublishedset, preprocess, args)
        results['0']['cost'] += cost / args.num_experiments
        results['0']['Q/M'] += cost / 25000 / args.num_experiments
        results['0']['detected'] += detected / args.num_experiments
        model.eval()
        test_loss = 0
        test_sim = 0
        test_img_acc = 0
        test_cap_acc = 0
        for batch in test_dataloader:
            with torch.no_grad():
                image = batch["image"].to(device)
                text = batch["caption"].to(device)
                # images, text = batch
                image_embed, caption_embed = model(image, text)

                image_embed_ = model.encode_image(image)
                caption_embed_ = model.encode_text(text)

                image_embed_ = image_embed_ / torch.norm(image_embed_, dim=-1, keepdim=True)
                caption_embed_ = caption_embed_ / torch.norm(caption_embed_, dim=-1, keepdim=True)

                similarity = image_embed_ @ caption_embed_.T

                test_sim += torch.mean(torch.diagonal(similarity)).item()

                ground_truth = torch.arange(len(image),dtype=torch.long,device=device)
                loss = (nn.CrossEntropyLoss()(image_embed, ground_truth) + nn.CrossEntropyLoss()(caption_embed,ground_truth))/2

                img_acc, cap_acc = metrics(similarity)

                test_loss += loss.item()
                test_img_acc += img_acc.item()
                test_cap_acc += cap_acc.item()

        # Print training statistics
        print(f"Batch Loss: {test_loss/len(test_dataloader)}, Sim: {test_sim/len(test_dataloader)}, Image acc: {test_img_acc/len(test_dataloader)}, Cap acc: {test_cap_acc/len(test_dataloader)}")
        results['0']['acc'] += (test_img_acc/len(test_dataloader) + test_cap_acc/len(test_dataloader))/2 / args.num_experiments

        # Define optimizer
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()}
            ],
            lr=1e-5,
        )

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_sim = 0
            train_img_acc = 0
            train_cap_acc = 0
            for batch in clip_dataloader:
                image = batch["image"].to(device)
                text = batch["caption"].to(device)
                # images, text = batch
                image_embed, caption_embed = model(image, text)

                ground_truth = torch.arange(len(image),dtype=torch.long,device=device)
                loss = (nn.CrossEntropyLoss()(image_embed, ground_truth) + nn.CrossEntropyLoss()(caption_embed,ground_truth))/2

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                # optimizer.step()

                train_loss += loss.item()

                if device == "cpu":
                    optimizer.step()
                else : 
                    convert_models_to_fp32(model)
                    optimizer.step()
                    clip.model.convert_weights(model)

            # Print training statistics
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {train_loss/len(clip_dataloader)}")

            model.eval()
            test_loss = 0
            test_sim = 0
            test_img_acc = 0
            test_cap_acc = 0
            for batch in test_dataloader:
                with torch.no_grad():
                    image = batch["image"].to(device)
                    text = batch["caption"].to(device)
                    # images, text = batch
                    image_embed, caption_embed = model(image, text)

                    image_embed_ = model.encode_image(image)
                    caption_embed_ = model.encode_text(text)

                    image_embed_ = image_embed_ / torch.norm(image_embed_, dim=-1, keepdim=True)
                    caption_embed_ = caption_embed_ / torch.norm(caption_embed_, dim=-1, keepdim=True)

                    similarity = image_embed_ @ caption_embed_.T

                    test_sim += torch.mean(torch.diagonal(similarity)).item()

                    ground_truth = torch.arange(len(image),dtype=torch.long,device=device)
                    loss = (nn.CrossEntropyLoss()(image_embed, ground_truth) + nn.CrossEntropyLoss()(caption_embed,ground_truth))/2

                    img_acc, cap_acc = metrics(similarity)

                    test_loss += loss.item()
                    test_img_acc += img_acc.item()
                    test_cap_acc += cap_acc.item()

            # Print training statistics
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {test_loss/len(test_dataloader)}, Sim: {test_sim/len(test_dataloader)}, Image acc: {test_img_acc/len(test_dataloader)}, Cap acc: {test_cap_acc/len(test_dataloader)}")

            cost, detected = detection(model, publishedset, unpublishedset, preprocess, args)
            results[str(epoch+1)]['cost'] += cost / args.num_experiments
            results[str(epoch+1)]['Q/M'] += cost / 25000 / args.num_experiments
            results[str(epoch+1)]['detected'] += detected / args.num_experiments
            results[str(epoch+1)]['acc'] += (test_img_acc/len(test_dataloader) + test_cap_acc/len(test_dataloader))/2 / args.num_experiments

    print('print out results averaged over {} experiments...'.format(args.num_experiments))
    print(results)
