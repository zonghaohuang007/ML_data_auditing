import torch
from torchvision import transforms
from torch import nn
import numpy as np
from PIL import Image
from src.utils import get_optimizer, repeat_to
from PIL import Image
from torchvision.models import resnet18


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def numpyTranspose(x):
    return np.transpose(x.numpy(), (1, 2, 0))


def numpyPixel(x, image_mean, image_std):
    pixel_image = torch.round(255 * ((x * torch.Tensor(image_std).view(-1, 1, 1)) + torch.Tensor(image_mean).view(-1, 1, 1))).clamp(0, 255)
    return np.transpose(pixel_image.numpy().astype(np.uint8), (1, 2, 0))


def roundPixel(x, image_mean, image_std):
    x_pixel = 255 * ((x * torch.Tensor(image_std).view(-1, 1, 1)) + torch.Tensor(image_mean).view(-1, 1, 1))
    y = torch.round(x_pixel).clamp(0, 255)
    y = ((y / 255.0) - torch.Tensor(image_mean).view(-1, 1, 1)) / torch.Tensor(image_std).view(-1, 1, 1)
    return y


def project_linf(x, y, radius, image_mean, image_std):
    delta = x - y
    delta = 255 * (delta * torch.Tensor(image_std).view(-1, 1, 1))
    delta = torch.clamp(delta, -radius, radius)
    delta = (delta / 255.0) / torch.Tensor(image_std).view(-1, 1, 1)
    return y + delta


def generate_mark_data(image, args):

    model = resnet18(pretrained=True)
    model.cuda()
    model = model.eval()
    model.fc = nn.Sequential()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    data_augmentation = transforms.Compose([transforms.Resize(224)])

    image_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    image_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    # marking
    img_orig = [transform(image).unsqueeze(0)]
    img = [x.clone() for x in img_orig]
    peturbation = [torch.randn_like(x) * args.radius / 255 / torch.mean(image_std) for x in img_orig]

    for k in range(len(peturbation)):
        peturbation[k].requires_grad = True

    moptimizer, schedule = get_optimizer(peturbation, args.moptimizer)
    if schedule is not None:
        schedule = repeat_to(schedule, args.mepochs)

    img_center = torch.cat([x.cuda(non_blocking=True) for x in img_orig], dim=0)
    ft_orig = model(data_augmentation(img_center)).detach()

    for iteration in range(args.mepochs):
        if schedule is not None:
            lr = schedule[iteration]
            for param_group in moptimizer.param_groups:
                param_group['lr'] = lr

        # Differentially augment images
        batch = []
        for x in range(len(img_orig)):
            aug_img = data_augmentation(img_orig[x] + peturbation[x])
            batch.append(aug_img.cuda(non_blocking=True))
        batch = torch.cat(batch, dim=0)

        batch2 = []
        for x in range(len(img_orig)):
            aug_img = data_augmentation(img_orig[x] - peturbation[x])
            batch2.append(aug_img.cuda(non_blocking=True))
        batch2 = torch.cat(batch2, dim=0)

        # Forward augmented images
        ft = model(batch)
        ft2 = model(batch2)

        loss_ft = - torch.norm(ft - ft2)

        loss_ft_l2 = args.lambda_ft_l2 * torch.norm(ft - ft_orig, dim=1).sum()

        loss_norm = 0
        for k in range(len(img_orig)):
            loss_norm += args.lambda_l2_img * (torch.norm(peturbation[k].cuda(non_blocking=True))**2)
        # loss = loss_ft + loss_norm + loss_ft_l2
        loss = loss_ft

        moptimizer.zero_grad()
        loss.backward()
        moptimizer.step()

        logs = {"keyword": "iteration",
                "loss": loss.item(),
                "loss_ft": loss_ft.item(),
                "loss_norm": loss_norm.item(),
                "loss_ft_l2": loss_ft_l2.item()
                }

        for k in range(len(peturbation)):
            peturbation[k].data[0] = project_linf(img_orig[k][0] + peturbation[k].data[0], img_orig[k][0], args.radius, image_mean, image_std) - img_orig[k][0]
            if iteration % 10 == 0:
                peturbation[k].data[0] = roundPixel(img_orig[k][0] + peturbation[k].data[0], image_mean, image_std) - img_orig[k][0]

    print(logs)

    img_new = [numpyPixel(img[0].data[0] + peturbation[0].data[0], image_mean, image_std).astype(np.float32)]
    img_new2 = [numpyPixel(img[0].data[0] - peturbation[0].data[0], image_mean, image_std).astype(np.float32)]

    saved_image = Image.fromarray(img_new[0].astype(np.uint8))
    saved_image2 = Image.fromarray(img_new2[0].astype(np.uint8))

    return saved_image, saved_image2
