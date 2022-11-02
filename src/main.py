import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms


from option import parse_options
from train import train_supcon
from test import test_supcon
from utils.tools import parse_result
from loader import set_test_loader, set_train_loader


if __name__ == "__main__":

    options = parse_options()

    exp_id = options.exp_id

    SEED = 27407 + exp_id

    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    invariant_aug = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(180, expand=False),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )
    augmentations = {
        "to_tensor": to_tensor,
        "invariant_aug": invariant_aug,
    }

    batch_size = min(options.batch_size, options.set_size)

    if options.dataset == "mixedwm38":
        train_loader, valid_loader = set_train_loader(
            options, "./dataset/MixedWM38/training.csv", batch_size, options.set_size
        )
        test_loader = set_test_loader(
            options, "./dataset/MixedWM38/validation.csv", batch_size
        )
        label_counts = None

    elif options.dataset == "wm811k":
        train_loader, valid_loader, label_counts = set_train_loader(
            options,
            "./dataset/WM811k/labeled_training.csv",
            batch_size,
            options.set_size,
        )
        test_loader = set_test_loader(
            options, "./dataset/WM811k/labeled_validation.csv", batch_size
        )

    supcon_train_results = train_supcon(
        options, train_loader, valid_loader, augmentations
    )
    parse_result(options, supcon_train_results, "train")

    supcon_test_results = test_supcon(options, test_loader, label_counts, augmentations)
    parse_result(options, supcon_test_results, "infer")

    torch.cuda.empty_cache()
