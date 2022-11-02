import pandas as pd
import torch.utils.data as data

from utils.dataset import WM811K, MixedWM38
from sklearn.model_selection import train_test_split


def set_train_loader(options, data_path, bat_size, set_size):

    wafers = pd.read_csv(data_path)

    if options.dataset == "mixedwm38":
        trains, valids = train_test_split(
            wafers,
            test_size=int(set_size * 0.2),
            train_size=int(set_size * 0.8),
            shuffle=True,
        )
        training_set = MixedWM38(root="./dataset/MixedWM38", df=trains)
        validation_set = MixedWM38(root="./dataset/MixedWM38/", df=valids)
        label_counts = None
    elif options.dataset == "wm811k":
        trains, valids, _, _ = train_test_split(
            wafers,
            wafers.failureNum,
            test_size=int(set_size * 0.2),
            train_size=int(set_size * 0.8),
            stratify=wafers.failureNum,
        )

        training_set = WM811K(root="./dataset/WM811k", df=trains)
        validation_set = WM811K(root="./dataset/WM811k", df=valids)
        label_counts = trains["failureNum"].value_counts()

    trains.reset_index(drop=True, inplace=True)
    valids.reset_index(drop=True, inplace=True)

    train_loader = data.DataLoader(
        dataset=training_set, shuffle=True, batch_size=bat_size, num_workers=4
    )
    valid_loader = data.DataLoader(
        dataset=validation_set, shuffle=True, batch_size=bat_size, num_workers=4
    )
    if options.dataset == "mixedwm38":
        return train_loader, valid_loader
    else:
        return train_loader, valid_loader, label_counts


def set_test_loader(options, data_path, bat_size):
    wafers = pd.read_csv(data_path)
    if options.dataset == "mixedwm38":
        test_set = MixedWM38(root="./dataset/MixedWM38", df=wafers)
    elif options.dataset == "wm811k":
        test_set = WM811K(root="./dataset/WM811k", df=wafers)

    test_loader = data.DataLoader(
        dataset=test_set, shuffle=True, batch_size=bat_size, num_workers=4
    )

    return test_loader
