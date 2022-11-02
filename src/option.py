import argparse


def parse_options():
    parser = argparse.ArgumentParser("Arguments for training")

    parser.add_argument(
        "--dataset",
        type=str,
        choices={"wm811k", "mixedwm38"},
    )

    parser.add_argument("--batch_size", type=int, help="batch size")

    parser.add_argument(
        "--set_size",
        type=int,
        help="data subset size : how many data instances you will use to train model",
    )

    parser.add_argument("--epochs", type=int, default=500, help="epochs")

    parser.add_argument("--patience", type=int, default=50, help="epochs")

    parser.add_argument(
        "--model_config",
        type=str,
        default="16",
        choices={"11", "13", "16", "19"},
        help="model configuration(depth)\n \
            vgg = [11,13,16,19]\n",
    )

    parser.add_argument(
        "--head",
        type=str,
        default="linear",
        choices={"linear", "mlp"},
        help="Architectu1re of projection head: mlp or linear, default = linear",
    )

    parser.add_argument(
        "--gamma", default=0.1, type=float, help="gamma value for contrastive loss"
    )

    parser.add_argument(
        "--temperature",
        default=0.1,
        type=float,
        help="temperature parameter for contrastive loss",
    )

    parser.add_argument("--exp_id", type=int, help="experiment id")

    opt = parser.parse_args()

    return opt
