import pathlib
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from loss import SupConLoss

from models.vgg import SupConVGG, VGGLinearClassifier
from utils.tools import EarlyStopping, generatePositive, label_hard_acc


# proposed method
def train_supcon(options, train_loader, valid_loader, augmentation: dict):

    model_name = "vgg" + options.model_config

    log_path = pathlib.Path("log")
    pt_path = pathlib.Path("saved")

    logdir = (
        log_path
        / options.dataset
        / str(options.exp_id)
        / (model_name + options.head)
        / str(options.set_size)
        / str(options.gamma)
    )
    pt_dir = (
        pt_path
        / options.dataset
        / str(options.exp_id)
        / (model_name + options.head)
        / str(options.set_size)
        / str(options.gamma)
    )

    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=False)
    if not pt_dir.exists():
        pt_dir.mkdir(parents=True, exist_ok=False)

    writer = SummaryWriter(log_dir=logdir)
    writer.flush()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCHS = options.epochs
    train_loader.dataset.set_transform(generatePositive(augmentation["invariant_aug"]))
    valid_loader.dataset.set_transform(augmentation["to_tensor"])

    if options.dataset == "mixedwm38":
        ce_loss = nn.BCEWithLogitsLoss()
        num_classes = 8
        sup_con_loss = SupConLoss(
            temperature=options.temperature,
            contrast_mode="multi_label",
            base_temperature=options.temperature,
            num_classes=8,
        )

    elif options.dataset == "wm811k":
        ce_loss = nn.CrossEntropyLoss()
        num_classes = 9
        sup_con_loss = SupConLoss(
            temperature=options.temperature,
            contrast_mode="multi_class",
            base_temperature=options.temperature,
            num_classes=9,
        )

    model = SupConVGG(name=model_name, head=options.head, feat_dim=128).to(device)
    classifier = VGGLinearClassifier(
        encoder_name=model_name, num_classes=num_classes
    ).to(device)

    params = list(model.parameters()) + list(classifier.parameters())

    optimizer = optim.Adam(params, lr=1e-5, betas=(0.9, 0.999))

    earlystop = EarlyStopping(patience=options.patience)

    min_val_loss = torch.inf
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    c = options.gamma  # balancing hyperparam between losses

    for epoch in range(EPOCHS):
        # Training
        train_loss = 0
        train_acc = 0
        model.train()
        classifier.train()
        num_step = 0
        for idx, (image, labels) in enumerate(train_loader, start=1):

            optimizer.zero_grad()

            x1 = image[0]
            x2 = image[1]

            images = torch.cat([x1, x2], dim=0)

            images = images.to(device)
            labels = labels.to(device)
            repr, projected = model(images)

            bsz = x1.shape[0]
            h1, h2 = torch.split(projected, [bsz, bsz], dim=0)
            h = torch.cat([h1.unsqueeze(1), h2.unsqueeze(1)], dim=1)

            contrastive_loss = sup_con_loss(features=h, labels=labels)

            logits = classifier(repr)
            label_cat = torch.cat([labels, labels], dim=0)
            classfication_loss = ce_loss(logits, label_cat)

            net_loss = c * contrastive_loss + classfication_loss

            net_loss.backward()

            if options.dataset == "mixedwm38":
                probs = logits.sigmoid()
                predicted = (probs > 0.5).float()
            else:
                probs = logits.softmax(dim=1)
                predicted = probs.argmax(1)

            train_acc += label_hard_acc(predicted, label_cat)
            train_loss += net_loss.item()
            optimizer.step()
            num_step = idx

        train_acc /= num_step
        train_loss /= num_step
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)

        # Validation
        valid_loss = 0
        valid_acc = 0
        model.eval()
        classifier.eval()

        num_step = 0
        with torch.no_grad():
            for idx, (image, labels) in enumerate(valid_loader, start=1):
                image = image.to(device)
                labels = labels.to(device)

                repr, _ = model(image)
                logits = classifier(repr)
                classfication_loss = ce_loss(logits, labels)

                bsz = image.shape[0]

                if options.dataset == "mixedwm38":
                    probs = logits.sigmoid()
                    predicted = (probs > 0.5).float()
                else:
                    probs = logits.softmax(dim=1)
                    predicted = probs.argmax(1)

                valid_acc += label_hard_acc(predicted, labels)
                valid_loss += classfication_loss.item()
                num_step = idx

        valid_loss /= num_step
        valid_acc /= num_step
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        writer.add_scalar("Loss/valid", valid_loss, epoch)
        writer.add_scalar("Acc/valid", valid_acc, epoch)

        print("Epoch -> ", epoch)
        print(" Training Loss -> ", train_loss, "Valid Loss -> ", valid_loss)
        print(" Training Acc -> ", train_acc, "Valid Acc -> ", valid_acc)

        if min_val_loss > valid_loss:
            print("Writing Model at epoch ", epoch)
            model_pt_file = pt_dir / (model_name + "_model.pt")
            classifier_pt_file = pt_dir / (model_name + "_classifier.pt")
            torch.save(model.state_dict(), model_pt_file)
            torch.save(classifier.state_dict(), classifier_pt_file)
            min_val_loss = valid_loss

        earlystop(valid_loss, optimizer)

        if earlystop.early_stop:
            break

    results = {
        "train_acc": train_accs,
        "train_losses": train_losses,
        "valid_accs": valid_accs,
        "valid_losses": valid_losses,
    }

    return results
