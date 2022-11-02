import pathlib
import torch

from models.vgg import SupConVGG, VGGLinearClassifier
from utils.tools import calculate_metrics


def test_supcon(options, test_loader, label_counts, augmentation: dict):

    model_name = "vgg" + options.model_config
    pt_path = pathlib.Path("saved")

    pt_dir = (
        pt_path
        / options.dataset
        / str(options.exp_id)
        / (model_name + options.head)
        / str(options.set_size)
        / str(options.gamma)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_pt_file = pt_dir / (model_name + "_model.pt")
    classifier_pt_file = pt_dir / (model_name + "_classifier.pt")

    if options.dataset == "mixedwm38":
        numclasses = 8
        label_index = None
    elif options.dataset == "wm811k":
        numclasses = 9
        label_index = label_counts.index

    model = SupConVGG(name=model_name, head=options.head, feat_dim=128).to(device)
    classifier = VGGLinearClassifier(
        encoder_name=model_name, num_classes=numclasses
    ).to(device)

    test_loader.dataset.set_transform(augmentation["to_tensor"])

    model.load_state_dict(torch.load(model_pt_file))
    model = model.to(device)

    classifier.load_state_dict(torch.load(classifier_pt_file))
    classifier = classifier.to(device)

    y_gt = torch.Tensor().to(device)
    y_hat = torch.Tensor().to(device)

    model.eval()
    classifier.eval()
    with torch.no_grad():
        for idx, (image, target) in enumerate(test_loader):

            image = image.to(device)
            target = target.to(device)

            repr, _ = model(image)
            output = classifier(repr)

            if options.dataset == "mixedwm38":
                probs = output.sigmoid()
                predicted = (probs > 0.5).float()
            else:
                probs = output.softmax(dim=1)
                predicted = probs.argmax(1)

            y_gt = torch.cat((y_gt, target), 0)
            y_hat = torch.cat((y_hat, predicted), 0)

    results = calculate_metrics(
        options=options, pred=y_hat, target=y_gt, label_index=label_index
    )
    torch.cuda.empty_cache()
    return results
