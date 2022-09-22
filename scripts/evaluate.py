import torch
from collections import defaultdict
from pathlib import Path

import sys
sys.path.append("../src")

from data.datasets import initialize_datasets
from models.models import initialize_classification_model


def evaluate_model(model, dataloader, device):
    model.to(device)
    running_corrects = 0
    total_size = 0

    for inputs1, labels1 in dataloader:
        total_size += inputs1.shape[0]

        inputs1 = inputs1.to(device)
        labels1 = labels1.to(device)

        model(inputs1)

        cls_out1, da_out1 = model(inputs1, 0)
        _, preds1 = torch.max(cls_out1, 1)
        running_corrects += torch.sum(preds1 == labels1.data).item()

    return running_corrects / total_size


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()


def evaluate_models_on_dataset(models_dir: Path, model_key: str, dataset_name: str):
    model, input_size = initialize_classification_model(model_key, 10)
    dataloaders, dataset_sizes = initialize_datasets(dataset_name, batch_size=128, input_size=input_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_accs = {}

    model_paths = [x for x in models_dir.glob('*.pt')]
    model_paths.sort(key = lambda x : int(x.stem[6:]))
    for model_path in model_paths:
        load_model(model, model_path)
        epoch_acc = evaluate_model(model, dataloaders['train'], device)
        epoch_num = int(model_path.stem[6:])
        epoch_accs[epoch_num] = epoch_acc
        print(f'{epoch_num} : {epoch_acc}')

    print(epoch_accs)


def main():
    # models_dir = Path("/home/rdjordjevic/master/repos/domain-adaptation-codebase/experiments/2022-09-12_10-57-37__model-svhn_custom_cnn_data-svhn--svhn_bs-256/models")
    # dataset_name = "mnist"
    # model_key = "svhn_custom_cnn"

    # evaluate_models_on_dataset(models_dir, model_key, dataset_name)

    models_dir = Path("/home/rdjordjevic/master/repos/domain-adaptation-codebase/experiments/2022-09-12_20-24-46__model-mnist_custom_cnn_data-mnist--mnist_bs-256/models")
    dataset_name = "mnistm"
    model_key = "mnist_custom_cnn"

    evaluate_models_on_dataset(models_dir, model_key, dataset_name)


if __name__ == "__main__":
    main()
