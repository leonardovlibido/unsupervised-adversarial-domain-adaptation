import torch
from collections import defaultdict
from pathlib import Path
from torchviz import make_dot

import sys
sys.path.append("../src")

from data.datasets import initialize_datasets
from models.models import initialize_classification_model


def visualize_backwards():
    model_key = 'mnist_custom_cnn'
    dataset_name = 'mnist'

    # model_key = "svhn_custom_cnn"
    model, input_size = initialize_classification_model(model_key, 10)
    dataloaders, dataset_sizes = initialize_datasets(dataset_name, batch_size=128, input_size=input_size)

    inputs, _ = next(iter(dataloaders['train']))
    yhat = model(inputs)
    make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")

from torchsummary import summary

def main():
    model_key = 'svhn_custom_cnn'
    dataset_name = 'svhn'

    # model_key = "svhn_custom_cnn"
    model, input_size = initialize_classification_model(model_key, 10)
    dataloaders, dataset_sizes = initialize_datasets(dataset_name, batch_size=128, input_size=input_size)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to("cuda:0")

    # inputs, _ = next(iter(dataloaders['train']))
    # yhat = model(inputs)

    print(summary(model, input_size=(3, input_size, input_size)))

if __name__ == "__main__":
    main()
