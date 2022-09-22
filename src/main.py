from typing import Dict
import torch
from torch import nn
from torch.utils.data import DataLoader
from collections import defaultdict
import copy
import time
import logging
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from math import prod

from torch.utils.tensorboard.writer import SummaryWriter

from data.datasets import initialize_datasets
from models.models import initialize_classification_model


# writer = SummaryWriter()
# for n_iter in range(100):
#     writer.add_scalar('Loss/train', np.random.random(), n_iter)
#     writer.add_scalar('Loss/test', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def train_model(
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloaders: Dict[str, DataLoader],
        dataset_sizes: Dict,
        device: torch.device,
        experiment_dir: Path,
        num_epochs: int = 25,
):

    start_time = time.time()

    logging.info(f'Putting model on {device}')
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    metrics = defaultdict(list)
    writer = SummaryWriter(log_dir=experiment_dir / "tb_log")

    sample_imgs, sample_lbls = next(iter(dataloaders['train']))
    img_grid = make_grid(sample_imgs)
    matplotlib_imshow(img_grid)
    writer.add_image('mnist_images', img_grid)

    # log embeddings
    features = sample_imgs.view(-1, prod(sample_imgs.shape[1:]))
    writer.add_embedding(features, metadata=sample_lbls, label_img=sample_imgs)

    sample_imgs = sample_imgs.to(device)
    writer.add_graph(model, sample_imgs)

    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch}/{num_epochs - 1}')
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            i = 0
            for inputs, labels in dataloaders[phase]:
                i += 32
                if i % 1024 == 0:
                    logging.info(i)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # if i > 10000:
                #     continue

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # outputs = model(inputs)
                    outputs, _ = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    # preds = torch.argmax(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            metrics[phase + "_loss"].append(epoch_loss)
            metrics[phase + "_acc"].append(epoch_acc)

            # Log to TensorBoard
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start_time
    print(f'Training complete in {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')
    print('Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
    )

    # Parameters
    dataset_name = 'svhn'
    # dataset_name = 'mnist'
    # model_key = 'resnet'
    # model_key = 'mobilenetv2'
    model_key = 'mnist_custom_cnn'
    output_labels = [str(x) for x in range(10)]
    num_of_labels = len(output_labels)
    num_epochs = 30
    batch_size = 64
    train_subset_size = 2000
    val_subset_size = 2000
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
    exp_name = f'{timestamp}_model-{model_key}_data-{dataset_name}_bs-{batch_size}'
    experiment_path = Path("/home/rdjordjevic/master/repos/domain-adaptation-codebase/experiments") / exp_name

    # Initializations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device being used: {device}')

    model, input_size = initialize_classification_model(model_key, num_of_labels)

    dataloaders, dataset_sizes = initialize_datasets(
        dataset_name, batch_size, input_size, train_subset_size, val_subset_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    best_model, metrics = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, experiment_path, num_epochs)
    torch.save(best_model.state_dict(), experiment_path / "model.pt")

    print(metrics)


if __name__ == "__main__":
    main()
