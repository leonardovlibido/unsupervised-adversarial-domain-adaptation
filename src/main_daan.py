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
from models.helper_functions import freeze_model, unfreeze_model


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


def log_basics_to_tensorboard(writer, model, dataloaders, device, dataset_name):
    sample_imgs, sample_lbls = next(iter(dataloaders['train']))
    img_grid = make_grid(sample_imgs)
    matplotlib_imshow(img_grid)
    writer.add_image(f'{dataset_name}', img_grid)

    # features = sample_imgs.view(-1, prod(sample_imgs.shape[1:]))
    # writer.add_embedding(features, metadata=sample_lbls, label_img=sample_imgs)

    sample_imgs = sample_imgs.to(device)
    writer.add_graph(model, sample_imgs)


def train_daan_model(
        model: nn.Module,
        criterion1: nn.Module,
        criterion2: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloaders1: Dict[str, DataLoader],
        dataloaders2: Dict[str, DataLoader],
        dataset1_sizes: Dict,
        dataset2_sizes: Dict,
        dataset1_name: str,
        dataset2_name: str,
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

    log_basics_to_tensorboard(writer, model, dataloaders1, device, dataset1_name)
    log_basics_to_tensorboard(writer, model, dataloaders2, device, dataset2_name)

    model_dir = experiment_dir / "models"
    model_dir.mkdir()

    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch}/{num_epochs - 1}')
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # running_loss = 0.0
            # running_corrects = 0
            running_losses = defaultdict(float)
            running_corrects = defaultdict(float)

            # Iterate over data.
            i = 0
            for (inputs1, labels1), (inputs2, labels2) in zip(dataloaders1[phase], dataloaders2[phase]):

                i += labels1.shape[0]
                i += labels2.shape[0]

                if phase == 'train':
                    p = float(i + epoch * (dataset1_sizes['train'] + dataset2_sizes['train'])) / num_epochs / (dataset1_sizes['train'] + dataset2_sizes['train'])
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                else:
                    p = float((1 + epoch) * (dataset1_sizes['train'] + dataset2_sizes['train'])) / num_epochs / (dataset1_sizes['train'] + dataset2_sizes['train'])
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1

                if i % (4*1024) == 0:
                    logging.info(i)
                    # print('alpha', alpha)
                inputs1 = inputs1.to(device)
                labels1 = labels1.to(device)

                inputs2 = inputs2.to(device)
                labels2 = labels2.to(device)

                # inputs = inputs1
                # labels = labels1

                # model.zero_grad()
                model.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    cls_out1, da_out1 = model(inputs1, alpha)

                    _, preds1 = torch.max(cls_out1, 1)
                    loss_cls1 = criterion1(cls_out1, labels1)

                    labels_da1 = torch.zeros_like(da_out1)
                    loss_da1 = criterion2(da_out1, labels_da1)

                    # model.eval()
                    # model.train()
                    # with torch.no_grad():
                    # model.eval()
                    # model.train()
                    # with torch.no_grad():

                    # if phase == 'train':
                    #     cls_out2, da_out2 = cls_out1, da_out1
                    # else:
                    #     cls_out2, da_out2 = cls_out1, da_out1

                    # cls_out2, da_out2 = model(inputs2, alpha)
                    cls_out2, da_out2 = cls_out1, da_out1

                    labels_da2 = torch.ones_like(da_out2)
                    loss_da2 = criterion2(da_out2, labels_da2)

                    preds_da1 = da_out1 > 0.0
                    preds_da2 = da_out2 > 0.0

                    loss = loss_cls1 + loss_da1 + loss_da2

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_cls1.backward()
                        # loss_cls1.backward()
                        optimizer.step()

                    _, preds2 = torch.max(cls_out2, 1)
                    loss_cls2 = criterion1(cls_out2, labels2)

                # statistics

                running_losses['cls1'] += loss_cls1.item() * inputs1.size(0)
                running_losses['cls2'] += loss_cls2.item() * inputs2.size(0)
                running_losses['da'] += loss_da1.item() * inputs1.size(0)
                running_losses['da'] += loss_da2.item() * inputs2.size(0)

                running_corrects['cls1'] += torch.sum(preds1 == labels1.data).item()
                running_corrects['cls2'] += torch.sum(preds2 == labels2.data).item()
                running_corrects['da'] += torch.sum(preds_da1 == labels_da1.data).item()
                running_corrects['da'] += torch.sum(preds_da2 == labels_da2.data).item()

                # running_loss += loss_cls1.item() * inputs1.size(0)
                # running_corrects += torch.sum(preds == labels1.data).item()

            # epoch_loss = running_loss / dataset1_sizes[phase]
            # epoch_acc = float(running_corrects) / dataset1_sizes[phase]

            epoch_loss = {}
            epoch_acc = {}
            epoch_loss['cls1'] = running_losses['cls1'] / dataset1_sizes[phase]

            epoch_loss['cls2'] = running_losses['cls2'] / dataset2_sizes[phase]
            epoch_loss['da'] = running_losses['da'] / (dataset1_sizes[phase] + dataset2_sizes[phase])

            epoch_acc['cls1'] = float(running_corrects['cls1']) / dataset1_sizes[phase]
            epoch_acc['cls2'] = float(running_corrects['cls2']) / dataset2_sizes[phase]
            epoch_acc['da'] = float(running_corrects['da']) / (dataset1_sizes[phase] + dataset2_sizes[phase])

            print(f'{phase} Loss: {epoch_loss["cls1"]:.4f} Acc: {epoch_acc["cls1"]:.4f}')

            metrics[phase + "_loss"].append(epoch_loss)
            metrics[phase + "_acc"].append(epoch_acc)

            # Log to TensorBoard
            writer.add_scalar(f'Accuracy/CLS/{dataset1_name}_{phase}', epoch_acc['cls1'], epoch)
            writer.add_scalar(f'Accuracy/CLS/{dataset2_name}_{phase}', epoch_acc['cls2'], epoch)
            writer.add_scalar(f'Accuracy/DA/{dataset1_name}-and-{dataset2_name}_{phase}', epoch_acc['da'], epoch)
            writer.add_scalar(f'Loss/CLS/{dataset1_name}_{phase}', epoch_loss['cls1'], epoch)
            writer.add_scalar(f'Loss/CLS/{dataset2_name}_{phase}', epoch_loss['cls2'], epoch)
            writer.add_scalar(f'Loss/DA/{dataset1_name}-and-{dataset2_name}_{phase}', epoch_loss['da'], epoch)

            # Save model
            if epoch % 2 == 0:
                torch.save(model.state_dict(), model_dir / f"model_{epoch}.pt")

            # deep copy the model
            if phase == 'val' and epoch_acc['cls1'] > best_acc:
                best_acc = epoch_acc['cls1']
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start_time
    print(f'Training complete in {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics


def main_daan():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
    )

    # Parameters
    # dataset_name = 'svhn'
    dataset_name1 = 'mnistm'
    dataset_name2 = 'mnistm'
    # model_key = 'resnet'
    # model_key = 'mobilenetv2'
    model_key = 'svhn_custom_cnn'
    output_labels = [str(x) for x in range(10)]
    num_of_labels = len(output_labels)
    num_epochs = 50
    batch_size = 128 * 2
    train_subset_size = 40000
    val_subset_size = 8000
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
    exp_name = f'{timestamp}_model-{model_key}_data-{dataset_name1}--{dataset_name2}_bs-{batch_size}'
    experiment_path = Path("/home/rdjordjevic/master/repos/domain-adaptation-codebase/experiments") / exp_name
    learning_rate = 0.01
    # weight_decay = 1e-6
    # weight_decay = 1e-3
    weight_decay = 0.001
    momentum = 0.9

    # Initializations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device being used: {device}')

    model, input_size = initialize_classification_model(model_key, num_of_labels)

    dataloaders1, dataset1_sizes = initialize_datasets(
        dataset_name1, batch_size // 2, input_size, train_subset_size, val_subset_size)
    dataloaders2, dataset2_sizes = initialize_datasets(
        dataset_name2, batch_size // 2, input_size, train_subset_size, val_subset_size)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    best_model, metrics = train_daan_model(
        model,
        criterion1, criterion2,
        optimizer,
        dataloaders1, dataloaders2,
        dataset1_sizes, dataset2_sizes,
        dataset_name1, dataset_name2+'2',
        device, experiment_path, num_epochs)
    torch.save(best_model.state_dict(), experiment_path / "model.pt")

    print(metrics)


if __name__ == "__main__":
    main_daan()

    # pred = net(input)
    # loss = crit(pred, target)
    # loss.backward()
    # pred = net(input)
    # loss = crit(pred, target)
    # loss.backward()
    # pred = net(input)
    # loss = crit(pred, target)
    # loss.backward()
    # opt.step()
    # opt.zero_grad()