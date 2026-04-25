import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
from .contrastive_loss import *


def _resolve_device(device):
    if isinstance(device, str) and device == 'cuda' and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return device


def _resolve_dataset_targets(dataset):
    if hasattr(dataset, 'targets'):
        return dataset.targets
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        base_targets = getattr(dataset.dataset, 'targets', None)
        if base_targets is not None:
            return [base_targets[index] for index in dataset.indices]
    raise AttributeError('Unable to resolve dataset targets for kNN evaluation.')


def _make_progress(iterable, desc=None):
    return tqdm(iterable, desc=desc, disable=False)

def train(model, data_loader, train_optimizer, epoch, epochs, batch_size=32, temperature=0.5, device='cuda'):
    """Trains the model defined in ./model.py with one epoch.
    
    Inputs:
    - model: Model class object as defined in ./model.py.
    - data_loader: torch.utils.data.DataLoader object; loads in training data. You can assume the loaded data has been augmented.
    - train_optimizer: torch.optim.Optimizer object; applies an optimizer to training.
    - epoch: integer; current epoch number.
    - epochs: integer; total number of epochs.
    - batch_size: Number of training samples per batch.
    - temperature: float; temperature (tau) parameter used in simclr_loss_vectorized.
    - device: the device name to define torch tensors.

    Returns:
    - The average loss.
    """
    device = _resolve_device(device)
    model.train()
    total_loss, total_num, train_bar = 0.0, 0, _make_progress(data_loader)
    for data_pair in train_bar:
        x_i, x_j, target = data_pair
        x_i, x_j = x_i.to(device), x_j.to(device)
        
        out_left, out_right, loss = None, None, None
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Take a look at the model.py file to understand the model's input and output.
        # Run x_i and x_j through the model to get out_left, out_right.              #
        # Then compute the loss using simclr_loss_vectorized.                        #
        ##############################################################################
        _, out_left = model(x_i)
        _, out_right = model(x_j)
        loss = simclr_loss_vectorized(out_left, out_right, temperature, device=device)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


def train_val(model, data_loader, train_optimizer, epoch, epochs, device='cuda'):
    device = _resolve_device(device)
    is_train = train_optimizer is not None
    model.train() if is_train else model.eval()
    # Keep a frozen backbone in eval mode so BatchNorm does not switch back to
    # the slower training path during linear probing / finetuning cells.
    if is_train and hasattr(model, 'f'):
        backbone_params = list(model.f.parameters())
        if backbone_params and all(not param.requires_grad for param in backbone_params):
            model.f.eval()
    loss_criterion = torch.nn.CrossEntropyLoss()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, _make_progress(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            topk = min(5, out.size(-1))
            prediction = torch.topk(out, k=topk, dim=-1).indices
            target_view = target.unsqueeze(dim=-1)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target_view).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, :topk] == target_view).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


def test(model, memory_data_loader, test_data_loader, epoch, epochs, c, temperature=0.5, k=200, device='cuda'):
    device = _resolve_device(device)
    model.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in _make_progress(memory_data_loader, desc='Feature extracting'):
            feature, out = model(data.to(device))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(_resolve_dataset_targets(memory_data_loader.dataset), device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = _make_progress(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.to(device), target.to(device)
            feature, out = model(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            
            # [B, K]
            if device == 'mps' or (hasattr(device, 'type') and device.type == 'mps'):
                # MPS currently supports topk only for k<=16; run kNN selection on CPU.
                sim_weight, sim_indices = sim_matrix.cpu().topk(k=k, dim=-1)
                sim_weight = sim_weight.to(feature.device)
                sim_indices = sim_indices.to(feature.device)
            else:
                sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = torch.topk(pred_scores.cpu(), k=min(5, c), dim=-1).indices
            target_cpu = target.cpu().unsqueeze(dim=-1)
            total_top1 += torch.sum((pred_labels[:, :1] == target_cpu).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :min(5, c)] == target_cpu).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100
