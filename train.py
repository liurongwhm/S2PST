import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader
from dataloader import get_sample_index, ListDataset, data_prefetcher
from loss import self_training_conditional
import os
import scipy.io as io
from models.model import model
from utils_HSI import sample_gt, get_device, seed_worker, metrics
from datasets import get_dataset
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='PyTorch NEW')

dataset_name = 'Houston'  # Houston,HyRANK,Pavia
parser.add_argument('--data_path', type=str, default='./dataset/' + dataset_name + '/',
                    help='the path to load the data')
if dataset_name == 'Houston':
    parser.add_argument('--source_name', type=str, default='Houston13',
                        help='the name of the source dir')
    parser.add_argument('--target_name', type=str, default='Houston18',
                        help='the name of the test dir')
    patch_size = 13
    PCA_channel = 7
    lambda_BNM = 0.5
    tau = 0.2

if dataset_name == 'HyRANK':
    parser.add_argument('--source_name', type=str, default='Dioni',
                        help='the name of the source dir')
    parser.add_argument('--target_name', type=str, default='Loukia',
                        help='the name of the test dir')
    patch_size = 5
    PCA_channel = 3
    lambda_BNM = 0.5
    tau = 1

if dataset_name == 'Pavia':
    parser.add_argument('--source_name', type=str, default='PaviaU',
                        help='the name of the source dir')
    parser.add_argument('--target_name', type=str, default='PaviaC',
                        help='the name of the test dir')
    patch_size = 13
    PCA_channel = 9
    lambda_BNM = 1
    tau = 0.2

parser.add_argument('--save_path', type=str, default="./results", help='the path to save the model')
parser.add_argument('--cuda', type=int, default=0, help="Specify CUDA device")

# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
group_train.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
group_train.add_argument('--batch_size', type=int, default=100, help="Batch size")
group_train.add_argument('--seed', type=int, default=2025, help='random seed')
group_train.add_argument('--l2_decay', type=float, default=1e-4, help='the L2 weight decay')
group_train.add_argument('--num_epoch', type=int, default=100, help='the number of epoch')
group_train.add_argument('--num_trials', type=int, default=10, help='the number of trials')
group_train.add_argument('--training_sample_number_per_class', type=float, default=200, help='training sample number')

args = parser.parse_args()
DEVICE = get_device(args.cuda)


def train(epoch, model, train_src_dataloader, train_tar_dataloader, num_epoch, lambda_BNM):
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch) / num_epoch), 0.75)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=args.momentum, weight_decay=args.l2_decay)
    CNN_correct, CNN_tar_correct = 0, 0

    iter_source = data_prefetcher(train_src_dataloader)
    iter_target = data_prefetcher(train_tar_dataloader)
    num_iter = len(train_src_dataloader)

    for i in range(num_iter):
        model.train()
        data_src, label_src = iter_source.next()
        data_tar, label_tar = iter_target.next()
        label_src = label_src - 1
        label_tar = label_tar - 1

        optimizer.zero_grad()

        class_c_src, class_c_tar, _, _ = model(data_src, data_tar)
        loss_cls = F.nll_loss(torch.log(class_c_src), label_src.long())

        list_svd, _ = torch.sort(torch.sqrt(torch.sum(torch.pow(class_c_tar, 2), dim=0)), descending=True)
        loss_BNM_tar = - torch.mean(list_svd[:min(class_c_tar.shape[0], class_c_tar.shape[1])])

        self_training_ = self_training_conditional(threshold=0.95)
        loss_self_training, mask, _, _ = self_training_(class_c_tar, class_c_src)

        loss_total = loss_cls + lambda_BNM * loss_BNM_tar + loss_self_training

        loss_total.backward()
        optimizer.step()

        pred = class_c_src.data.max(1)[1]
        CNN_correct += pred.eq(label_src.data.view_as(pred)).cpu().sum()
        pred_tar = class_c_tar.data.max(1)[1]
        CNN_tar_correct += pred_tar.eq(label_tar.data.view_as(pred_tar)).cpu().sum()

    CNN_acc = CNN_correct.item() / len(train_src_dataloader.dataset)
    CNN_tar_acc = CNN_tar_correct.item() / len(train_tar_dataloader.dataset)

    print(f'Epoch {epoch + 1}: Train Accuracy: {CNN_acc:.4f}, Target Accuracy: {CNN_tar_acc:.4f}')


def test(model, test_loader):
    model.eval()
    correct = 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            label = label - 1
            soft_pred, _ = model.predict(data)
            pred = soft_pred.data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    accuracy = correct.item() / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy, pred_list, label_list


if __name__ == '__main__':
    seed_worker(args.seed)
    acc_test_list = np.zeros([args.num_trials, 1])

    for flag in range(args.num_trials):
        img_src, gt_src_, LABEL_VALUES_src, _, _, _ = get_dataset(args.source_name, args.data_path)
        img_tar, gt_tar_, LABEL_VALUES_tar, IGNORED_LABELS, _, _ = get_dataset(args.target_name, args.data_path)
        N_CLASS = int(gt_tar_.max())
        N_BANDS = img_src.shape[-1]
        N_ROWS = img_tar.shape[0]
        N_COLS = img_tar.shape[1]

        r = int(patch_size / 2) + 1
        img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
        img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
        gt_src = np.pad(gt_src_, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
        gt_tar = np.pad(gt_tar_, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

        train_src_index = get_sample_index(gt_src, args.training_sample_number_per_class)
        train_tar_index = get_sample_index(gt_tar, args.training_sample_number_per_class)
        test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')

        train_dataset = ListDataset(img_src, gt_src, train_src_index, patch_size)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        train_tar_dataset = ListDataset(img_tar, gt_tar, train_tar_index, patch_size)
        train_tar_loader = DataLoader(train_tar_dataset, shuffle=True, batch_size=args.batch_size)
        test_dataset = ListDataset(img_tar, test_gt_tar, np.argwhere(test_gt_tar != 0), patch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

        model_ = model(N_BANDS, num_classes=N_CLASS, patch_size=patch_size, reduction_channel=PCA_channel,
                       img_src=img_src, img_tar=img_tar, tau=tau).to(DEVICE)

        # train
        for epoch in range(args.num_epoch):
            train(epoch, model_, train_loader, train_tar_loader, args.num_epoch, lambda_BNM)

        # test
        test_acc, pred, label = test(model_, test_loader)
        acc_test_list[flag] = test_acc
        print(classification_report(np.concatenate(pred), np.concatenate(label), target_names=LABEL_VALUES_tar))
        results = metrics(np.concatenate(pred), np.concatenate(label), ignored_labels=IGNORED_LABELS,
                          n_classes=gt_src.max())

        # output classification map
        prediction_matrix = np.zeros((N_ROWS, N_COLS), dtype=int)
        for index in range(len(np.nonzero(gt_tar_)[0])):
            prediction_matrix[int(test_dataset.indices[index][0] - r)][int(test_dataset.indices[index][1] - r)] = \
                results['prediction'][index] + 1

        io.savemat(os.path.join(args.save_path,
                                'ClassificationMap_' + str(int(flag + 1)) + 'times_' + args.target_name + '.mat'),
                   {'ClassificationMap': prediction_matrix})
    print(f'Average Test Accuracy: {np.mean(acc_test_list):.5f}')
