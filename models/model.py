import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from models.FE import FE


class model(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size, reduction_channel, img_src, img_tar, tau):
        super(model, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pca_conv = nn.Conv2d(kernel_size=1, stride=1, in_channels=in_channels, out_channels=reduction_channel)

        _, _, eigenvectors = avg_pca(img_src, img_tar, reduction_channel)
        eigenvectors = eigenvectors.reshape(reduction_channel, self.in_channels, 1, 1)
        if isinstance(eigenvectors, np.ndarray):
            eigenvectors = torch.from_numpy(eigenvectors).float()
        with torch.no_grad():
            self.pca_conv.weight.data = eigenvectors
            self.pca_conv.bias.data.zero_()

        self.tau = tau
        self._register_pca_conv_hook()

        self.channel = 256
        self.conv1 = nn.Sequential(
            nn.Conv2d(kernel_size=1, stride=1, in_channels=reduction_channel, out_channels=self.channel),
            nn.BatchNorm2d(self.channel),
            nn.ReLU())
        self.FE_class = FE(self.channel, patch_size)
        self.hidden_channel = self.FE_class.hidden_layer

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_channel, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, source, target):
        class_src, feature_src = self.predict(source)
        class_tar, feature_tar = self.predict(target)

        return class_src, class_tar, feature_src, feature_tar

    def predict(self, x):
        x = self.pca_conv(x)
        x = self.conv1(x)
        feature = self.FE_class(x)
        x = self.classifier(feature)
        return x, feature

    def _register_pca_conv_hook(self):
        def hook_fn(module, grad_input, grad_output):
            if grad_input is None or all(g is None for g in grad_input):
                print("梯度未能成功传递。")
            else:
                modified_grad = tuple(grad * self.tau if grad is not None else None for grad in grad_input)
                return modified_grad

        self.pca_conv.register_backward_hook(hook_fn)
        print(f"已为 pca_conv 层注册梯度修改 Hook，梯度将乘以 {self.tau}。")

def avg_pca(data1, data2, n_components):
    # data1: [row1, column1, band]; data2: [row2, column2, band]
    reshaped_data1 = data1.reshape(data1.shape[0] * data1.shape[1], data1.shape[2])
    reshaped_data2 = data2.reshape(data2.shape[0] * data2.shape[1], data2.shape[2])

    pca1 = PCA(n_components)
    pca2 = PCA(n_components)

    pca1.fit(reshaped_data1)
    pca2.fit(reshaped_data2)

    eigenvalues1, eigenvectors1 = pca1.explained_variance_, pca1.components_
    eigenvalues2, eigenvectors2 = pca2.explained_variance_, pca2.components_

    average_eigenvectors = (eigenvectors1 + eigenvectors2) / 2

    projected_data1 = np.dot(reshaped_data1, average_eigenvectors.T)
    projected_data2 = np.dot(reshaped_data2, average_eigenvectors.T)

    principal_components1_reshaped = projected_data1.reshape(data1.shape[0], data1.shape[1], n_components)
    principal_components2_reshaped = projected_data2.reshape(data2.shape[0], data2.shape[1], n_components)

    return principal_components1_reshaped, principal_components2_reshaped,average_eigenvectors