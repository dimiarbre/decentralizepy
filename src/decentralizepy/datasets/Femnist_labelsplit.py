import json
import logging
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from decentralizepy.datasets.Data import Data
from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.datasets.Femnist import Femnist
from decentralizepy.datasets.Partitioner import DataPartitioner
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model
from decentralizepy.models.Resnet import BasicBlock, Bottleneck, conv1x1

NUM_CLASSES = 62
IMAGE_SIZE = (28, 28)
FLAT_SIZE = 28 * 28
PIXEL_RANGE = 256.0


class FemnistLabelSplit(Femnist):
    """
    Class for the FEMNIST dataset

    """

    def load_trainset(self):
        """
        Loads the training set. Partitions it if needed.

        """
        logging.info("Loading training set.")
        files = os.listdir(self.train_dir)
        files = [f for f in files if f.endswith(".json")]
        files.sort()
        c_len = len(files)

        if self.sizes == None:  # Equal distribution of data among processes
            e = c_len // self.num_partitions
            frac = e / c_len
            self.sizes = [frac] * self.num_partitions
            self.sizes[-1] += 1.0 - frac * self.num_partitions
            logging.debug("Size fractions: {}".format(self.sizes))

        my_clients = DataPartitioner(files, self.sizes, seed=self.random_seed).use(
            self.dataset_id
        )
        my_train_data = {"x": [], "y": []}
        self.clients = []
        self.num_samples = []
        logging.debug("Clients Length: %d", c_len)
        logging.debug("My_clients_len: %d", my_clients.__len__())
        for i in range(my_clients.__len__()):
            cur_file = my_clients.__getitem__(i)

            clients, _, train_data = self.__read_file__(
                os.path.join(self.train_dir, cur_file)
            )
            for cur_client in clients:
                self.clients.append(cur_client)
                my_train_data["x"].extend(train_data[cur_client]["x"])
                my_train_data["y"].extend(train_data[cur_client]["y"])
                self.num_samples.append(len(train_data[cur_client]["y"]))
        self.train_x = (
            np.array(my_train_data["x"], dtype=np.dtype("float32"))
            .reshape(-1, 28, 28, 1)
            .transpose(0, 3, 1, 2)
        )
        self.train_y = np.array(my_train_data["y"], dtype=np.dtype("int64")).reshape(-1)
        logging.info("train_x.shape: %s", str(self.train_x.shape))
        logging.info("train_y.shape: %s", str(self.train_y.shape))
        assert self.train_x.shape[0] == self.train_y.shape[0]
        assert self.train_x.shape[0] > 0

        # Plot the first few examples and save the images. This is for debug purposes
        # TODO: remove this block once we are sure partitionning and post-regeneration is correct
        node_data = Data(self.train_x, self.train_y)
        # subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(10, 1)
        for i in range(10):
            to_plot = node_data.x[i][0]
            axarr[i].imshow(to_plot)
        f.savefig(
            os.path.join(
                f"/tmp/logs/machine{self.machine_id}/", f"dataset{self.dataset_id}.png"
            )
        )

        if self.__validating__ and self.validation_source == "Train":
            num_samples = int(self.train_x.shape[0] * self.validation_size)
            validation_indexes = np.random.choice(
                self.train_x.shape[0], num_samples, replace=False
            )

            self.validation_x = self.train_x[validation_indexes]
            self.validation_y = self.train_y[validation_indexes]

            self.train_x = np.delete(self.train_x, validation_indexes, axis=0)
            self.train_y = np.delete(self.train_y, validation_indexes, axis=0)

    def load_testset(self):
        """
        Loads the testing set.

        """
        logging.info("Loading testing set.")
        _, _, d = self.__read_dir__(self.test_dir)
        test_x = []
        test_y = []
        for test_data in d.values():
            for x in test_data["x"]:
                test_x.append(x)
            for y in test_data["y"]:
                test_y.append(y)
        self.test_x = (
            np.array(test_x, dtype=np.dtype("float32"))
            .reshape(-1, 28, 28, 1)
            .transpose(0, 3, 1, 2)
        )
        self.test_y = np.array(test_y, dtype=np.dtype("int64")).reshape(-1)
        logging.info("test_x.shape: %s", str(self.test_x.shape))
        logging.info("test_y.shape: %s", str(self.test_y.shape))
        assert self.test_x.shape[0] == self.test_y.shape[0]
        assert self.test_x.shape[0] > 0

        if self.__validating__ and self.validation_source == "Test":
            num_samples = int(self.test_x.shape[0] * self.validation_size)
            validation_indexes = np.random.choice(
                self.test_x.shape[0], num_samples, replace=False
            )

            self.validation_x = self.test_x[validation_indexes]
            self.validation_y = self.test_y[validation_indexes]

            self.test_x = np.delete(self.test_x, validation_indexes, axis=0)
            self.test_y = np.delete(self.test_y, validation_indexes, axis=0)


class LogisticRegression(Model):
    """
    Class for a Logistic Regression Neural Network for FEMNIST

    """

    def __init__(self):
        """
        Constructor. Instantiates the Logistic Regression Model
            with 28*28 Input and 62 output classes

        """
        super().__init__()
        self.fc1 = nn.Linear(FLAT_SIZE, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model

        Parameters
        ----------
        x : torch.tensor
            The input torch tensor

        Returns
        -------
        torch.tensor
            The output torch tensor

        """
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x


class CNN(Model):
    """
    Class for a CNN Model for FEMNIST

    """

    def __init__(self):
        """
        Constructor. Instantiates the CNN Model
            with 28*28*1 Input and 62 output classes

        """
        super().__init__()
        # 1.6 million params
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model

        Parameters
        ----------
        x : torch.tensor
            The input torch tensor

        Returns
        -------
        torch.tensor
            The output torch tensor

        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RNET(Model):
    """
    From PyTorch:
    Class for a Resnet Model for FEMNIST
    Copied and modified from https://github.com/pytorch/pytorch/blob/75024e228ca441290b6a1c2e564300ad507d7af6/benchmarks/functional_autograd_benchmark/torchvision_models.py
    For the license see models/Resnet.py
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        zero_init_residual=False,
        groups=1,
        width_per_group=32,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(RNET, self).__init__()
        block = BasicBlock
        layers = [2, 2, 2, 2]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(
            block, 64, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 128, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
