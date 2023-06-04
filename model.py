import sys
from basic_config import PATH_TO_CIFAR
sys.path.append(PATH_TO_CIFAR)

import torch.nn as nn
import torch.nn.functional as F
import train as cifar_train


def get_model_from_name(args, idx=0, model_name=None, model_config=None):
    if idx != -1 and idx == (args.num_models - 1):
        # only passes for the second model
        width_ratio = args.width_ratio
    else:
        width_ratio = -1

    if model_name is None:
        model_name = args.model_name_list[idx]

    if model_name == "net":
        return Net(args)
    elif model_name == "mlpnet":
        if args.parse_config:
            assert 0 <= idx <= args.num_models - 1
            hidden_layer_sizes = getattr(args, f"model{idx}_config")
            return MlpNetFromConfig(args, hidden_layer_sizes, width_ratio=width_ratio)
        else:
            return MlpNet(args, width_ratio=width_ratio)
    elif model_name == "cifarmlpnet":
        return CifarMlpNet(args)
    elif model_name[0:3] == "vgg" or model_name[0:3] == "res":
        barebone_config = {"model": model_name, "dataset": args.dataset}

        # if you want pre-relu acts, set relu_inplace to False
        return cifar_train.get_model(
            barebone_config, args.gpu_id, relu_inplace=not args.prelu_acts, model_config=model_config
        )


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=not args.disable_bias)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias=not args.disable_bias)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50, bias=not args.disable_bias)
        self.fc2 = nn.Linear(50, 10, bias=not args.disable_bias)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MlpNet(nn.Module):
    def __init__(self, args, width_ratio=-1):
        super(MlpNet, self).__init__()
        if args.dataset == "mnist":
            # 28 x 28 x 1
            input_dim = 784
        elif args.dataset.lower()[0:7] == "cifar10":
            # 32 x 32 x 3
            input_dim = 3072
        elif args.dataset.lower() == "tinyimagenet":
            # 64 x 64 x 3
            input_dim = 12288
        if width_ratio != -1:
            self.width_ratio = width_ratio
        else:
            self.width_ratio = 1
        self.num_hidden_layers = args.num_hidden_layers

        self.fc = {}
        for layer_idx in range(self.num_hidden_layers + 1):
            if layer_idx == 0:  # input layer
                self.fc[layer_idx] = nn.Linear(
                    input_dim,
                    int(getattr(args, "num_hidden_nodes" + str(layer_idx + 1)) / self.width_ratio),
                    bias=not args.disable_bias,
                )
                self.add_module("fc" + str(layer_idx + 1), self.fc[layer_idx])
            elif layer_idx == self.num_hidden_layers:  # output layer
                self.fc[layer_idx] = nn.Linear(
                    int(getattr(args, "num_hidden_nodes" + str(layer_idx)) / self.width_ratio),
                    10,
                    bias=not args.disable_bias,
                )
                self.add_module("fc" + str(layer_idx + 1), self.fc[layer_idx])
            else:  # hidden layer
                self.fc[layer_idx] = nn.Linear(
                    int(getattr(args, "num_hidden_nodes" + str(layer_idx)) / self.width_ratio),
                    int(getattr(args, "num_hidden_nodes" + str(layer_idx + 1)) / self.width_ratio),
                    bias=not args.disable_bias,
                )
                self.add_module("fc" + str(layer_idx + 1), self.fc[layer_idx])

        self.enable_dropout = args.enable_dropout

    def forward(self, x, disable_logits=False):
        x = x.view(x.shape[0], -1)
        for layer_idx in range(0, self.num_hidden_layers):
            x = F.relu(self.fc[layer_idx](x))
            if self.enable_dropout:
                x = F.dropout(x, training=self.training)
        x = self.fc[self.num_hidden_layers](x)

        if disable_logits:
            return x
        else:
            return F.log_softmax(x, dim=1)


class MlpNetFromConfig(nn.Module):
    def __init__(self, args, hidden_layer_sizes, width_ratio=-1):
        super(MlpNetFromConfig, self).__init__()
        if args.dataset == "mnist":
            # 28 x 28 x 1
            input_dim = 784
        elif args.dataset.lower()[0:7] == "cifar10":
            # 32 x 32 x 3
            input_dim = 3072
        elif args.dataset.lower() == "tinyimagenet":
            # 64 x 64 x 3
            input_dim = 12288
        if width_ratio != -1:
            self.width_ratio = width_ratio
        else:
            self.width_ratio = 1
        self.num_hidden_layers = len(hidden_layer_sizes)

        self.fc = {}
        for layer_idx in range(self.num_hidden_layers + 1):
            if layer_idx == 0:  # input layer
                self.fc[layer_idx] = nn.Linear(
                    input_dim, int(hidden_layer_sizes[0] / self.width_ratio), bias=not args.disable_bias
                )
                self.add_module("fc" + str(layer_idx + 1), self.fc[layer_idx])
            elif layer_idx == self.num_hidden_layers:  # output layer
                self.fc[layer_idx] = nn.Linear(
                    int(hidden_layer_sizes[-1] / self.width_ratio), 10, bias=not args.disable_bias
                )
                self.add_module("fc" + str(layer_idx + 1), self.fc[layer_idx])
            else:  # hidden layer
                self.fc[layer_idx] = nn.Linear(
                    int(hidden_layer_sizes[layer_idx - 1] / self.width_ratio),
                    int(hidden_layer_sizes[layer_idx] / self.width_ratio),
                    bias=not args.disable_bias,
                )
                self.add_module("fc" + str(layer_idx + 1), self.fc[layer_idx])

        self.enable_dropout = args.enable_dropout

    def forward(self, x, disable_logits=False):
        x = x.view(x.shape[0], -1)
        for layer_idx in range(0, self.num_hidden_layers):
            x = F.relu(self.fc[layer_idx](x))
            if self.enable_dropout:
                x = F.dropout(x, training=self.training)
        x = self.fc[self.num_hidden_layers](x)

        if disable_logits:
            return x
        else:
            return F.log_softmax(x, dim=1)


class CifarMlpNet(nn.Module):
    def __init__(self, args):
        super(CifarMlpNet, self).__init__()
        input_dim = 3072
        self.fc1 = nn.Linear(input_dim, 1024, bias=not args.disable_bias)
        self.fc2 = nn.Linear(1024, 512, bias=not args.disable_bias)
        self.fc3 = nn.Linear(512, 128, bias=not args.disable_bias)
        self.fc4 = nn.Linear(128, 10, bias=not args.disable_bias)
        self.enable_dropout = args.enable_dropout

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)
