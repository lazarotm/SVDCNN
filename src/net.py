import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalBlock(nn.Module):

    def __init__(self, input_dim=128, n_filters=256, kernel_size=3, padding=1, stride=1, shortcut=False, downsample=None):
        super(ConvolutionalBlock, self).__init__()
        
        self.downsample = downsample
        self.shortcut = shortcut        

        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=kernel_size, padding=padding, stride=stride, groups=input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(input_dim, n_filters, kernel_size=1, padding=0, stride=1)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, stride=stride, groups=n_filters)
        self.bn3 = nn.BatchNorm1d(n_filters)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(n_filters, n_filters, kernel_size=1, padding=0, stride=1)
        self.bn4 = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()

    def forward(self, x):        

        residual = x

        out = self.conv1(x)        
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)        
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)        
        out = self.bn3(out)
        out = self.relu3(out)

        out = self.conv4(out)        
        out = self.bn4(out)        

        if self.shortcut:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        out = self.relu(out)

        return out

class SVDCNN(nn.Module):

    def __init__(self, n_classes=2, num_embedding=141, embedding_dim=16, depth=9, shortcut=False):
        super(SVDCNN, self).__init__()

        layers = []
        fc_layers = []

        self.embed = nn.Embedding(num_embedding, embedding_dim, padding_idx=0, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
        layers.append(nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1))

        if depth == 9:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 1, 1, 1, 1
        elif depth == 17:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 2, 2, 2, 2
        elif depth == 29:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 5, 5, 2, 2
        elif depth == 49:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 8, 8, 5, 3

        layers.append(ConvolutionalBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
        for _ in range(n_conv_block_64-1):
            layers.append(ConvolutionalBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1)) # l = initial length / 2

        ds = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(128))
        layers.append(ConvolutionalBlock(input_dim=64, n_filters=128, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_128-1):
            layers.append(ConvolutionalBlock(input_dim=128, n_filters=128, kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1)) # l = initial length / 4

        ds = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(256))
        layers.append(ConvolutionalBlock(input_dim=128, n_filters=256, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_256 - 1):
            layers.append(ConvolutionalBlock(input_dim=256, n_filters=256, kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(512))
        layers.append(ConvolutionalBlock(input_dim=256, n_filters=512, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_512 - 1):
            layers.append(ConvolutionalBlock(input_dim=512, n_filters=512, kernel_size=3, padding=1, shortcut=shortcut))
        
        layers.append(nn.AdaptiveAvgPool1d(8))
        fc_layers.extend([nn.Linear(8*512, n_classes)])

        self.layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(*fc_layers)

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        out = self.embed(x)
        out = out.transpose(1, 2)

        out = self.layers(out)

        out = out.view(out.size(0), -1)

        out = self.fc_layers(out)

        return out