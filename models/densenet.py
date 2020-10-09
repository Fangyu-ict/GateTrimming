import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0, config=None):
        super(BasicBlock, self).__init__()
        if config == None:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.droprate = dropRate
        else:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0, config=None):
        super(BottleneckBlock, self).__init__()
        if config==None:
            inter_planes = out_planes * 4
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(inter_planes)
            self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.droprate = dropRate
        else:
            inter_planes = out_planes * 4
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, config[0], kernel_size=1, stride=1,
                                   padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(config[0])
            self.conv2 = nn.Conv2d(config[0], config[1], kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0, config=None):
        super(TransitionBlock, self).__init__()
        if config==None:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)
            self.droprate = dropRate
        else:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)
            self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0, config=None):
        super(DenseBlock, self).__init__()
        if config==None:
            self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate, config)
        else:
            self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate,config)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate, config):
        if config == None:
            layers = []
            for i in range(nb_layers):
                layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        else:
            layers = []
            # layers.append(block(in_planes, growth_rate, dropRate, config[0:2]))
            # layers.append(block(in_planes +config[1], growth_rate, dropRate, config[2:4]))
            # layers.append(block(in_planes +config[1]+config[3], growth_rate, dropRate, config[4:6]))
            # layers.append(block(in_planes +config[1]+config[3]+config[5], growth_rate, dropRate, config[6:8]))
            # layers.append(block(in_planes +config[1]+config[3]+config[5]+config[7], growth_rate, dropRate, config[8:10]))
            # layers.append(block(in_planes +config[1]+config[3]+config[5]+config[7]+config[9], growth_rate, dropRate, config[10:12]))

            sum_planes = in_planes
            for i in range(nb_layers):
                layers.append(block(sum_planes, config[i], dropRate, config))
                sum_planes = sum_planes + config[i]



        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0, config=None):
        super(DenseNet3, self).__init__()
        if config == None:
            in_planes = 2 * growth_rate
            n = (depth - 4) / 3
            if bottleneck == True:
                n = n/2
                block = BottleneckBlock
            else:
                block = BasicBlock
            n = int(n)
            # 1st conv before any dense block
            self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            # 1st block
            self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
            in_planes = int(in_planes+n*growth_rate)
            self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
            in_planes = int(math.floor(in_planes*reduction))
            # 2nd block
            self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
            in_planes = int(in_planes+n*growth_rate)
            self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
            in_planes = int(math.floor(in_planes*reduction))
            # 3rd block
            self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
            in_planes = int(in_planes+n*growth_rate)
            # global average pooling and classifier
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(in_planes, num_classes)
            self.in_planes = in_planes
        else:
            in_planes = config[0]
            n = (depth - 4) / 3
            if bottleneck == True:
                n = n / 2
                block = BottleneckBlock
            else:
                block = BasicBlock
            n = int(n)
            # 1st conv before any dense block
            self.conv1 = nn.Conv2d(3, config[0], kernel_size=3, stride=1,
                                   padding=1, bias=False)
            # 1st block
            self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate, config[1:13])
            for i in range(1,13):
                in_planes += config[i]

            # in_planes = config[0]+config[2]+config[4]+config[6]+config[8]+config[10]+config[12]
            self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate,config=config)
            in_planes = in_planes
            # 2nd block
            self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate, config[13:25])
            for i in range(13,25):
                in_planes += config[i]

            # in_planes = in_planes+config[15]+config[17]+config[19]+config[21]+config[23]+config[25]
            self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate,config=config)
            in_planes = in_planes
            # 3rd block
            self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate, config[25:])
            for i in range(25,37):
                in_planes += config[i]

            # in_planes = in_planes+config[28]+config[30]+config[32]+config[34]+config[36]+config[38]
            # global average pooling and classifier
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(in_planes, num_classes)
            self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)

def densenet40(num_classes=10):
    model = DenseNet3(40, num_classes, 12, reduction=1.0, bottleneck=False, config=None)
    return model

def densenet40_reconstrcut(num_classes=10, config=None):
    model = DenseNet3(40, num_classes, 12, reduction=1.0, bottleneck=False, config=config)
    return model

if __name__ == '__main__':
    model = DenseNet3(40, 10, 12, reduction=1.0, bottleneck=False)
    # a = torch.rand(1, 3, 32, 32)
    print(model)
    from ptflops import get_model_complexity_info
    flops_o, params_o = get_model_complexity_info(model, (32, 32), as_strings=True, print_per_layer_stat=True)
    print('Origin:')
    print('Flops:  ' + flops_o)
    print('Params: ' + params_o)
    # x = torch.randn(2, 3, 32, 32)
    # y = model(x)
    # print(y.size())
    # with SummaryWriter(comment='dense40') as w:
    #     w.add_graph(model, (a,))