import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from utils import *
import models
import torch.nn.functional as F

import os
import time
import copy
import random
import argparse

import math
import csv
from ptflops import get_model_complexity_info

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', default='../data', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet20', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.01, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[1, 20, 40], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[1, 0.2, 0.2], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Acceleration
parser.add_argument('--gpus', type=str, default='0', help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--pre_train', default=False, type=bool, help='whether to use pre_train model')
parser.add_argument('--baseline_path', default='./baseline/', type=str, help='Path to teacherModel')
# random seed
parser.add_argument('--manualSeed', type=int, default=888, help='manual seed')
#compress rate
parser.add_argument('--pruning_rate', type=float, default=0.3, help='compress rate of model')
parser.add_argument('--epoch_prune', type=int, default=1,  help='compress layer of model')
# Checkpoints
parser.add_argument('--resume', default=False, type=bool, help='whether to resume (default: False)')#, metavar='PATH',
parser.add_argument('--retrain', default=False, type=bool, help='whether to retrain the model (default: False)')
parser.add_argument('--checkpoints', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--note', metavar='SAVE NAME PATH', type=str, default='note', help='Folder to save checkpoints and log.')

args = parser.parse_args()
model_name = args.arch+'_cr'+str(args.pruning_rate)+'_pre'+str(args.pre_train)+'_'+args.note

random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)
torch.backends.cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
cudnn.benchmark = True

def train_and_evaluate(model, dataloaders, criterion, optimizer, start_epoch=0, num_epochs=1,
                log=None):
    best_epoch = 0
    best_epoch_prate = 0.
    since = time.time()

    writer = SummaryWriter(comment=model_name)

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.

    m = Mask(model, args.pruning_rate)

    for epoch in range(start_epoch, num_epochs):

        current_learning_rate = adjust_rate(optimizer, epoch, args.gammas, args.schedule)

        print_log('[Epoch]: [{}/{}] current_learning_rate: [{:.6f}] [{}]'.format(
            epoch, num_epochs - 1,current_learning_rate, model_name), log)

        train_and_calIG(model, dataloaders, criterion, optimizer, m, epoch, since, writer, log)

        if epoch % args.epoch_prune == 0 or epoch == args.epochs - 1:

            m.model = model

            p_rate = m.if_zero('unmasked', epoch, writer)
            if epoch == 0:
                m.core_function()
                m.do_mask()
                p_rate = m.if_zero('masked', epoch, writer)
                model = m.model
                model = model.cuda()
                m.gradW_avg = {} #clear the grad_avg

                if epoch == 0:
                    save_arch_and_cal_FLOPs(model, m)

            eval_acc = evaluate(model, dataloaders, criterion, since, log, epoch, writer, state='masked')

            if eval_acc > best_acc:

                best_acc = eval_acc
                best_model = copy.deepcopy(model.state_dict())

                best_epoch = epoch
                best_epoch_prate = p_rate


        print_log('best_acc: [{:.4f}] best_err: [{:.4f}] best_epoch: [{:d}] p_rate: [{:.4f}]'.format(best_acc, 1.0 - best_acc,
                                                                                  best_epoch, best_epoch_prate), log)

        save_checkpoint({'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_model_state_dict': best_model,
                'optimizer': optimizer.state_dict(),
                # 'accuracy':eval_acc,
                'best_acc':best_acc},model_name)

    print()

    time_elapsed = time.time() - since

    finish_msg = 'Model name: '+model_name +'\n'\
                 +str(args)+'\n'\
                 +'Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60) + '\n'\
                 +'Best val Acc: {:4f} best_err: {:4f} best_epoch: {:d} p_rate: {:.4f}'.format(best_acc, 1.0 -best_acc,
                                                                              best_epoch, best_epoch_prate)
    print_log(finish_msg, log)


    # load best model weights
    model.load_state_dict(best_model)

    torch.save(model.module.state_dict(), 'checkpoints/' + model_name + "/parameter.pkl")

    torch.cuda.empty_cache()
    return model

def train_and_calIG(model, dataloaders, criterion, optimizer,m , epoch,since,writer,log):

    model.train()  # Set model to training mode

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    iter_per_epoch = dataloaders['train'].__len__()
    process_bar = ShowProcess(iter_per_epoch)

    # Iterate over data.
    for iter, (inputs, labels) in enumerate(dataloaders['train']):

        inputs = inputs.cuda()

        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(True):
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))


            writer.add_scalar('scalar/train' + '/loss', loss.item(), iter_per_epoch * epoch + iter)
            writer.add_scalar('scalar/train' + '/acc', prec1, iter_per_epoch * epoch + iter)

            # calculate first derivatie IG
            if iter ==0 and epoch == 0:
                loss_ig =  -F.softmax(outputs, dim=1).mul(F.log_softmax(outputs, dim=1)).mean()
                loss_ig.backward(retain_graph=True)

                gradW = {}
                for index, item in enumerate(model.parameters()):
                    if index in m.mask_bn_index:
                        gradW[index] = item.grad * item.data
                    if index in m.mask_bn_index_b:
                        gradW[index-1] += item.grad * item.data
                        if iter == 0:
                            m.gradW_avg[index-1] = 0.

                        m.gradW_avg[index-1] = (m.gradW_avg[index-1] * iter + gradW[index-1]) / (iter + 1)

            if epoch != 0:
                optimizer.zero_grad()
                # backward the kd loss
                loss.backward()
                # optimize
                optimizer.step()

        time_elapsed = time.time() - since
        process_bar.show_process(loss, prec1, prec5, time_elapsed // 60, time_elapsed % 60)

    print_log('{} Loss: [{:.4f}] Prec1: [{:.4f}] Prec5: [{:.4f}] Err: [{:.4f}] Time: {}'.format(
        'train', losses.avg, top1.avg, top5.avg, 1.0 - top1.avg, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())), log)

    return losses.avg, top1.avg, top5.avg

def evaluate(model, dataloaders, criterion, since, log, epoch=0, writer=None, state=''):
    model.eval()  # Set model to evaluate mode

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    iter_per_epoch = dataloaders['val'].__len__()
    process_bar = ShowProcess(iter_per_epoch)
    # Iterate over data.
    for iter, (inputs, labels) in enumerate(dataloaders['val']):

        inputs = inputs.cuda()

        labels = labels.cuda()

        with torch.no_grad():
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)

            # loss = criterion(outputs, labels)
            loss = 0.0  # force validation loss to zero to reduce computation time

            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss, inputs.size(0))#loss.item()
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            if writer != None:
                writer.add_scalar('scalar/' + state +  '/acc', prec1, iter_per_epoch * epoch + iter)

        time_elapsed = time.time() - since
        process_bar.show_process(loss, prec1, prec5, time_elapsed // 60, time_elapsed % 60)

    print_log('{} Loss: [{:.4f}] Prec1: [{:.4f}] Prec5: [{:.4f}] Err: [{:.4f}] Time: {}'.format(
        state+'-val', losses.avg, top1.avg, top5.avg, 1.0 - top1.avg, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())), log)

    return top1.avg


def save_arch_and_cal_FLOPs(model, m):

    if 'vgg' in args.arch:
        config = vggReconstrcut(model, m)

        with open('./checkpoints/' + model_name + '/config.csv', 'w') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerows([config])

        compact_model = models.vgg_reconstruct(10, config, fc=config[-2])
    elif 'resnet' in args.arch:
        config = resnetReconstrcut(model, m)
        depth = int(args.arch[6:])

        with open('./checkpoints/' + model_name + '/config.csv', 'w') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerows([config])
        compact_model = models.resnet_reconstruct(10, depth, config)
    elif 'densenet' in args.arch:
        config = resnetReconstrcut(model, m)
        with open('./checkpoints/' + model_name + '/config.csv', 'w') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerows([config])

        compact_model = models.densenet40_reconstrcut(10, config)

    compact_model.cuda()

    # Count the number of FLOPs
    compact_model.cpu()

    flops_p, params_p = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                                  print_per_layer_stat=True)

    flops_p, params_p = get_model_complexity_info(compact_model, (3, 32, 32), as_strings=True,
                                                  print_per_layer_stat=True)
    print('Pruned:')
    print('Flops:  ' + flops_p)
    print('Params: ' + params_p)

def vggReconstrcut(sparse_model,m):
    config = models.cfg[args.arch]
    i = 0
    for index, item in enumerate(sparse_model.named_parameters()):
        name = item[0]
        m_item = item[1]

        if index in m.mask_index:

            temp = m_item.data.view(m.model_length[index])
            temp = temp.cpu().numpy()

            nonzerofilter = np.count_nonzero(temp)

            for index2 in range(0, len(m.model_size[index])):
                if index2 == 0:
                    layerfilter = m.model_size[index][0]
                else:
                    nonzerofilter /= m.model_size[index][index2]

            nonzerofilter = math.ceil(nonzerofilter)
            print(name + ': %d / %d' % (nonzerofilter, layerfilter))
            if config[i] == 'M':
                i += 1
            config[i] = int(nonzerofilter)
            i += 1

    return config

def resnetReconstrcut(sparse_model,m):
    config = []
    # i = 0
    for index, item in enumerate(sparse_model.named_parameters()):
        name = item[0]
        m_item = item[1]

        if index in m.mask_index:

            temp = m_item.data.view(m.model_length[index])
            temp = temp.cpu().numpy()

            nonzerofilter = np.count_nonzero(temp)

            for index2 in range(0, len(m.model_size[index])):
                if index2 == 0:
                    layerfilter = m.model_size[index][0]
                else:
                    nonzerofilter /= m.model_size[index][index2]

            nonzerofilter = math.ceil(nonzerofilter)
            print(name + ': %d / %d' % (nonzerofilter, layerfilter))

            config.append(nonzerofilter)

    return config


def main():

    #############################  #check if the directions exist ############################
    if not os.path.exists('./checkpoints/'):
        os.mkdir('./checkpoints/')
    if not os.path.exists('./logs/'):
        os.mkdir('./logs/')
    if not os.path.exists('./checkpoints/'+model_name):
        os.mkdir('./checkpoints/'+model_name)

    t = time.strftime("%Y_%m_%d", time.localtime())

    log = open(os.path.join('./logs', t+'_{}.txt'.format(model_name)), 'a')

    #############################  then loading the data #####################################

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    val_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                                download=False, transform=train_transform)
        valset = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                              download=False, transform=val_transform)
        classnumber = 10
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        valset = torchvision.datasets.CIFAR100(args.data_path, train=False, transform=val_transform, download=True)
        classnumber = 100

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers)


    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers)

    dataloaders = {'train':trainloader, 'val':valloader}

    print_log('Model name: '+model_name, log)
    print_log(args, log)
    print_log("-" * 60, log)

    #################################### build the model ####################################

    model = models.__dict__[args.arch](classnumber)
    # print("=> network :\n {}".format(model))

    model.cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)

    if args.pre_train == True:
        checkpoint = torch.load(args.baseline_path+str(args.arch)+'_baseline.pth.tar')

        model.load_state_dict(checkpoint['best_model_state_dict'])
        model.cuda()

    model = torch.nn.DataParallel(model).cuda()

    #################################### resume or train ####################################
    if args.resume:
        checkpoint = torch.load('./checkpoints/' + args.checkpoints)
        epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print_log("epoch: {}".format(epoch), log)
        print_log("best_acc: {}".format(best_acc), log)

        if args.retrain:
            #############################  resume to train ####################################
            best_model = copy.deepcopy(model.load_state_dict(checkpoint['best_model_state_dict']))
            optimizer.load_state_dict(checkpoint['optimizer'])

            model = train_and_evaluate(model, dataloaders, criterion, optimizer, start_epoch=epoch+1,
                              num_epochs=args.epochs,best_acc=best_acc,best_model=best_model,log=log)
    else:
        model = train_and_evaluate(model, dataloaders, criterion, optimizer,
                            num_epochs=args.epochs, log=log)



class Mask:
    def __init__(self, model,pruning_rate=0):
        self.model_size = {}
        self.model_length = {}
        self.mat = {}
        self.gradW_avg = {}

        self.mask_index = []
        self.mask_bn_index = []
        self.mask_bn_index_b = []
        self.skip_list = []
        self.constrain_list = []

        self.lookup_table = []

        self.base_time=0

        self.pruning_rate = pruning_rate

        self.model = model
        self.filter_number = 0
        self.layer_end = 0
        self.hessian = None

        self.init_length()

    def init_length(self):

        if args.arch == 'resnet20':
            self.layer_end = 54
        elif args.arch == 'resnet32':
            self.layer_end = 90
        elif args.arch == 'resnet56':
            self.layer_end = 162
        elif args.arch == 'resnet110':
            self.layer_end = 324
        elif args.arch == 'vgg16':
            self.layer_end = 36
        elif args.arch == 'densenet40':
            self.layer_end = 114
            self.skip_list = [39, 78]
        elif args.arch == 'vgg19':
            self.layer_end = 45

        last_index = self.layer_end + 1

        self.mask_index = [x for x in range(0, last_index, 3)]
        self.mask_bn_index = [(x+1) for x in range(0, last_index, 3)]
        self.mask_bn_index_b = [(x+2) for x in range(0, last_index, 3)]

        if 'resnet' in args.arch:
            self.constrain_list = [(x+1) for x in range(0, last_index, 6)]
            self.skip_list = []

            for x in self.skip_list:
                self.mask_index.remove(x)
                print(self.mask_index)
        if 'densenet' in args.arch:
            for x in self.skip_list:
                self.mask_index.remove(x)


        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            if index1 in self.mask_index:
                self.filter_number += self.model_size[index1][0]
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
            elif index in self.skip_list:
                pass
            elif index < self.layer_end + 3:
                a = item.data
                b = a * self.mat[index]
                item.data = b
        print("mask Done")

    def if_zero(self, state, epoch, writer):

        filterPruned = 0
        for index, item in enumerate(self.model.named_parameters()):
            name = item[0]
            param = item[1]

            if index < self.layer_end + 3:#in self.mask_index:
                temp = param.data.view(self.model_length[index])
                temp = temp.cpu().numpy()

                if index in self.mask_index:# only record the number of pruned convlayer
                    nonzerofilter = np.count_nonzero(temp)

                    for index2 in range(0, len(self.model_size[index])):
                        if index2 == 0:
                            layerfilter = self.model_size[index][0]
                        else:
                            nonzerofilter /= self.model_size[index][index2]

                    nonzerofilter = math.ceil(nonzerofilter)

                    # if state in 'masked':
                    print(name + ': %d / %d' % (nonzerofilter, layerfilter))

                    writer.add_scalar(state+'/_'+name, nonzerofilter, epoch)
                    filterPruned += layerfilter - nonzerofilter

        rataPrune = filterPruned / self.filter_number
        print(state+' :rataPrune is [%.4f]' % rataPrune)
        writer.add_scalar('scalar/'+state + '/p_rate', rataPrune, epoch)


        return rataPrune

    def core_function(self):
        normCoordinate = self.get_normCoordinate()

        filter_selected = normCoordinate.argsort(0)[:,2]#[:filter_pruned_num]

        self.init_mask(filter_selected, normCoordinate)

    def get_normCoordinate(self):
        gradW = {}
        normCoordinate = []

        #cal the gradW matrix of all weight
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_bn_index:

                gradW[index] = self.gradW_avg[index]#item.grad *item.data

                # if len(gradW[index].size()) == 4:
                gradW_vec = gradW[index].view(gradW[index].size()[0], -1)
                norm2 = torch.norm(gradW_vec, 2, 1)
                norm2_np = norm2.cpu().numpy().reshape(-1,1)

                layerIndex = np.array([index for x in range(norm2_np.shape[0])]).reshape(-1,1)
                filterIndex = np.array([x for x in range(norm2_np.shape[0])]).reshape(-1,1)
                normIndex = np.hstack((layerIndex, filterIndex))
                normIndex = np.hstack((normIndex, norm2_np))

                if index == self.mask_bn_index[0]:
                    normCoordinate = normIndex
                else:
                    normCoordinate = np.vstack((normCoordinate, normIndex))

        return normCoordinate

    def set_mask(self, layer, filter, prunedFilters):

        kernel_length = self.model_length[layer-1] // self.model_size[layer-1][0]

        self.mat[layer - 1][filter * kernel_length: (filter + 1) * kernel_length] = 0.
        self.mat[layer][filter] = 0.
        self.mat[layer + 1][filter] = 0.
        prunedFilters += 1

        return prunedFilters

    def prune_2nd_conv(self, filter, mat_2ndconv, prunedFilters):

        sum_of_mask = 0

        for i in self.constrain_list:
            if i <= self.layer_end//3:
                sum_of_mask += mat_2ndconv[i][filter % 16]
            elif self.layer_end//3 < i and i <= self.layer_end//3 *2:
                sum_of_mask += mat_2ndconv[i][filter % 16] + mat_2ndconv[i][filter % 16 + 16]
            elif self.layer_end // 3 * 2 < i and i <= self.layer_end:
                sum_of_mask += mat_2ndconv[i][filter % 16] + mat_2ndconv[i][filter % 16 + 16] + \
                               mat_2ndconv[i][filter % 32] + mat_2ndconv[i][filter % 16 + 48]

        if int(sum_of_mask) == 0:

            for i in self.constrain_list:
                if i <= self.layer_end // 3:
                    prunedFilters = self.set_mask(i, filter % 16, prunedFilters)
                elif self.layer_end // 3 < i and i <= self.layer_end // 3 *2:
                    prunedFilters = self.set_mask(i, filter % 16, prunedFilters)
                    prunedFilters = self.set_mask(i, filter % 16 + 16, prunedFilters)
                if self.layer_end // 3 *2 < i and i <= self.layer_end:
                    prunedFilters = self.set_mask(i, filter % 16, prunedFilters)
                    prunedFilters = self.set_mask(i, filter % 16 + 16, prunedFilters)
                    prunedFilters = self.set_mask(i, filter % 16 + 32, prunedFilters)
                    prunedFilters = self.set_mask(i, filter % 16 + 48, prunedFilters)
        return prunedFilters

    def init_mask(self, filter_selected, normCoordinate):
        mat_2ndconv = {}
        prunedFilters = 0

        for index, item in enumerate(self.model.parameters()):
            if index < self.layer_end + 3:

                self.mat[index] = np.ones(self.model_length[index])


                if index in self.constrain_list:
                    mat_2ndconv[index] = np.ones(self.model_size[index][0])


        for arr in normCoordinate[filter_selected]:
            layer = int(arr[0])
            filter = int(arr[1])
            kernel_length = self.model_length[layer-1]//self.model_size[layer-1][0]

            if self.mat[layer-1].sum() != kernel_length:
                if layer in self.constrain_list:
                    # mat_2ndconv[layer][filter] = 0
                    # prunedFilters = self.prune_2nd_conv(filter, mat_2ndconv, prunedFilters)
                    pass
                else:
                    prunedFilters = self.set_mask(layer, filter, prunedFilters)


            if prunedFilters/self.filter_number > self.pruning_rate:
                break

        for index, item in enumerate(self.model.parameters()):
            if index < self.layer_end + 3:
                self.mat[index] = torch.FloatTensor(self.mat[index]).cuda()


def adjust_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if epoch >= step:
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

if __name__ == '__main__':
    main()
