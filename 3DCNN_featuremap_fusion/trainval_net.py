from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import pickle
import logging as Logger

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16_fusion import vgg16f
from model.faster_rcnn.vgg16 import vgg16, vgg16c

from model.faster_rcnn.resnet import resnet


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=5, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=2, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=2, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and display
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')

    # modal
    parser.add_argument('--modal', dest='modal',
                        help='which modal: rgb or ir or fusion',
                        default='rgb', type=str)

    # fusion model
    parser.add_argument('--fusion_mode', dest='fusion_mode',
                        help='how to fusion: early or half',
                        default='half', type=str)

    args = parser.parse_args()
    return args


def trainset_select(dataset):
    if dataset == "kaist_train-all02-rgb":
        imdb_name = "kaist_train-all02-rgb"
        imdbval_name = "kasit_test-all02-rgb"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif dataset == "kaist_train-all20-rgb":
        imdb_name = "kaist_train-all20-rgb"
        imdbval_name = "kasit_test-all20-rgb"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif dataset == "kaist_train-all02-fusion":
        imdb_name = "kaist_train-all02-fusion"
        imdbval_name = "kasit_test-all02-fusion"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif dataset == "kaist_train-all20-fusion":
        imdb_name = "kaist_train-all20-fusion"
        imdbval_name = "kasit_test-all20-fusion"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif dataset == "kaist_train-all02-thermal":
        imdb_name = "kaist_train-all02-thermal"
        imdbval_name = "kasit_test-all02-thermal"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif dataset == "kaist_train-all20-thermal":
        imdb_name = "kaist_train-all20-thermal"
        imdbval_name = "kasit_test-all20-thermal"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']



    elif dataset == "pascal_voc":
        imdb_name = "voc_2007_trainval"
        imdbval_name = "voc_2007_test"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif dataset == "pascal_voc_0712":
        imdb_name = "voc_2007_trainval+voc_2012_trainval"
        imdbval_name = "voc_2007_test"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif dataset == "coco":
        imdb_name = "coco_2014_train+coco_2014_valminusminival"
        imdbval_name = "coco_2014_minival"
        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif dataset == "imagenet":
        imdb_name = "imagenet_train"
        imdbval_name = "imagenet_val"
        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        imdb_name = "vg_150-50-50_minitrain"
        imdbval_name = "vg_150-50-50_minival"
        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    return imdb_name, imdbval_name, set_cfgs


def training_rgb():
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.cuda:
        fasterRCNN.cuda()

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        load_name = os.path.join(output_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(train_size / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                #print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                #      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                Log.info("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                Log.info("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                Log.info("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))


                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()

        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))

        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()


def training_fusion():
    # initilize the tensor holder here.
    im_data1 = torch.FloatTensor(1)
    im_data2 = torch.FloatTensor(1)

    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data1 = im_data1.cuda()
        im_data2 = im_data2.cuda()

        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data1 = Variable(im_data1)
    im_data2 = Variable(im_data2)

    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == 'vgg16f':
        fasterRCNN = vgg16f(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, fusion_mode=args.fusion_mode)

    elif args.net == 'vgg16c':
        fasterRCNN = vgg16c(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)

    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.cuda:
        fasterRCNN.cuda()

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        load_name = os.path.join(output_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(train_size / args.batch_size)


    # tfboard
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")
        #TODO
        #logger.add_graph(fasterRCNN, (im_data1, im_data2, im_info, gt_boxes, num_boxes))
        #TODO
    if args.net == 'vgg16f':
        for epoch in range(args.start_epoch, args.max_epochs + 1):
            # setting to train mode
            fasterRCNN.train()
            loss_temp = 0
            start = time.time()

            if epoch % (args.lr_decay_step + 1) == 0:
                adjust_learning_rate(optimizer, args.lr_decay_gamma)
                lr *= args.lr_decay_gamma

            data_iter = iter(dataloader)
            for step in range(iters_per_epoch):
                data = next(data_iter)

                im_data1.resize_(data[0].size()).copy_(data[0])
                im_data2.resize_(data[1].size()).copy_(data[1])
                im_info.resize_(data[2].size()).copy_(data[2])
                gt_boxes.resize_(data[3].size()).copy_(data[3])
                num_boxes.resize_(data[4].size()).copy_(data[4])

                fasterRCNN.zero_grad()
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = fasterRCNN(im_data1, im_data2, im_info, gt_boxes, num_boxes)

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                       + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                loss_temp += loss.item()

                # backward
                optimizer.zero_grad()
                loss.backward()
                if args.net == "vgg16":
                    clip_gradient(fasterRCNN, 10.)
                optimizer.step()

                if step % args.disp_interval == 0:
                    end = time.time()
                    if step > 0:
                        loss_temp /= (args.disp_interval + 1)

                    if args.mGPUs:
                        loss_rpn_cls = rpn_loss_cls.mean().item()
                        loss_rpn_box = rpn_loss_box.mean().item()
                        loss_rcnn_cls = RCNN_loss_cls.mean().item()
                        loss_rcnn_box = RCNN_loss_bbox.mean().item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt
                    else:
                        loss_rpn_cls = rpn_loss_cls.item()
                        loss_rpn_box = rpn_loss_box.item()
                        loss_rcnn_cls = RCNN_loss_cls.item()
                        loss_rcnn_box = RCNN_loss_bbox.item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt

                    Log.info("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                          % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                    Log.info("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                    Log.info("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                          % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                    if args.use_tfboard:
                        info = {
                            'loss': loss_temp,
                            'loss_rpn_cls': loss_rpn_cls,
                            'loss_rpn_box': loss_rpn_box,
                            'loss_rcnn_cls': loss_rcnn_cls,
                            'loss_rcnn_box': loss_rcnn_box
                        }
                        logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                           (epoch - 1) * iters_per_epoch + step)

                    loss_temp = 0
                    start = time.time()

            save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))

            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
            print('save model: {}'.format(save_name))
    elif args.net == 'vgg16c':
        for epoch in range(args.start_epoch, args.max_epochs + 1):
            # setting to train mode
            fasterRCNN.train()
            loss_temp = 0
            start = time.time()

            if epoch % (args.lr_decay_step + 1) == 0:
                adjust_learning_rate(optimizer, args.lr_decay_gamma)
                lr *= args.lr_decay_gamma

            data_iter = iter(dataloader)
            for step in range(iters_per_epoch):
                data = next(data_iter)

                im_data1.resize_(data[0].size()).copy_(data[0])
                im_data2.resize_(data[1].size()).copy_(data[1])
                im_info.resize_(data[2].size()).copy_(data[2])
                gt_boxes.resize_(data[3].size()).copy_(data[3])
                num_boxes.resize_(data[4].size()).copy_(data[4])

                fasterRCNN.zero_grad()
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls1, rpn_loss_box1, \
                rpn_loss_cls2, rpn_loss_box2, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = fasterRCNN(im_data1, im_data2, im_info, gt_boxes, num_boxes)

                loss = rpn_loss_cls1.mean() + rpn_loss_box1.mean() \
                        + rpn_loss_cls2.mean() + rpn_loss_box2.mean() \
                        + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                loss_temp += loss.item()

                # backward
                optimizer.zero_grad()
                loss.backward()
                if args.net == "vgg16":
                    clip_gradient(fasterRCNN, 10.)
                optimizer.step()

                if step % args.disp_interval == 0:
                    end = time.time()
                    if step > 0:
                        loss_temp /= (args.disp_interval + 1)

                    if args.mGPUs:
                        loss_rpn_cls1 = rpn_loss_cls1.mean().item()
                        loss_rpn_cls2 = rpn_loss_cls2.mean().item()
                        loss_rpn_box1 = rpn_loss_box1.mean().item()
                        loss_rpn_box2 = rpn_loss_box2.mean().item()
                        loss_rcnn_cls = RCNN_loss_cls.mean().item()
                        loss_rcnn_box = RCNN_loss_bbox.mean().item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt
                    else:
                        loss_rpn_cls1 = rpn_loss_cls1.item()
                        loss_rpn_cls2 = rpn_loss_cls2.item()
                        loss_rpn_box1 = rpn_loss_box1.item()
                        loss_rpn_box2 = rpn_loss_box2.item()
                        loss_rcnn_cls = RCNN_loss_cls.item()
                        loss_rcnn_box = RCNN_loss_bbox.item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt

                    Log.info("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                             % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                    Log.info("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                    Log.info("\t\t\trpn_cls1: %.4f,rpn_cls2: %.4f, rpn_box1: %.4f, rpn_box2: %.4f,rcnn_cls: %.4f, rcnn_box %.4f" \
                             % (loss_rpn_cls1, loss_rpn_cls2,loss_rpn_box1,loss_rpn_box2, loss_rcnn_cls, loss_rcnn_box))
                    if args.use_tfboard:
                        info = {
                            'loss': loss_temp,
                            'loss_rpn_cls1': loss_rpn_cls1,
                            'loss_rpn_cls2': loss_rpn_cls2,
                            'loss_rpn_box1': loss_rpn_box1,
                            'loss_rpn_box2': loss_rpn_box2,
                            'loss_rcnn_cls': loss_rcnn_cls,
                            'loss_rcnn_box': loss_rcnn_box
                        }
                        logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                           (epoch - 1) * iters_per_epoch + step)

                    loss_temp = 0
                    start = time.time()

            save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))

            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
            print('save model: {}'.format(save_name))
    #logger.add_graph(fasterRCNN, (im_data1, im_data2, im_info, gt_boxes, num_boxes))

    if args.use_tfboard:
        logger.close()


def training_thermal():
    pass

class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

if __name__ == '__main__':

    args = parse_args()
    if args.modal == "rgb":
        cfg.INPUT_IMAGE = "rgb"
    if args.modal == "thermal":
        cfg.INPUT_IMAGE = "thermal"
    if args.modal == "fusion":
        cfg.INPUT_IMAGE = "fusion"

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    ###################### merge some parameters
    args.imdb_name, args.imdbval_name, args.set_cfgs = trainset_select(args.dataset)
    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda


    ##################### output path
    save_name = 'faster_rcnn_10' +'_train_log'
    output_path = get_output_dir(args.imdb_name, save_name)
    # log output
    log_output = output_path + '/trian.log'
    Log = Logger.getLogger("train.log")
    ##################### model path
    output_dir = args.save_dir + '/' + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ###################### dump parameters
    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)
    with open(output_path + '/parameters.pkl', 'wb') as f:
        pickle.dump(cfg, f)
    Log.info(cfg)

    ###################### dump all the args
    Log.info('Called with args:')
    Log.info(args)
    with open(output_path + '/args.pkl', 'wb') as f:
        pickle.dump(args, f)
    Log.info(args)



    ###################### load data
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    print('{:d} roidb entries'.format(len(roidb)))



    sampler_batch = sampler(train_size, args.batch_size)
    # initilize dataset
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)


    # 选择模式, 开始训练
    if args.modal == "rgb":
        training_rgb()
    if args.modal == "thermal":
        training_thermal()
    if args.modal == "fusion":
        training_fusion()
