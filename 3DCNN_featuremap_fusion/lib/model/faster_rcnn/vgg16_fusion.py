# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn_fusion import _fasterRCNNF
from model.faster_rcnn.vgg_myself import make_vgg16, make_vgg16_2C4, make_vgg16_C5

import pdb

class vgg16f(_fasterRCNNF):
  def __init__(self, classes, pretrained=False, class_agnostic=False, modal="fusion",
               fusion_mode='half',
               #model_path='models/faster_rcnn_1_5_5533.pth'
               model_path="data/pretrained_model/vgg16_caffe.pth" ):
    self.model_path = model_path
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    #单双通道模态
    self.modal = modal
    self.fusion_mode = fusion_mode


    _fasterRCNNF.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    #vgg = models.vgg16()
    #使用我自己的vgg16
    vgg = make_vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])


    #early fusion
    self.NIN = self.nin(in_channel=6,out_channel=3)

    #self.early_fusion = self.NIN
    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

  def _init_modules_halfway(self):
      # vgg = models.vgg16()
      # 使用我自己的vgg16

      #conv4前
      vgg = make_vgg16_2C4()

      #1*1卷积
      self.NIN = self.nin(in_channel=1, out_channel=1)

      #融合后的部分
      vgg_fusion = make_vgg16_C5()

      if self.pretrained:
          print("Loading pretrained weights from %s" % (self.model_path))
          state_dict = torch.load(self.model_path)
          vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})
          #vgg_fusion.load_state_dict({k: v for k, v in state_dict.items() if k in vgg_fusion.state_dict()})




      # not using the last maxpool layer
      # half fusion 前半部分,融合前
      self.RCNN_base_half = nn.Sequential(*list(vgg.features._modules.values()))

      #half fusion 后半部分,融合后
      self.RCNN_base_fusion = nn.Sequential(*list(vgg_fusion.features._modules.values()))

      #分类部分，暂不知道用在哪
      vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

      # Fix the layers before conv3:
      for layer in range(10):
          #print(self.RCNN_base_half[layer])
          for p in self.RCNN_base_half[layer].parameters(): p.requires_grad = False

      # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

      self.RCNN_top = vgg.classifier

      # not using the last maxpool layer
      self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

      if self.class_agnostic:
          self.RCNN_bbox_pred = nn.Linear(4096, 4)
      else:
          self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

  def nin(self,in_channel, out_channel):

      con3d1 = nn.Conv3d(in_channel,1,kernel_size=(3,3,3),stride=1,padding=1)
      con3 = nn.Conv3d(1,1,kernel_size=(3,3,3),stride=(2,1,1),padding=1) 
      layers = [con3d1, con3, nn.ReLU(inplace=True)]
      #con2d = nn.Conv2d(in_channel, out_channel, 1, 1)
      #layers = [con2d, nn.ReLU(inplace=True)]
      return nn.Sequential(*layers)


