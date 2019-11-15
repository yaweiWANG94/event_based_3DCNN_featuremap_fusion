import torch
import torch.nn as nn


from model.faster_rcnn.vgg_myself import make_vgg16,make_vgg16_2C4
from model.faster_rcnn.vgg16_fusion import vgg16f
from model.faster_rcnn.vgg16 import vgg16

classes = ('__background__',  # always index 0
                 'pedestrian')
fusion_mode = 'half'
fasterRCNN = vgg16f(classes, pretrained=False, fusion_mode=fusion_mode, model_path='../../../models/faster_rcnn_1_5_5533.pth')
print(fasterRCNN)
fasterRCNN.create_architecture()
print(fasterRCNN)




fasterRCNN = vgg16(classes, pretrained=True, model_path='../../../models/faster_rcnn_1_5_5533.pth')
#fasterRCNN.create_architecture()