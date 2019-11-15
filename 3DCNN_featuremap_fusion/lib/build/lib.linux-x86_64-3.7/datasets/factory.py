# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets
#from datasets.caltech import caltech
from datasets.kaist_rgb import kaist_rgb
#from datasets.kaist_thermal import kaist_thermal
from datasets.kaist_fusion import kaist_fusion
from datasets.pascal_voc import pascal_voc
#from datasets.coco import coco

import numpy as np

# set up caltech
imageset = 'test';
name = 'caltech_{}'.format(imageset)
__sets[name] = (lambda imageset = imageset: caltech('test'))

imageset = 'train04';
name = 'caltech_{}'.format(imageset)
__sets[name] = (lambda imageset = imageset: caltech('train04'))

#kaist

#rgb
imageset = 'test-all20-rgb';
name = 'kaist_{}'.format(imageset)
__sets[name] = (lambda imageset = imageset: kaist_rgb('test-all20-rgb'))

imageset = 'train-all02-rgb';
name = 'kaist_{}'.format(imageset)
__sets[name] = (lambda imageset = imageset: kaist_rgb('train-all02-rgb'))

imageset = 'train-all20-rgb';
name = 'kaist_{}'.format(imageset)
__sets[name] = (lambda imageset = imageset: kaist_rgb('train-all20-rgb'))

#thermal
imageset = 'test-all20-thermal';
name = 'kaist_{}'.format(imageset)
__sets[name] = (lambda imageset = imageset: kaist_thermal('test-all20-thermal'))

imageset = 'train-all02-thermal';
name = 'kaist_{}'.format(imageset)
__sets[name] = (lambda imageset = imageset: kaist_thermal('train-all02-thermal'))

imageset = 'train-all20-thermal';
name = 'kaist_{}'.format(imageset)
__sets[name] = (lambda imageset = imageset: kaist_thermal('train-all20-thermal'))

#fusion
imageset = 'test-all20-fusion';
name = 'kaist_{}'.format(imageset)
__sets[name] = (lambda imageset = imageset: kaist_fusion('test-all20-fusion'))

imageset = 'train-all02-fusion';
name = 'kaist_{}'.format(imageset)
__sets[name] = (lambda imageset = imageset: kaist_fusion('train-all02-fusion'))

imageset = 'train-all20-fusion';
name = 'kaist_{}'.format(imageset)
__sets[name] = (lambda imageset = imageset: kaist_fusion('train-all20-fusion'))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    #if not __sets.has_key(name):
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
