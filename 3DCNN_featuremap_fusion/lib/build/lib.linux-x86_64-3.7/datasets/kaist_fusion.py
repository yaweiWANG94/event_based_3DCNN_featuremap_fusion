# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

#import datasets.kaist_fusion
import os
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
#import utils.cython_bbox
import pickle
import subprocess
from model.utils.config import cfg

class kaist_fusion(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, image_set)  # image_set: train04 or test
        self._image_set = image_set
        self._devkit_path = self._get_default_path()
        self._data_path = os.path.join(self._devkit_path)
        self._classes = ('__background__', # always index 0
                         'pedestrian')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        #print(self._image_index[i])
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        #print(index)
        image_path_1 = os.path.join(self._data_path, self._image_set, 'images',
                                  index[:-6] + 'visible' + index[-7:] + self._image_ext)
        #print(image_path_1)
        image_path_2 = os.path.join(self._data_path, self._image_set, 'images',
				  index[:-6] + 'lwir' + index[-7:] + self._image_ext)
        assert os.path.exists(image_path_1), \
               'Path does not exist: {}'.format(image_path_1)
        assert os.path.exists(image_path_2), \
               'Path does not exist: {}'.format(image_path_2)
        return image_path_1, image_path_2

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, self._image_set,
                                      self._image_set + '.txt')
        
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        #print(image_index)
        return image_index

    def _get_default_path(self):
        """
        Return the default path where kaist dataset is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'kaist')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print ('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_kaist_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_kaist_annotation(self, index):
        """
        Load image and bounding boxes info from text file in the kaist dataset
        format.
        """


        filename = os.path.join(self._data_path, self._image_set, 'annotations', index + '.txt')
        print('Loading: {}'.format(filename))

        #从第二行开始读
        """
        with open(filename) as f:
            lines = f.readlines()[1:]

        num_objs = len(lines)

        for obj in lines:
            info = obj.split()
            if info[0] != "person":
                
        

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.

        #读进来的数据格式为
        # obj,x1,y1,width,height
        #其中obj为person\person?/people/cyclist

        ix = 0
        for obj in lines:
            # Make pixel indexes 0-based
            info = obj.split()
            x1 = float(info[1])
            y1 = float(info[2])
            x2 = x1 + float(info[3])
            y2 = y1 + float(info[4])
            assert(x2>=x1)
            assert(y2>=y1)

            #部分标注使得x2 > 图片宽度,去除掉这部分
            if x2 >= 640 or y2 >= 512:
                x2 = 640
                y2 = 512


            cls = self._class_to_ind['pedestrian']
            #import pdb;pdb.set_trace()
            boxes[ix, :] = [x1-1, y1-1, x2-1, y2-1]
            #boxes[ix, :] = [x1, y1, x2, y2]

            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            ix = ix + 1
        overlaps = scipy.sparse.csr_matrix(overlaps)
        """

        with open(filename) as f:
            lines = f.readlines()[1:]

        num_pers = 0
        for obj in lines:
            info = obj.split()
            if info[0] == 'person':
                num_pers += 1
            #elif info[0] == 'ignore':
            #    num_igns += 1
            else:
                print(info[0])
                #raise NotImplementedError

        boxes = np.zeros((num_pers, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_pers), dtype=np.int32)
        overlaps = np.zeros((num_pers, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_pers), dtype=np.float32)

        #ign_boxes = np.zeros((num_igns, 4), dtype=np.uint16)

        # Load object bounding boxes into a data frame.
        ixp = 0
        for obj in lines:
            # Make pixel indexes 0-based
            info = obj.split()
            x1 = float(info[1]) - 1
            y1 = float(info[2]) - 1
            x2 = x1 + float(info[3]) - 1
            y2 = y1 + float(info[4]) - 1

            assert(x2>=x1)
            assert(y2>=y1)

            assert(x2 <= 640)
            assert(y2 <= 512)


            if info[0] == 'person':
                cls = self._class_to_ind['pedestrian']
                boxes[ixp, :] = [x1, y1, x2, y2]
                gt_classes[ixp] = cls
                overlaps[ixp, cls] = 1.0
                seg_areas[ixp] = (x2 - x1 + 1) * (y2 - y1 + 1)
                ixp = ixp + 1

        overlaps = scipy.sparse.csr_matrix(overlaps)





        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    #############################matrix##########################

    def evaluate_detections(self, all_boxes, output_dir, suffix=''):
        self._write_kaist_results_file(all_boxes, output_dir, suffix)
        self._do_matlab_eval(output_dir, suffix)

    def _write_kaist_results_file(self, all_boxes, output_dir, suffix=''):

        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            detdir = os.path.join(output_dir, 'det' + suffix)
            if not os.path.exists(detdir):
                os.makedirs(detdir)
            for im_ind, index in enumerate(self.image_index):
                filename = os.path.join(
                    detdir, index + '.txt')

                #创建文件夹
                filepath = os.path.join(detdir,index[:-6])
                if not os.path.exists(filepath):
                    os.makedirs(filepath)

                with open(filename, 'w') as f:
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.4f} {:.4f} {:.4f} {:.4f} {:.8f}\n'.
                                format(cls, dets[k, 0],
                                       dets[k, 1], dets[k, 2],
                                       dets[k, 3], dets[k, 4]))
        print('Writing kaist pedestrian detection results file done.')

    def _do_matlab_eval(self, output_dir='output', suffix=''):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')

        if not os.path.exists(output_dir + '/metrix_result.txt'):
            os.mknod(output_dir + '/metrix_result.txt')

        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'KAISTdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'kaist_eval_full(\'{:s}\',\'{:s}\'); quit;"' \
            .format(os.path.join(output_dir, 'det' + suffix), self._data_path)
        cmd += '|tee %s/metrix_result.txt' % output_dir
        print(('Running:\n{}'.format(cmd)))
        status = subprocess.call(cmd, shell=True)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.kaist_rgb import kaist_rgb
    d = kaist('train-all02')
    res = d.roidb
    from IPython import embed; embed()
