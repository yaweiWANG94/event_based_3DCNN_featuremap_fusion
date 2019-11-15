from roi_data_layer.roidb import combined_roidb
import cv2
import os
import numpy as np
imdbval_name = "kaist_test-all20-rgb"



imdb, roidb, ratio_list, ratio_index = combined_roidb(imdbval_name, False)


num_images= len(roidb)


def vis_notations(im, roidb):
    """Visual debugging of detections."""
    gt_boxes1 = roidb['boxes']
    for gt_ind in range(gt_boxes1.shape[0]):
        bbox = gt_boxes1[gt_ind,:]
        cv2.rectangle(im, (bbox[0], bbox[3]), (bbox[2], bbox[1]), (0, 255, 0), 1)

    gt_boxes2 = roidb['boxes2']
    for gt_ind in range(gt_boxes2.shape[0]):
        bbox = gt_boxes2[gt_ind,:]
        cv2.rectangle(im, (bbox[0], bbox[3]), (bbox[2], bbox[1]), (0, 0, 255), 1)

    return im

output_dir = "/home/mkc/mkc/Project/pytorch1_0_frcnn/faster-rcnn.pytorch/output/"

pic_path = os.path.join(output_dir, 'pic')
if not os.path.exists(pic_path):
    os.makedirs(pic_path)

for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))
    im2show = np.copy(im)
    im2show = vis_notations(im, roidb[i])
    cv2.imwrite(pic_path + '/result_%s.png' % str(i), im2show)



print("haha")