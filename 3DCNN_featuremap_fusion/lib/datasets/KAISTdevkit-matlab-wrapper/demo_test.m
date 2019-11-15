function demo_test()

%% specify path of detections
%dtDir = '/home/mkc/mkc/Project/pytorch1_0_frcnn/faster-rcnn.pytorch/output/vgg16/test-all20-fusion/faster_rcnn_10_7_18575/det';
%dtDir = '/home/mkc/mkc/Project/pytorch1_0_frcnn/faster-rcnn.pytorch/output/vgg16/kaist_test-all20-rgb/faster_rcnn_10/det';
dtDir = '/home/mkc/mkc/Project/pytorch1_0_frcnn/faster-rcnn.pytorch/output/vgg16/test-all20-fusion/faster_rcnn_10_6_4643/det';

%% specify path of groundtruth annotaions
gtDir = '/home/mkc/mkc/Project/pytorch1_0_frcnn/faster-rcnn.pytorch/data/kaist';

%% evaluate detection results
kaist_eval_full(dtDir, gtDir, false, true);

end
