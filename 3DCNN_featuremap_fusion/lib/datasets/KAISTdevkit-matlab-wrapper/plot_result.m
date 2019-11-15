load("/home/mkc/mkc/Project/pytorch1_0_frcnn/faster-rcnn.pytorch/output/vgg16/test-all20-fusion/faster_rcnn_10/res.mat")

exps = {
  'Reasonable-all',       'test-all20',       [55 inf],    {{'none','partial'}}
  'Reasonable-day',    'test-day',    [55 inf],    {{'none','partial'}}
  'Reasonable-night', 'test-night', [55 inf],    {{'none','partial'}}
  'Scale=near',              'test-all20',       [115 inf], {{'none'}}
  'Scale=medium',      'test-all20',        [45 115],   {{'none'}}
  'Scale=far',                  'test-all20',       [1 45],   {{'none'}}
  'Occ=none',               'test-all20',       [1 inf],      {{'none'}}
  'Occ=partial',             'test-all20',       [1 inf],      {{'partial'}}
  'Occ=heavy',              'test-all20',        [1 inf],     {{'heavy'}}
  };

for i = 1:9
    fprintf('%-30s \t log-average miss rate = %02.2f%% (%02.2f%%) recall = %02.2f%% (%02.2f%%)\n', iexp{1}, miss_ori*100, miss_imp*100, roc_ori(end, 3)*100, roc_imp(end, 3)*100);
end