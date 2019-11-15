%mkc Eval
addpath( genpath( 'libs' ) );

% gt,dt datadir
dataDir='data/kaist-rgbt/';
imgDir = [dataDir 'images']; 
gtDir = [dataDir 'annotations'];
testSet = [dataDir 'imageSets/test-all-20.txt'];
%bbsNm= 'models/AcfKAIST/train-all-20/RGB-T-TM+TO/AcfKAIST-RGB-T-TM+TODets.txt';

% detfile
Halfway_det = 'Liu_det/det.txt';
fusion = 'D:/mkc/Graduate/material_for_graduate_all/multispectral_pedestrain_results/Last_result_important/desktop_result_last/result/fusion/det_full_path_test-all-20.txt';
color = 'D:/mkc/Graduate/material_for_graduate_all/multispectral_pedestrain_results/Last_result_important/desktop_result_last/result/color/det_full_path_test-all-20.txt';
pytorch = 'D:/mkc/Graduate/det.txt'
%?????????????test_day_20:1455?????????test_all_20?gt,??????????????
%Halfway_det:test%
%����·��������%
%det_file = fusion;
det_file = pytorch

thr = 0.5; %the threshold on oa for comparing two bbs
mul = 0; %if true allow multiple matches to each gt
ref = 10.^(-2:.25:0);

pLoad={'lbls',{'person'},'ilbls',{'people','person?','cyclist'},'squarify',{3,.41}};
pLoad = [pLoad, 'hRng',[45 inf],'vType',{{'none','partial'}},'xRng',[5 635],'yRng',[5 475]];

imgNms = bbGt2('getSubsetFiles',{imgDir,imgDir,gtDir}, testSet);

%1, load gt, dt from dataDir 
[gt,dt] = bbGt2('loadAll',gtDir,det_file,pLoad,testSet);


% run evaluation using bbGt
[gt,dt] = bbGt2('evalRes',gt,dt,thr,mul);
[fp,tp,score,miss] = bbGt2('compRoc',gt,dt,1,ref);
miss=exp(mean(log(max(1e-10,1-miss)))); 
roc=[score fp tp];
lims = [3.1e-4 3.1e1 0 1];
clr = 'g';lineSt = '-';

figure; 
plotRoc([fp tp],'logx',1,'logy',0,'xLbl','False positives per image',...
  'lims',lims,'color',clr,'lineSt', lineSt,'smooth',1,'fpTarget',ref,...
  'lineWd',2);
title(sprintf('log-average miss rate = %.2f%%',miss*100));   
