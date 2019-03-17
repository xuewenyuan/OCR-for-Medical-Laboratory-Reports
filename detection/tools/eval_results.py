from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
#import utils.cython_bbox
import pickle
import subprocess
import uuid
from datasets.voc_eval import voc_eval
from model.config import cfg


def _do_python_eval(_devkit_path, _detpath, _year, _image_set, _classes, output_dir):
    
    annopath = os.path.join(
      _devkit_path, 
      'VOC' + _year,
      'Annotations',
      '{:s}.xml')
    imagesetfile = os.path.join(
      _devkit_path,
      'VOC' + _year,
      'ImageSets',
      'Main',
      _image_set + '.txt')
    cachedir = os.path.join(_devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(_year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    for i, cls in enumerate(_classes):#('__background__', 'text')
      if cls == '__background__':
        continue
      filename = _detpath
      rec, prec, ap = voc_eval(
        filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.6,
        use_07_metric=use_07_metric)
      aps += [ap]
      print('Recall: {}'.format(rec[-1]))
      print('Precision: {}'.format(prec[-1]))
      f1 = 2.*rec[-1]*prec[-1]/(rec[-1]+prec[-1])
      print('F1: {}'.format(f1))
      print(('AP for {} = {:.4f}'.format(cls, ap)))
      with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
      print(('{:.3f}'.format(ap)))
    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')


if __name__ == '__main__':
    #load results
    # faster rcnn crop
    _devkit_path = './data/VOCdevkit2007'
    _detpath = './tools/results/results.txt'
    output_dir = './tools/results/test'
    

    _year = '2007'
    _image_set = 'test'
    _classes = ('__background__', 'text')
    _do_python_eval(_devkit_path, _detpath, _year, _image_set, _classes, output_dir)
    
