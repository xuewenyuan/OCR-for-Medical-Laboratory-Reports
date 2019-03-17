#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
#from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

os.environ['CUDA_VISIBLE_DEVICES']='5'

#CLASSES = ('__background__',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('__background__', 'text')


#NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_30000.ckpt',),'res101': ('res101_faster_rcnn_iter_50000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def save_results(image_name,im,line,thresh):
    inds=np.where(line[:,-1]>=thresh)[0]
    if len(inds)==0:
        return
    with open('./tools/src_multix0.5/results.txt','a') as f:
        for i in inds:
            bbox=line[i,:4]
            score=line[i,-1]
            f.write(image_name+' ')
            f.write(str(score)+' ')
            f.write(str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+'\n')

def crop(img, sess, net, NMS_THRESH, thresh=0.5):


	# Detect all object classes in this crop and regress object bounds
	scores, boxes = im_detect(sess, net, img)
	scores = scores[:, 1]
    #print('score shape',scores.shape,'boxes shape',boxes.shape)
	dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
	keep = nms(dets, 0.2)
	dets = dets[keep, :]
	results = dets[np.where(dets[:, -1] >= thresh),:][0]
	#print('results shape',results.shape)
	#print('count_crop:',count_crop)
	return results

def vis_detections(im, class_name, dets, thresh):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
	print("no text detected!")
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=1.5)
            )
        #ax.text(bbox[0], bbox[1] - 2,
        #        '{:s} {:.3f}'.format(class_name, score),
        #        bbox=dict(facecolor='blue', alpha=0.5),
        #        fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  0.8),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(sess, net, image_name):
	"""Detect object classes in an image using pre-computed object proposals."""

	# Load the demo image
	#im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
	im_file = os.path.join('/home/xuewenyuan/Dataset/CMDD/cmdd-multi-set2/VOC2007/JPEGImages', image_name+'.jpg')
	im = cv2.imread(im_file)
	height,width = im.shape[:2]


    # Detect all object classes with crops and regress object bounds
	CONF_THRESH = 0.6
	NMS_THRESH = 0.7
	timer = Timer()
	timer.tic()
	#bboxes, scores = crop(im , H_window, W_window, h_gap, w_gap, sess, net, NMS_THRESH, thresh=CONF_THRESH)
	results = crop(im , sess, net, NMS_THRESH, thresh=CONF_THRESH)
	timer.toc()
    #combine boxes
	#bboxes = combine(bboxes)
	save_results(image_name, im, results,CONF_THRESH)
	print('Detection took {:.3f}s for {:d} bboxes'.format(timer.total_time, len(results)))
    # Visualize detections for each class
	#vis_detections(im, 'text', results, CONF_THRESH)
    #CONF_THRESH = 0.8
    #NMS_THRESH = 0.3
    #for cls_ind, cls in enumerate(CLASSES[1:]):
    #    cls_ind += 1 # because we skipped background
    #    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    #    cls_scores = scores[:, cls_ind]
    #    dets = np.hstack((cls_boxes,
    #                      cls_scores[:, np.newaxis])).astype(np.float32)
    #    keep = nms(dets, NMS_THRESH)
    #    dets = dets[keep, :]
    #vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('./output', demonet, 'temp2019.3.5',DATASETS[dataset][0], 'default', NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 2, tag='default',
		#anchor_scales=[16,32,64,128,256,512], anchor_ratios=[0.5,1,2,5,10,15,25])
		#anchor_scales=[8, 16, 32], anchor_ratios=[0.5,1,2,5])
                           anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))


    #im_names = ['scan_item10+_24.jpg']#,'scan_item10-_41.jpg','illu_item10+_R_2.jpg', 'illu_item10-_RT_4.jpg']

    im_names = []
    testFile = '/home/xuewenyuan/Dataset/CMDD/cmdd-multi-set2/VOC2007/ImageSets/Main/test.txt'
    with open(testFile,'r') as f:
        for name in f.readlines():
            name = name.strip().replace(' ','').replace('\n','').replace('\r','')
            name = unicode(name, 'utf-8')
            im_names.append(name)

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)

    #plt.show()
