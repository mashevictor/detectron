#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import cv2
from primesense import openni2
from primesense import _openni2 as c_api
from collections import defaultdict
import argparse
import cv2
import glob
import logging
import os
import sys
import time
from caffe2.python import workspace
from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)
np.set_printoptions(threshold=np.inf)
def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /home/victor/facebook/infer_simple)',
        default='/home/victor/detectron/detectron-visualizations',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

dist ='/home/victor/software/OpenNI-Linux-x64-2.3/Redist'
openni2.initialize(dist)
if (openni2.is_initialized()):
    print "openNI2 initialized"
else:
    print "openNI2 not initialized"
dev = openni2.Device.open_any()
rgb_stream = dev.create_color_stream()
depth_stream = dev.create_depth_stream()
rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=640, resolutionY=480, fps=30))
rgb_stream.start()
depth_stream.set_mirroring_enabled(False)
depth_stream.start()
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
utils.logging.setup_logging(__name__)
args = parse_args()
def get_rgb():
    bgr   = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(480,640,3)
    rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return rgb
def get_depth():
    dmap = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(),dtype=np.uint16).reshape(480,640)
    d4d = np.uint8(dmap.astype(float) *255/ 2**12-1)
    d4d = cv2.cvtColor(d4d,cv2.COLOR_GRAY2RGB)
    d4d = 255 - d4d    
    return dmap, d4d
s=0
done = False
while not done:
    key = cv2.waitKey(1) & 255
    if key == 27:
        print "\tESC key detected!"
        done = True
    elif chr(key) =='s':
        print "\ts key detected. Saving image {}".format(s)
        cv2.imwrite("ex2_"+str(s)+'.png', rgb)
        cv2.imwrite("ex1_"+str(s)+'.png', d4d)
        np.savetxt("ex1dmap_"+str(s)+'.out',dmap)
    dmap,d4d = get_depth()
    rgb = get_rgb()
    cv2.imshow('rgb', rgb)
    cv2.imshow('depth', d4d)

logger = logging.getLogger(__name__)
merge_cfg_from_file(args.cfg)
cfg.NUM_GPUS = 1
args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
assert_and_infer_cfg()
model = infer_engine.initialize_model_from_cfg(args.weights)
dummy_coco_dataset = dummy_datasets.get_coco_dataset()
im = get_rgb()
vis_utils.vis_one_image(
        im[:, :, ::-1],
        args.output_dir,
        cls_boxes,
        cls_segms,
        cls_keyps,
        dataset=dummy_coco_dataset,
        box_alpha=0.3,
        show_class=True,
        thresh=0.7,
        kp_thresh=2
)
cv2.destroyAllWindows()
rgb_stream.stop()
openni2.unload()
print ("Terminated")
