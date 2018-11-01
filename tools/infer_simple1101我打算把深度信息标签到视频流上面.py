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

dist ='/home/victor/software/OpenNI-Linux-x64-2.3/Redist'
openni2.initialize(dist)
if (openni2.is_initialized()):
    print ("openNI2 initialized")
else:
    print ("openNI2 not initialized")
dev = openni2.Device.open_any()
rgb_stream = dev.create_color_stream()
depth_stream = dev.create_depth_stream()
rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=640, resolutionY=480, fps=30))
rgb_stream.start()
depth_stream.set_mirroring_enabled(False)
depth_stream.start()

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
logger = logging.getLogger(__name__)
merge_cfg_from_file(args.cfg)
cfg.NUM_GPUS = 1
args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
assert_and_infer_cfg()
model = infer_engine.initialize_model_from_cfg(args.weights)
dummy_coco_dataset = dummy_datasets.get_coco_dataset()
while not done:
    key = cv2.waitKey(1) & 255
    if key == 27: 
        print ("\tESC key detected!")
        done = True
    elif chr(key) =='s': #screen capture
        print ("\ts key detected. Saving image {}".format(s))
        cv2.imwrite("ex2_"+str(s)+'.png', rgb)
    im = get_rgb()
    dmap,d4d = get_depth()
    timers = defaultdict(Timer)
    t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )
    logger.info('Inference time: {:.3f}s'.format(time.time() - t))
    for k, v in timers.items():
        logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        logger.info(
            ' \ Note: inference on the first image will be slower than the '
            'rest (caches and auto-tuning need to warm up)'
        )

    vis_utils.vis_one_image(
        im[:, :, ::-1],  # BGR -> RGB for visualization
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
    im=vis_utils.vis_one_image_opencv(im,cls_boxes, cls_segms,cls_keyps,dataset=dummy_coco_dataset,thresh=0.7,show_class=True)
    classxy=vis_utils.vis_one_image_opencv2(im,cls_boxes, cls_segms,cls_keyps,dataset=dummy_coco_dataset,thresh=0.7,show_class=True)
    
    if type(classxy[0][0])==np.ndarray:
	pass
    else:
	#print ("***********************IMIM****IMIM")
    	print(classxy)
    yinying=[]
    yinying=    vis_utils.vis_one_image(
        im,
        args.output_dir,
        cls_boxes,
        cls_segms,
        cls_keyps,
        dataset=dummy_coco_dataset,
        box_alpha=0.3,
        show_class=True,
        thresh=0.7,
)
    if yinying is None:
	pass
    else:
	for p in range(yinying.shape[0]):
		zhuan=tuple(yinying[p])
		f=[]
		woca=[]
		for i in range(len(zhuan)):
			zuobiao=zhuan[i]
			for d in range(dmap.shape[0]):
				for e in range(dmap.shape[1]):
					f=(d,e)
					if (f==zuobiao).all():
						#print ("zuizhong*****")
						if dmap[f]<20:
							pass
						else:					
							#print (dmap[f])
							woca.append(dmap[f])

					else:
						pass
	
		sumhe=0
		pingjun=0
		print ("**********woca******************")
		print (woca)
		if woca==[]:
			pass
		else:
			for i in range(len(woca)):
				sumhe=sumhe+woca[i]
			print ("**********pingjun******************")
			print (sumhe/len(woca))
	


	
    cv2.imshow('rgb', im)
    cv2.imshow('depth', d4d)
  

cv2.destroyAllWindows()
rgb_stream.stop()
openni2.unload()
print ("Terminated")
