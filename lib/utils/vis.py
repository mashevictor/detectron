from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import cv2
import numpy as np
import os
import sys
import pycocotools.mask as mask_util
import random
from utils.colormap import colormap
import utils.env as envu
import utils.keypoints as keypoint_utils
envu.set_up_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
np.set_printoptions(threshold=np.inf)
from itertools import chain
plt.rcParams['pdf.fonttype'] = 42
_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)

def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines

def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes

def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')

def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
    img = img.astype(np.float32)
    idx = np.nonzero(mask)
    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col
    if show_border:
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)
    return img.astype(np.uint8)

def vis_class(img, pos, class_str, font_scale=0.35):
    x0, y0 = int(pos[0]), int(pos[1])
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, _GREEN, -1)
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img

def vis_bbox(img, bbox, thick=1):
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
    return img

def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    dataset_keypoints, _ = keypoint_utils.get_keypoints()
    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    kp_mask = np.copy(img)
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_one_image_opencv(
        im, boxes, segms=None, keypoints=None, thresh=0.9, kp_thresh=2,
        show_box=False, dataset=None, show_class=False):
    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)
    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return im
    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)
        color_list = colormap()
        mask_color_id = 0
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue
        if show_box:
            im = vis_bbox(
                im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
        if show_class:
            class_str = get_class_string(classes[i], score, dataset)
	    #print ("woca**************************class_str")
	    #print (class_str)
            im = vis_class(im, (bbox[0], bbox[1] - 2), class_str)
        if segms is not None and len(segms) > i:
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            im = vis_mask(im, masks[..., i], color_mask)
        if keypoints is not None and len(keypoints) > i:
            im = vis_keypoints(im, keypoints[i], kp_thresh)
    return (im)

def vis_one_image_opencv2(
        im, boxes, segms=None, keypoints=None, thresh=0.9, kp_thresh=2,
        show_box=False, dataset=None, show_class=False):
    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)
    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return im
    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)
        color_list = colormap()
        mask_color_id = 0
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)
    #print ("sorted_inds&&&&&&&&&&&&&&&&&&")
    #print (sorted_inds)
    class_shuzu=[]
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue
        if show_box:
            im = vis_bbox(
                im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
        if show_class:
            class_str = get_class_string(classes[i], score, dataset)
	    class_shuzu.append(class_str)
            im = vis_class(im, (bbox[0], bbox[1] - 2), class_str)
        if segms is not None and len(segms) > i:

            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            im = vis_mask(im, masks[..., i], color_mask)
        if keypoints is not None and len(keypoints) > i:
            im = vis_keypoints(im, keypoints[i], kp_thresh)
    return (class_shuzu)

def vis_one_image(
        im, im_name,  boxes, segms=None, keypoints=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext=None):
    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)
    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return
    dataset_keypoints, _ = keypoint_utils.get_keypoints()
    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)
    color_list = colormap(rgb=True) / 255
    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)
    mask_color_id = 0
    C=[]
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='g',
                          linewidth=0.5, alpha=box_alpha))
        if show_class:
            ax.text(
                bbox[0], bbox[1] - 2,
                get_class_string(classes[i], score, dataset),
                fontsize=3,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')
        if segms is not None and len(segms) > i:
	    #print ("duoshaogeIIIIIIIIIIIIIIIIIIIIIIIIIII")
	    #print (i)
	    #print ("len(segms)*********")
	    #print (len(segms))
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = masks[:, :, i]
            _, contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=color_mask,
                    edgecolor='w', linewidth=0.2,
                    alpha=0.5)	
		ddd=list(c)
		woca= c.reshape((-1, 2))
		arr = np.array(woca)
		key = np.unique(woca)
		result = {}
		k_qu=[]
		arr_end=[]
		arr_end2=[]
		arr_endlast=[]
		A=[]
		B=[]
		Bplus=[]
		jjj=[]
		Bwanmei=[]
		Bcao=[]
		Bao=[]
		for k in key:
			mask = (arr == k)
			arr_new = arr[mask]
			v = arr_new.size
			result[k] = v
			x=np.argwhere(arr== k)
			if v>1:
				x=np.argwhere(arr== k)
				x=np.array(x)
				x0=arr[:,0]
				y0=arr[:,1]
				y0lie=[]
				for i in range(0,len(x0)):
					if x0[i]!=k:
						pass
					if x0[i]==k:
						y0lie.append(y0[i])					
				y0lienew=[]
				arr_new=[]
				arr_new_2=[]
				if y0lie==[]:
					pass
				else:
					miny0=np.min(y0lie)
					maxy0=np.max(y0lie)
					for i in range(miny0,maxy0+1):
						y0lienew.append(i)
					y0liezuizhong=[]
					if y0lienew==[]:
						pass
					else:
						miny0lienew=np.min(y0lienew)
						maxy0lienew=np.max(y0lienew)
						for i in range(miny0lienew,maxy0lienew+1):
							y0liezuizhong.append(i)				
					for i in range(0,len(y0liezuizhong)):
						arr_temp=[k,y0liezuizhong[i]]
						arr_new.append(arr_temp)
					arr_end.append(arr_new)
				x0lie=[]
				for i in range(0,len(y0)):
					if y0[i]!=k:
						pass
					if y0[i]==k:
						x0lie.append(x0[i])					
				x0lienew=[]
				arr_new2=[]
				if x0lie==[]:
					pass
				else:
					minx0=np.min(x0lie)
					maxx0=np.max(x0lie)
					for i in range(minx0,maxx0+1):
						x0lienew.append(i)
					x0liezuizhong=[]
					if x0lienew==[]:
						pass
					else:
						minx0lienew=np.min(x0lienew)
						maxx0lienew=np.max(x0lienew)
						for i in range(minx0lienew,maxx0lienew+1):
							x0liezuizhong.append(i)						
					for i in range(0,len(x0liezuizhong)):
						arr_temp=[x0liezuizhong[i],k]
						arr_new2.append(arr_temp)
					arr_end2.append(arr_new2)
	    arr_endlast=arr_end+arr_end2+ddd
	    A=list(chain(*arr_endlast))
	    B=np.array(list(set([tuple(t) for t in A])))
	
	    if len(B)>10:
	    	Bplus=random.sample(B,5)
	    if len(B)<10:
	    	jjj=arr_endlast
	    Bwanmei=Bplus+jjj
	    Bcaotmp=list(chain(*Bwanmei))
	    Bcao=np.array(Bcaotmp)
	    Bao=Bcao.reshape(-1,2)
	    C.append(Bao)


        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            plt.autoscale(False)
            for l in range(len(kp_lines)):
                i1 = kp_lines[l][0]
                i2 = kp_lines[l][1]
                if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                    x = [kps[0, i1], kps[0, i2]]
                    y = [kps[1, i1], kps[1, i2]]
                    line = plt.plot(x, y)
                    plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                if kps[2, i1] > kp_thresh:
                    plt.plot(
                        kps[0, i1], kps[1, i1], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)
                if kps[2, i2] > kp_thresh:
                    plt.plot(
                        kps[0, i2], kps[1, i2], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)
            mid_shoulder = (
                kps[:2, dataset_keypoints.index('right_shoulder')] +
                kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            sc_mid_shoulder = np.minimum(
                kps[2, dataset_keypoints.index('right_shoulder')],
                kps[2, dataset_keypoints.index('left_shoulder')])
            mid_hip = (
                kps[:2, dataset_keypoints.index('right_hip')] +
                kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            sc_mid_hip = np.minimum(
                kps[2, dataset_keypoints.index('right_hip')],
                kps[2, dataset_keypoints.index('left_hip')])
            if (sc_mid_shoulder > kp_thresh and
                    kps[2, dataset_keypoints.index('nose')] > kp_thresh):
                x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
                y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                x = [mid_shoulder[0], mid_hip[0]]
                y = [mid_shoulder[1], mid_hip[1]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines) + 1], linewidth=1.0,
                    alpha=0.7)
    plt.close('all')
    C=np.array(C)             
    return C
