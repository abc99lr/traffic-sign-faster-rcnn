from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn import data_generators
from keras_frcnn.simple_parser import get_data
from sklearn.metrics import average_precision_score


sys.setrecursionlimit(40000)

def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    fx = width / float(new_width)
    fy = height / float(new_height)
    return img, ratio, fx, fy


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio, fx, fy = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio, fx, fy


def get_real_coordinates(ratio, x1, y1, x2, y2):
    """ Method to transform the coordinates of the bounding box to its original size """
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)


def get_map(pred, gt, f):
    T = {}
    P = {}
    fx, fy = f

    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array([s['prob'] for s in pred])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']
        pred_prob = pred_box['prob']
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        for gt_box in gt:
            gt_class = gt_box['class']
            gt_x1 = gt_box['x1']/fx
            gt_x2 = gt_box['x2']/fx
            gt_y1 = gt_box['y1']/fy
            gt_y2 = gt_box['y2']/fy
            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            if iou >= 0.5:
                found_match = True
                gt_box['bbox_matched'] = True
                break
            else:
                continue

        T[pred_class].append(int(found_match))

    for gt_box in gt:
        if not gt_box['bbox_matched']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []

            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)
    return T, P


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-p", "--path", dest="test_path", help="Path to test data.", default='../dataset/PNG_test')
    parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                    help="Number of ROIs per iteration. Higher means more memory use.", default=32)
    parser.add_option("--config_filename", dest="config_filename", help=
                    "Location to read the metadata related to the training (generated when training).",
                    default="zfnet_config.pickle")
    parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='zfnet')

    (options, args) = parser.parse_args()
    config_output_filename = options.config_filename
    img_path = options.test_path

    if not options.test_path:   # if filename is not given
        parser.error('Error: path to test data must be specified. Pass --path to command line')

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    if C.network == 'fcnet':
        import keras_frcnn.fcnet as nn
    elif C.network == 'zfnet':
        import keras_frcnn.zfnet as nn
    elif C.network == 'vgg':
        import keras_frcnn.vgg as nn


    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    class_mapping = C.class_mapping
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    C.num_rois = int(options.num_rois)

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, C.num_features)
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (zfnet)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(C.model_path))       # model_path specified in config file
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    all_imgs = []
    classes = {}
    all_imgs, _, _ = get_data(options.test_path)
    #test_imgs = [s for s in all_imgs if s['imageset'] == 'train']      # mAP
    print("DEBUGGING 218: test imgs: ", len(test_imgs))
    classification_threshold = 0.8		# threshold above which we classify as positive

    # for mAP
    T, P = {}, {}

    counter = 0
    for idx, img_data in enumerate(test_imgs):
        print('{}/{}'.format(idx, len(test_imgs)))
        img_name = img_data['filepath'].split('/')[-1]
        print("DEBUGGING 232 img_name:", img_name)
        print("img {}: {}".format(str(counter), img_name))
        counter += 1
        start_time = time.time()
        img_path = '../faster-rcnn/dataset/PNG_train/'
        filepath = os.path.join(img_path, img_name)
        print("DEBUGGING filepath:", filepath)
        img = cv2.imread(filepath)

        X, ratio, fx, fy = format_img(img, C)

        assert K.image_dim_ordering() == 'tf'
        X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)
        # R = bboxes (300 ,4)
        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0]//C.num_rois + 1):    	# R.shape[0] = 300
            ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            # [pred_cls, pred_regr] = model_classifier_only.predict([F, ROIs])
            [pred_cls, pred_regr] = model_classifier.predict([F, ROIs])

            for ii in range(pred_cls.shape[1]):

                if np.max(pred_cls[0, ii, :]) < classification_threshold or np.argmax(pred_cls[0, ii, :]) == (pred_cls.shape[2] - 1):
                    pass

                cls_name = class_mapping[np.argmax(pred_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(pred_cls[0, ii, :])	 # index of predicted class
                try:
                    (tx, ty, tw, th) = pred_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(pred_cls[0, ii, :]))

        bboxes.pop('bg')    # added to avoid plotting background bbox
        probs.pop('bg')     # added to avoid plotting background bbox

        pred_bboxs = []

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]

                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

                textLabel = '{}: {}'.format(key, int(100*new_probs[jk]))
                pred_bboxs.append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]})

                (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                textOrg = (real_x1, real_y1-0)

                cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        print('Elapsed time = {}'.format(time.time() - start_time))
        print("pred_bboxs:", pred_bboxs)
        #print("gt boxes:", img_data['bboxes'])
        cv2.imwrite('./results_imgs/{}'.format(img_name), img)

        '''
        # Calculate mAP
        t, p = get_map(pred_bboxs, img_data['bboxes'], (fx, fy))
        for key in t.keys():
            if key not in T:
                T[key] = []
                P[key] = []
            T[key].extend(t[key])
            P[key].extend(p[key])
        all_aps = []
        for key in T.keys():
            print("T:", T[key])
            print("P:", P[key])
            ap = average_precision_score(T[key], P[key])
            print('{} AP: {}'.format(key, ap))
            if ap == float('nan'):
                continue
            all_aps.append(ap)
        print('mAP = {}'.format(np.mean(np.array(all_aps))))
        '''
