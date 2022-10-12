#!/usr/bin/env python
from time import time
import glob
import os
import sys
import semantic_change_segmentation as scs
import jaccard_index as jaccard
import numpy as np
from PIL import Image

# Check input directories.
submit_dir = os.path.join(sys.argv[1], 'res')
truth_dir = os.path.join(sys.argv[1], 'ref')
if not os.path.isdir(submit_dir):
    print("submit_dir {} doesn't exist".format(submit_dir))
    sys.exit()
if not os.path.isdir(truth_dir):
    print("truth_dir {} doesn't exist".format(truth_dir))
    sys.exit()

# Create output directory.
output_dir = sys.argv[2]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

time_start = time()

pred_dir = submit_dir
gt_dir = truth_dir

def _read_image_as_np(path):
    with open(path, 'rb') as f:
        return np.array(Image.open(f))

def _get_cube_id(filename):
    return filename.split('.')[0][:-11]

IMGS_PER_CUBE = 24

scs_metric = scs.SCSMetric(num_classes=6, ignore_label=255)
miou_metric = jaccard.JaccardIndexMetric(num_classes=6, ignore_label=255)

image_list = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
gt, gt_next = None, None
pred, pred_next = None, None

for i in range(len(image_list)):

    gt_img_name = image_list[i]
    filename = gt_img_name.split('/')[-1]
    pred_img_name = os.path.join(pred_dir, filename)
    if not os.path.exists(pred_img_name):
        raise ValueError("Submission segmentation map not found - terminating!\n"
                         "Missing segmentation map: {}".format(filename))
    print("Found corresponding submission file {} for reference file {}"
          "".format(pred_img_name, gt_img_name))

    gt = _read_image_as_np(gt_img_name)
    pred = _read_image_as_np(pred_img_name)

    if gt.shape != pred.shape:
        raise AttributeError("Shapes do not match! Image id {}, Prediction mask {}, "
                             "ground truth mask {}"
                             "".format(os.path.basename(filename),
                                       pred.shape,
                                       gt.shape))

    if (pred.astype(int) >= 6).any():
        raise ValueError(
            "Input {} does not correspond to submission format! All predicted mask values need to be from 0 to 5!!!".format(
                filename))

    cube_id = _get_cube_id(filename)
    miou_metric.update_state(gt, pred)
    if (i % IMGS_PER_CUBE) >= (IMGS_PER_CUBE - 1):
        continue

    gt_img_name_next = image_list[i + 1]
    filename = gt_img_name_next.split('/')[-1]
    pred_img_name_next = os.path.join(pred_dir, filename)

    if not os.path.exists(pred_img_name_next):
        raise ValueError("Submission segmentation map not found - terminating!\n"
                         "Missing segmentation map: {}".format(filename))
    print("Found corresponding submission file {} for reference file {}"
          "".format(pred_img_name_next, gt_img_name_next))

    gt_next = _read_image_as_np(gt_img_name_next)
    pred_next = _read_image_as_np(pred_img_name_next)

    if gt_next.shape != pred_next.shape:
        raise AttributeError("Shapes for the images do not match! Image id {}, Prediction mask {}, "
                             "ground truth mask {}"
                             "".format(os.path.basename(filename),
                                       pred_next.shape,
                                       gt_next.shape))
    #if ((pred_next.astype(int) >= 6) & pred_next.astype(int)).any():
    if (pred_next.astype(int) >= 6).any():
        raise ValueError(
            "Input {} does not correspond to submission format! All predicted mask values need to be from 0 to 5!!!".format(
                filename))


    # Semantic Change Segmentation
    gt_change = np.not_equal(gt, gt_next)
    pred_change = np.not_equal(pred, pred_next)

    scs_metric.update_state(gt_change, gt_next, pred_change, pred_next, cube_id)

scs_score = scs_metric.result()
miou_score = miou_metric.result()

# Write scores to a file named "scores.txt"
with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
    output_file.write("SCS: %f\n" % np.round(scs_score['SCS'] * 100, decimals=1))
    output_file.write("BC: %f\n" % np.round(scs_score['BC'] * 100, decimals=1))
    output_file.write("SC: %f\n" % np.round(scs_score['SC'] * 100, decimals=1))
    output_file.write("mIoU: %f\n" % np.round(miou_score['mIoU'] * 100, decimals=1))
    output_file.write("imp: %f\n" % np.round(miou_score['IoU_per_class'][0] * 100, decimals=1))
    output_file.write("agr: %f\n" % np.round(miou_score['IoU_per_class'][1] * 100, decimals=1))
    output_file.write("for: %f\n" % np.round(miou_score['IoU_per_class'][2] * 100, decimals=1))
    output_file.write("wet: %f\n" % np.round(miou_score['IoU_per_class'][3] * 100, decimals=1))
    output_file.write("soil: %f\n" % np.round(miou_score['IoU_per_class'][4] * 100, decimals=1))
    output_file.write("water: %f\n" % np.round(miou_score['IoU_per_class'][5] * 100, decimals=1))

total_time = time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))

