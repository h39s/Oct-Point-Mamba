
import os
import argparse
import json
import wget
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement


parser = argparse.ArgumentParser()
parser.add_argument('--path_in', type=str, default='data/nuscenes/train')
parser.add_argument('--path_out', type=str, default='logs/nucenes/pred')
parser.add_argument('--path_pred', type=str, default='logs/nuscenes/pred_eval')
parser.add_argument('--filelist', type=str, default='data/nuscenes/v1.0-mini.txt')
parser.add_argument('--run', type=str, default='generate_output_seg',  # noqa
    help='Choose from `generate_output_seg` and `calc_iou`')
args = parser.parse_args()


class_ids = tuple(range(32))
label_dict = np.arange(len(class_ids), dtype=np.int32)
ilabel_dict = np.array(class_ids)


def generate_output_seg():
  ''' Converts the predicted probabilities to segmentation labels: merge the
  predictions for each chunk; map the predicted labels to the original labels.
  '''

  # load filelist
  filename_scans = []
  with open(args.filelist, 'r') as fid:
    for line in fid:
      filename_scans.append(line.split()[0][:-4])

  # process
  probs = {}
  for filename_scan in tqdm(filename_scans, ncols=80):
    filename_pred = os.path.join(args.path_pred, filename_scan + '.eval.npz')
    pred = np.load(filename_pred)
    prob0 = pred['prob']
    probs[filename_scan] = probs.get(filename_scan, 0) + prob0

  # output
  os.makedirs(args.path_out, exist_ok=True)
  for filename, prob in tqdm(probs.items(), ncols=80):
    filename_label = filename + '.txt'
    filename_npy = filename + '.npy'
    label = np.argmax(prob, axis=1)
    label = ilabel_dict[label]
    np.savetxt(os.path.join(args.path_out, filename_label), label, fmt='%d')
    # np.save(os.path.join(args.path_out, filename_npy), prob)


def calc_iou():
  # init
  intsc, union, accu = {}, {}, 0
  for k in class_ids[1:]:
    intsc[k] = 0
    union[k] = 0

  # load files
  pred_files = sorted(os.listdir(args.path_pred))
  pred_files = [f for f in pred_files if f.endswith('.txt')]
  for filename in tqdm(pred_files, ncols=80):
    label_pred = np.loadtxt(os.path.join(args.path_pred, filename))
    label_gt = np.loadtxt(os.path.join(args.path_in, filename))

    ac = (label_gt == label_pred).mean()
    tqdm.write("Accu: %s, %.4f" % (filename, ac))
    accu += ac

    for k in class_ids[1:]:
      pk, lk = label_pred == k, label_gt == k
      intsc[k] += np.sum(np.logical_and(pk, lk).astype(np.float32))
      union[k] += np.sum(np.logical_or(pk, lk).astype(np.float32))

  # iou
  iou_part = 0
  for k in class_ids[1:]:
    iou_part += intsc[k] / (union[k] + 1.0e-10)
  iou = iou_part / len(class_ids[1:])
  print('Accu: %.6f' % (accu / len(pred_files)))
  print('IoU: %.6f' % iou)


if __name__ == '__main__':
  eval('%s()' % args.run)
