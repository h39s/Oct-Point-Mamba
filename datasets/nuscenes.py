import ocnn
import torch
import random
import scipy.interpolate
import scipy.ndimage
import numpy as np

from ocnn.octree import Points, Octree
from ocnn.dataset import CollateBatch
from thsolver import Dataset
from typing import List

from .utils import ReadFile, Transform


def align_z(points: Points):
  points.points[:, 2] -= points.points[:, 2].min()
  return points


def rand_crop(points: Points, max_npt: int):
  r''' Keeps `max_npt` pts at most centered by a radomly chosen pts. 
  '''

  pts = points.points
  npt = points.npt
  crop_mask = torch.ones(npt, dtype=torch.bool)
  if npt > max_npt:
    rand_idx = torch.randint(low=0, high=npt, size=(1,))
    sort_idx = torch.argsort(torch.sum((pts - pts[rand_idx])**2, 1))
    crop_idx = sort_idx[max_npt:]
    crop_mask[crop_idx] = False
    points = points[crop_mask]
  return points, crop_mask


class NuscenesTransform(Transform):

  def __init__(self, flags):
    super().__init__(flags)

    # The `self.scale_factor` is used to normalize the input point cloud to the
    # range of [-1, 1]. If this parameter is modified, the `self.elastic_params`
    self.scale_factor = 207.64874267578125

  def __call__(self, sample, idx=None):

    # normalize points
    xyz = sample['points']
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
    xyz = (xyz - center) / self.scale_factor  # xyz in [-1, 1]
    
    # construct points
    points = Points(points=torch.from_numpy(xyz),
                    features=torch.from_numpy(sample['colors']), 
                    labels=torch.from_numpy(sample['labels']))

    # transform provided by `ocnn`,
    # including rotatation, translation, scaling, and flipping
    points, inbox_mask = self.transform(points, idx)   # points and inbox_mask

    # align z
    points = align_z(points)
    return {'points': points, 'inbox_mask': inbox_mask}


class CollateBatch(CollateBatch):

  def __init__(self):
    super().__init__()

  def __call__(self, batch: list):
    assert type(batch) == list

    # a list of dicts -> a dict of lists
    outputs = {key: [b[key] for b in batch] for key in batch[0].keys()}

    points = outputs['points']
    outputs['points'] = points
    return outputs


def get_nuscenes_dataset(flags):
  transform = NuscenesTransform(flags)
  read_file = ReadFile(has_normal=False, has_color=True, has_label=True)
  collate_batch = CollateBatch()

  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file)
  return dataset, collate_batch
