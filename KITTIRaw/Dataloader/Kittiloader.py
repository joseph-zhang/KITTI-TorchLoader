#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image
from .filldepth import fill_depth_colorization
from .bin2depth import get_depth, get_focal_length_baseline


class Kittiloader(object):
    """
    param kittiDir: KITTI dataset root path, e.g. ~/data/kitti/
    param mode: 'train', 'test' or 'val'
    param cam: camera id. 2 represents the left cam, 3 represents the right one
    """
    def __init__(self, kittiDir, mode, cam=2):
        self.mode = mode
        self.cam = cam
        self.files = []
        self.kitti_root = kittiDir

        # read filenames files
        currpath = os.path.dirname(os.path.realpath(__file__))
        filepath = currpath + '/filenames/eigen_{}_files.txt'.format(self.mode)
        with open(filepath, 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                data_info = data.split(' ')

                self.files.append({
                    "l_rgb": data_info[0],
                    "r_rgb": data_info[1],
                    "cam_intrin": data_info[2],
                    "depth": data_info[3]
                })

    def data_length(self):
        return len(self.files)

    def _check_path(self, filename, err_info):
        file_path = os.path.join(self.kitti_root, filename)
        assert os.path.exists(file_path), err_info
        return file_path

    def _read_data(self, item_files):
        l_rgb_path = self._check_path(item_files['l_rgb'], "Panic::Cannot find Left Image")
        r_rgb_path = self._check_path(item_files['r_rgb'], "Panic::Cannot find Right Image")
        cam_path = self._check_path(item_files['cam_intrin'], "Panic::Cannot find Camera Infos")
        depth_path = self._check_path(item_files['depth'], "Panic::Cannot find depth file")

        l_rgb = Image.open(l_rgb_path).convert('RGB')
        r_rgb = Image.open(r_rgb_path).convert('RGB')
        w, h = l_rgb.size
        focal_length, baseline = get_focal_length_baseline(cam_path, cam=self.cam)
        depth, depth_interp = get_depth(cam_path, depth_path, [h,w], cam=self.cam, interp=True, vel_depth=True)

        data = {}
        data['left_img'] = l_rgb
        data['right_img'] = r_rgb
        data['depth'] = depth.astype(np.float32)
        data['depth_interp'] = depth_interp.astype(np.float32)
        data['fb'] = np.array(focal_length * baseline).astype(np.float32)
        return data

    def load_item(self, idx, interp_method='linear'):
        """
        load an item for training or test
        interp_method can be selected from [linear', 'nyu']
        """
        item_files = self.files[idx]
        data_item = self._read_data(item_files)

        if interp_method == 'nyu':
            image_data = (data_item['left_img'], data_item['right_img'])[self.cam==3]
            image_gray_arr = np.array(image_data.convert('L'))
            data_item['depth_interp'] = fill_depth_colorization(image_gray_arr, data_item['depth'])

        return data_item
