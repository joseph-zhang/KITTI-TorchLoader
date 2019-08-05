#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image
from .interpd import interpdepth
from .filldepth import fill_depth_colorization

class Kittiloader(object):
    """
    param kittiDir: KITTI dataset root path, e.g. ~/data/KITTI/
    param mode: 'train' or 'test'
    param cam: camera id. 2 represents the left cam, 3 represents the right one
    """
    def __init__(self, kittiDir, mode, cam=2):
        self.mode = mode
        self.cam = cam
        self.files = []
        self.shared_idx = []
        self.kitti_root = kittiDir

        # read filenames files
        currpath = os.path.dirname(os.path.realpath(__file__))
        filepath = currpath + '/filenames/eigen_{}_files.txt'.format(self.mode)
        shared_path = currpath + '/filenames/eigen692_652_shared_index.txt'

        with open(filepath, 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                data_info = data.split(' ')

                assert cam == 2 or cam == 3, "Panic::Param 'cam' should be 2 or 3"
                data_idx_select = (0, 1)[cam==3]

                self.files.append({
                    "rgb": data_info[data_idx_select],
                    "depth": data_info[data_idx_select+2]
                })

        with open(shared_path, 'r') as f:
            shared_list = f.read().split('\n')
            for item in shared_list:
                if len(item) == 0:
                    continue
                self.shared_idx.append(int(item))

    def shared_index(self):
        return self.shared_idx

    def data_length(self):
        return len(self.files)

    def _check_path(self, filename, err_info):
        file_path = os.path.join(self.kitti_root, filename)
        assert os.path.exists(file_path), err_info
        return file_path

    def _read_depth(self, depth_path):
        # (copy from kitti devkit)
        # loads depth map D from png file
        # and returns it as a numpy array,

        depth_png = np.array(Image.open(depth_path), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert(np.max(depth_png) > 255)

        depth = depth_png.astype(np.float32) / 256.
        #depth[depth_png == 0] = -1.
        return depth

    def _read_data(self, item_files, interp_method):
        rgb_path = self._check_path(item_files['rgb'], "Panic::Cannot find RGB Image")
        depth_path = self._check_path(item_files['depth'], "Panic::Cannot find depth file")

        rgb = Image.open(rgb_path).convert('RGB')
        depth = self._read_depth(depth_path)

        data = {}
        data['img'] = rgb
        data['depth'] = depth

        if interp_method in ['nop', 'linear', 'nyu']:
            if interp_method == 'linear':
                data['depth_interp'] = interpdepth(depth)
            elif interp_method == 'nyu':
                image_data = rgb.convert('L')
                image_gray_arr = np.array(image_data)
                data['depth_interp'] = fill_depth_colorization(image_gray_arr, depth)
            else:
                pass
        else:
            raise ValueError("Panic::Invalid 'interp_method' parameter")

        return data

    def load_item(self, idx, interp_method='nop'):
        """
        load an item for training or test
        interp_method can be selected from ['nop', 'linear', 'nyu']
        """
        item_files = self.files[idx]
        data_item = self._read_data(item_files, interp_method)
        return data_item
