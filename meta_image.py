
from scipy.misc import imresize
import math
import numpy as np

class MetaImage:
    def __init__(self):
        self.image=None
        self.ori_row, self.ori_col, self.ori_dep=None,None,None
        self.tiles=None
        self.meta=None

    def __init__(self,file,img,cfg):
        # ori, resize, pad, row, col, dep, full = True
        self.image = imresize(img, cfg.resize) if cfg.resize is not None and cfg.resize!=1. else img
        if cfg.pad is not None and cfg.pad>0:
            self.image=np.pad(self.image,((cfg.pad, cfg.pad), (cfg.pad, cfg.pad), (0, 0)),mode='reflect')
        self.ori_row, self.ori_col, self.ori_dep = self.image.shape
        if cfg.full:
            r_len = int(math.ceil((self.ori_row - 2 * cfg.pad) / float(cfg.row - 2 * cfg.pad)))*2
            c_len = int(math.ceil((self.ori_col - 2 * cfg.pad) / float(cfg.col - 2 * cfg.pad)))*2
        else:
            r_len = int(math.floor((self.ori_row - 2 * cfg.pad) / float(cfg.row - 2 * cfg.pad)))
            c_len = int(math.floor((self.ori_col - 2 * cfg.pad) / float(cfg.col - 2 * cfg.pad)))
        print("%s target %d x %d (pad %d full %r): original %d x %d x %d ->  row /%d col /%d" %
              (file, cfg.row, cfg.col, cfg.pad, cfg.full, self.ori_row, self.ori_col, self.ori_dep, r_len, c_len))
        self.tiles = np.empty((0, cfg.row, cfg.col, self.ori_dep), dtype=np.uint8)
        r_step = float(self.ori_row - cfg.row) / (r_len - 1) if r_len > 1 else 0
        c_step = float(self.ori_col - cfg.col) / (c_len - 1) if c_len > 1 else 0
        self.meta = []
        p_index = 0
        for r_index in range(r_len):
            for c_index in range(c_len):
                ri = int(round(r_index * r_step))
                ci = int(round(c_index * c_step))
                entry=MetaInfo(file, p_index, self.ori_row, self.ori_col, self.ori_dep, cfg.row, cfg.col, cfg.row, cfg.col, ri, ci)
                self.meta.append(entry)
                self.tiles = np.append(self.tiles, self.image[ri:ri + cfg.row, ci:ci + cfg.col, ...][np.newaxis,...], axis=0)  # reference view
                p_index += 1

