import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from typing import List

from eval import verification_consine
import backbones
import losses
from config import config as config
from backbones.iresnet import iresnet100

class CallBackVerification(object):
    def __init__(self, frequent, rank, val_targets, rec_prefix, image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification_consine.test(
                self.ver_list[i], backbone, 10, 10)
            print('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            print(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification_consine.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, backbone: torch.nn.Module):
        self.ver_test(backbone, 0)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = iresnet100().to(device)
    cur_path = os.path.dirname(os.path.abspath(__file__))
    backbone_pth = os.path.join(
        cur_path, "ms1mv3_arcface_r100_fp16/backbone.pth")
    model.load_state_dict(torch.load(backbone_pth))
    model.eval()
    callback_verification = CallBackVerification(2000, 0, ["lfw", "cfp_fp", "agedb_30"], "/media/mengchao/dataset/ms1m-retinaface-t1")
    callback_verification(model)

if __name__ == "__main__":
    main()
