import os
import argparse
import pandas as pd

from image_gen import ImageTrainPair, ImageSet, ImagePredictPair
from process_image import scale_input, augment_image_pair, scale_output
from tensorboard_train_val import TensorBoardTrainVal
from model_config import ModelConfig
from util import mk_dir_if_nonexist, to_excel_sheet
from unet.my_model import *

if __name__ == '__main__':
    # -d "D:\Cel files\2018-07.13 Adam Brenderia 2X LPS CGS" -t "071318 Cleaned 24H post cgs" -p "2018-07.20 Kyle MMP13 Smoke Flu Zander 2X" -o "Paren,InflamMild,InflamSevere"
    # -d "I:/NonParen" -t "Rui Xiao 2017-09-07 (14332)(27)" -p "Rui Xiao 2017-12-05 (15634)(9)" -o "NonParen"
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', '--dir', dest='dir', action='store',
                        default='', help='work directory, empty->current dir')
    parser.add_argument('-t', '--train', dest='train_dir', action='store',
                        default='train', help='train sub-directory')
    parser.add_argument('-p', '--pred', dest='pred_dir', action='store',
                        default='pred', help='predict sub-directory')
    parser.add_argument('-m', '--mode', dest='mode', action='store',
                        default='both', help='mode: train pred both')
    parser.add_argument('-c', '--width', dest='width', type=int,
                        default='512', help='width/columns')
    parser.add_argument('-r', '--height', dest='height', type=int,
                        default='512', help='height/rows')
    parser.add_argument('-e', '--ext', dest='ext', action='store',
                        default='jpg', help='extension')
    parser.add_argument('-i', '--input', dest='input', type=str,
                        default='Original', help='input: Original')
    parser.add_argument('-o', '--output', dest='output', type=str,
                        default='Paren,InflamMild,InflamSevere', help='output: targets separated by comma')
    args = parser.parse_args()

    os.chdir(os.getcwd() if (args.dir == '') else args.dir)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # force cpu
    origins = args.input.split(',')
    targets = args.output.split(',')
    from unet.unet_conv_trans import unet_conv_trans_1f1, unet_conv_trans_2f1, unet_conv_trans_2f2
    from unet.unet_pool_trans import unet_pool_trans_1f1, unet_pool_trans_2f1, unet_pool_trans_2f2
    from unet.unet_pool_up import unet_pool_up_1f1, unet_pool_up_2f1, unet_pool_up_2f2
    models = [
        # unet_conv_trans_1f1,
        # unet_conv_trans_2f1,
        # unet_conv_trans_2f2,
        # unet_pool_trans_1f1,
        # unet_pool_trans_2f1,
        # unet_pool_trans_2f2,
        # unet_pool_up_1f1,
        unet_pool_up_2f1,
        # unet_pool_up_2f2,
        # unet_vgg_7conv,
    ]
    configs = [
        # ModelConfig((1060, 1060, 3), (1060,1060, 1), resize=1.0, separate=True, tr_coverage=0.9 prd_coverage=1.4, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024,1024, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), resize=1.0, padding=1.0, separate=True, tr_coverage=0.9, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192], resize=1.0, padding=1.0, separate=True, tr_coverage=0.9, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192, 256], resize=1.0, padding=1.0, separate=True, tr_coverage=0.9, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192, 256, 384], resize=1.0, padding=1.0, separate=True, tr_coverage=0.9, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192, 256, 384, 512], resize=1.0, padding=1.0, separate=True, tr_coverage=0.9, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((256, 256, 3), (256, 256, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((768, 768, 3), (674, 674, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, out_fun='sigmoid', loss_fun=loss_bce_dice),  # 5 valid
    ]
    mode = args.mode[0].lower()
    if mode != 'p':
        for cfg in configs:
            for mod in models:
                model= MyModel(mod, cfg, save=False)
                print("Network specifications: " + model.name.replace("_", " "))
                for origin in origins:
                    ori_set=ImageSet(cfg, os.path.join(os.getcwd(), args.train_dir), origin, train=True)
                    for target in targets:
                        tgt_set=ImageSet(cfg, os.path.join(os.getcwd(), args.train_dir), target, train=True)
                        pair=ImageTrainPair(cfg, ori_set, tgt_set)
                        model.train(pair)
    if mode != 't':
        for cfg in configs:
            for mod in models:
                model= MyModel(mod, cfg, save=False)
                xls_file = "Result_%s_%s.xlsx" % (args.pred_dir, model.name)
                for origin in origins:
                    prd_set=ImageSet(cfg, os.path.join(os.getcwd(), args.pred_dir), origin, train=False)
                    pair=ImagePredictPair(cfg, prd_set)
                    res_ind = np.zeros((len(prd_set.images), len(targets)), dtype=np.uint32)
                    res_grp = np.zeros((len(prd_set.groups), len(targets)), dtype=np.uint32)
                    for i, target in enumerate(targets):
                        pair.change_target(target)
                        res_ind[..., i], res_grp[...,i]=model.predict(pair)
                    df=pd.DataFrame(res_ind, index=prd_set.images, columns=targets)
                    to_excel_sheet(df, xls_file, origin) # per slice
                    if cfg.separate:
                        df=pd.DataFrame(res_grp, index=prd_set.groups, columns=targets)
                        to_excel_sheet(df, xls_file, origin + "_sum")
                        # to_excel_sheet(df.groupby([vc.image_file for vc in prd_set.view_coord]).sum(), xls_file, origin + "_sum") # simple sum

