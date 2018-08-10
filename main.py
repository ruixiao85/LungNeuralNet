import os
import argparse
import pandas as pd
from PIL import ImageDraw, Image, ImageFont
from skimage.io import imsave, imread
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from image_gen import ImageTrainPair, ImageSet, ImagePredictPair
from process_image import scale_input, augment_image_pair, scale_output
from tensorboard_train_val import TensorBoardTrainVal
from model_config import ModelConfig
from util import mk_dir_if_nonexist



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
    from unet.unet_pool_up_31 import unet_pool_up_5_dure
    from unet.unet_pool_up_valid import unet_pool_up_5_valid
    models = [
        # unet_pool_trans_5,
        # unet_pool_trans_7,
        # unet_conv_trans_5,
        # unet_conv_trans_7,
        # unet_pool_up_5,
        # unet_pool_up_7,
        unet_pool_up_5_dure,
        # unet_pool_up_6_dure,
        # unet_pool_up_7_dure,
        # unet_pool_up_5_valid,
        # unet_vgg_7conv,
    ]
    from unet.my_model import *
    configs = [
        ModelConfig((1024, 1024, 3), (1024,1024, 1), resize=1.0, padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), resize=1.0, padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),
        # config((768, 768, 3), (674, 674, 1), resize=1.0, padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),  # 5 valid
        # config(256, 256, 3, 1, resize=1., padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),
        # config(256, 256, 3, 1, resize=1., padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),
        # config(768, 768, 3, 1, resize=1.0, padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),
        # config(512, 512, 3, 1, resize=0.8, padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),
        # config(256, 256, 3, 1, resize=1., padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),
        # config(3, 1, 'elu', 'sigmoid', loss_dice),
        # config(3, 1, 'elu', 'sigmoid', 'binary_crossentropy'), # mid
        # config(3, 1, 'elu', 'sigmoid', loss_bce), # no good
        # config(3, 1, 'elu', 'sigmoid', loss_jaccard), # no good
        # config(3, 2, 'relu', 'softmax', 'categorical_crossentropy'),
        # config(3, 2, 'elu', 'softmax'),
        # config(3, 2, 'tanh', 'softmax', 'categorical_crossentropy'),
        # config(3, 2, 'softsign', 'softmax', 'categorical_crossentropy'),  # trouble
        # config(3, 2, 'selu', 'softmax', 'categorical_crossentropy'),  # trouble
        # config(3, 2, 'softplus', 'softmax', 'categorical_crossentropy'), # trouble
    ]
    mode = args.mode[0].lower()
    if mode != 'p':
        for cfg in configs:
            cfg.full = False  # less overlap
            for mod in models:
                model= MyModel(mod, cfg, save=False)
                print("Network specifications: " + model.name.replace("_", " "))
                for origin in origins:
                    ori_set=ImageSet(cfg, os.path.join(os.getcwd(), args.train_dir), origin)
                    for target in targets:
                        tgt_set= ImageSet(cfg, os.path.join(os.getcwd(), args.train_dir), target)
                        pair=ImageTrainPair(cfg, ori_set, tgt_set)
                        model.train(pair)

    if mode != 't':
        for cfg in configs:
            cfg.full = True  # insure coverage
            for mod in models:
                model= MyModel(mod, cfg, save=False)
                for origin in origins:
                    prd_set=ImageSet(cfg, os.path.join(os.getcwd(), args.pred_dir), origin)
                    for target in targets:
                        pair=ImagePredictPair(cfg, prd_set, target)
                        model.predict(pair)
                    # res = np.zeros((len(tst.view_coord), len(targets)), np.uint32)
                    # for x, target in enumerate(targets):
                    #     predict(target + nn, res[:, x])
                    # res_df.to_csv("result_" + args.pred_dir + "_" + nn + ".csv")
                    # res_df = pd.DataFrame(res,map(str,tst.view_coord), targets)
                    # xls_file = "Result_" + args.pred_dir + "_" + nn + ".xlsx"
                    # res_df.to_excel(xls_file, sheet_name = 'Individual')
                    # append_excel_sheet(res_df.groupby(res_df.index).sum(),xls_file,"Whole")

