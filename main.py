import argparse

from image_gen import ImageSet, ImagePairMulti
from util import to_excel_sheet
from model import *

if __name__ == '__main__':
    # -d "D:\Cel files\2018-07.13 Adam Brenderia 2X LPS CGS" -t "071318 Cleaned 24H post cgs" -p "2018-07.20 Kyle MMP13 Smoke Flu Zander 2X" -o "Paren,InflamMild,InflamSevere"
    # -d "I:/NonParen" -t "Rui Xiao 2017-09-07 (14332)(27)" -p "Rui Xiao 2017-12-05 (15634)(9)" -o "Background,ConductingAirway,ConnectiveTissue,LargeBloodVessel,RespiratoryAirway,SmallBloodVessel" -m a
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
    from densenet.dn121 import DenseNet
    from unet_pool_up_resd import unet_pool_up_deep_2f2
    from unet.unet_pool_up import unet_pool_up_2f1
    from unet.unet_pool_up_dual import unet_pool_up_dual_2f1
    from unet.unet_pool_up_dual_residual import unet_pool_up_dual_residual_2f1, unet_pool_up_dual_residual_c13_2f1
    from unet.unet_pool_up_resf import unet_pool_up_res_1f1, unet_pool_up_res_2f1, unet_pool_up_res_2f2
    models = [
        # unet_conv_trans_1f1,
        # unet_conv_trans_2f1,
        # unet_conv_trans_2f2,
        # unet_pool_trans_1f1,
        # unet_pool_trans_2f1,
        # unet_pool_trans_2f2,
        # unet_pool_up_dual_2f1, #very good
        # unet_pool_up_1f1,
        unet_pool_up_2f1,
        unet_pool_up_res_2f1,
        # unet_pool_up_2f2,
        # unet_pool_up_dual_residual_c13_2f1,
        # unet_pool_up_dual_residual_2f1,
        # unet_pool_up_deep_2f2,
        # unet_vgg_7conv,
        # unet_recursive, # not working
        # DenseNet,
    ]
    configs = [
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 128, 192, 192, 256, 256, 384], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 128, 192, 192, 256, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 192, 256, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 192, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 128, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),


        # ModelConfig((768, 768, 3), (768, 768, 6), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        # ModelConfig((512, 512, 3), (512, 512, 6), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        # ModelConfig((768, 768, 3), (768, 768, 6), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.2, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        # ModelConfig((512, 512, 3), (512, 512, 6), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.2, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),

        ModelConfig((2048, 2048, 3), (2048, 2048, 1), model_filter=[20,28,40,57,81,115,163,231,327,462], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        ModelConfig((768, 768, 3), (768, 768, 1), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        ModelConfig((512, 512, 3), (512, 512, 1), model_filter=[32, 64, 96, 128, 192], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        # ModelConfig((768, 768, 3), (768, 768, 1), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.2, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        # ModelConfig((512, 512, 3), (512, 512, 1), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.2, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),

        # ModelConfig((768, 768, 3), (768, 768, 1), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        # ModelConfig((512, 512, 3), (512, 512, 1), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        # ModelConfig((768, 768, 3), (768, 768, 1), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.2, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        # ModelConfig((512, 512, 3), (512, 512, 1), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.2, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),

        # ModelConfig((768, 768, 3), (768, 768, 6), model_filter=[8, 16, 32, 64, 128, 256, 512, 1024], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0, model_out='softmax', model_loss='categorical_crossentropy'),
        # ModelConfig((768, 768, 3), (768, 768, 6), model_filter=[32, 64, 128, 256, 512, 1024, 2048, 4096], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0, model_out='softmax', model_loss='categorical_crossentropy'),
        # ModelConfig((768, 768, 3), (768, 768, 6), model_filter=[64,80,100,125,156,195,244,305], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0, model_out='softmax', model_loss='categorical_crossentropy'),
        # ModelConfig((768, 768, 3), (768, 768, 6), model_filter=[80,95,113,134,159,189,225,268], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0, model_out='softmax', model_loss='categorical_crossentropy'),

        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 192, 256, 256, 384], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024, 1024, 1), filter_size=[64, 96, 128, 192, 256, 256, 384], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 256], kernel_size=(3,1), resize=0.5, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),

        # ModelConfig((572, 572, 3), (572, 572, 1), filter_size=[64, 128, 256, 512, 1024], resize=1.0, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # Vanilla U-Net 572 -> 388 (valid padding, Center 67%), pool_up_2f2 with 64, 128, 256, 512, 1024 filters

        # ModelConfig((1060, 1060, 3), (1060,1060, 1), resize=1.0, separate=True, tr_coverage=0.9 prd_coverage=1.4, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024,1024, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), resize=1.0, padding=1.0, separate=True, tr_coverage=0.9, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192], resize=1.0, padding=1.0, separate=False, tr_coverage=0.9, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192, 256], resize=0.5, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024, 1024, 1), filter_size=[64, 96, 128, 192, 256], resize=0.5, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # last-use ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 256], resize=1.0, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024, 1024, 1), filter_size=[32, 48, 64, 96, 128, 192, 256, 384, 512], resize=0.5, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024, 1024, 1), filter_size=[64, 96, 128, 192, 256, 384], resize=0.5, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192, 256, 384], resize=1.0, padding=1.0, separate=False, tr_coverage=0.9, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192, 256, 384, 512], resize=1.0, padding=1.0, separate=False, tr_coverage=0.9, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((256, 256, 3), (256, 256, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((768, 768, 3), (674, 674, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, model_out='sigmoid', model_loss=loss_bce_dice),  # 5 valid
    ]
    mode = args.mode[0].lower()
    if mode != 'p':
        for cfg in configs:
            model= MyModel(cfg, save=False)
            print("Network specifications: " + model.name.replace("_", " "))
            for origin in origins:
                if cfg.dep_out==1:
                    for target in targets:
                        multi_set = ImagePairMulti(cfg, os.path.join(os.getcwd(), args.train_dir), origin, [target], is_train=True)
                        model.train(cfg, multi_set)
                else:
                    multi_set = ImagePairMulti(cfg, os.path.join(os.getcwd(), args.train_dir), origin, targets, is_train=True)
                    model.train(cfg, multi_set)

    if mode != 't':
        for cfg in configs:
            model= MyModel(cfg, save=False)
            xls_file = "Result_%s_%s.xlsx" % (args.pred_dir, model.name)
            for origin in origins:
                if cfg.dep_out==1:
                    for target in targets:
                        multi_set = ImagePairMulti(cfg, os.path.join(os.getcwd(), args.pred_dir), origin, [target], is_train=False)
                        model.predict(multi_set, xls_file)
                else:
                    multi_set = ImagePairMulti(cfg, os.path.join(os.getcwd(), args.pred_dir), origin, targets, is_train=False)
                    model.predict(multi_set, xls_file)
