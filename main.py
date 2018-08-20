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
    from unet_pool_up_deep import unet_pool_up_deep_2f2
    from unet.unet_pool_up import unet_pool_up_2f1
    from unet.unet_pool_up_dual import unet_pool_up_dual_2f1
    from unet.unet_pool_up_dual_residual import unet_pool_up_dual_residual_2f1, unet_pool_up_dual_residual_c13_2f1
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
        # unet_pool_up_2f2,
        # unet_pool_up_dual_residual_c13_2f1,
        # unet_pool_up_dual_residual_2f1,
        # unet_pool_up_deep_2f2,
        # unet_vgg_7conv,
        # unet_recursive, # not working
        # DenseNet,
    ]
    configs = [
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 128, 192, 192, 256, 256, 384], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 128, 192, 192, 256, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 192, 256, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 192, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 128, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),

        # ModelConfig((512, 512, 3), (512, 512,6), filter_size=[64, 128, 256, 512, 768], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='softmax', loss_fun='categorical_crossentropy'),
        # ModelConfig((512, 512, 3), (512, 512,6), filter_size=[64, 128, 256, 256, 512, 512, 768, 768], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='softmax', loss_fun='categorical_crossentropy'),
        # ModelConfig((768, 768, 3), (768, 768, 6), filter_size=[64, 128, 256, 512, 768], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='softmax', loss_fun='categorical_crossentropy'),
        # ModelConfig((768, 768, 3), (768, 768, 6), filter_size=[64, 96, 128, 192, 256, 384, 512, 512], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='softmax', loss_fun='categorical_crossentropy'),
        ModelConfig((768, 768, 3), (768, 768, 6), filter_size=[32, 64, 96, 128, 192, 256, 384, 512], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='softmax', loss_fun='categorical_crossentropy'),

        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 192, 256, 256, 384], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024, 1024, 1), filter_size=[64, 96, 128, 192, 256, 256, 384], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 256], kernel_size=(3,1), resize=0.5, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),

        # ModelConfig((572, 572, 3), (572, 572, 1), filter_size=[64, 128, 256, 512, 1024], resize=1.0, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # Vanilla U-Net 572 -> 388 (valid padding, Center 67%), pool_up_2f2 with 64, 128, 256, 512, 1024 filters

        # ModelConfig((1060, 1060, 3), (1060,1060, 1), resize=1.0, separate=True, tr_coverage=0.9 prd_coverage=1.4, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024,1024, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), resize=1.0, padding=1.0, separate=True, tr_coverage=0.9, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192], resize=1.0, padding=1.0, separate=False, tr_coverage=0.9, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192, 256], resize=0.5, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024, 1024, 1), filter_size=[64, 96, 128, 192, 256], resize=0.5, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # last-use ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 256], resize=1.0, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024, 1024, 1), filter_size=[32, 48, 64, 96, 128, 192, 256, 384, 512], resize=0.5, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024, 1024, 1), filter_size=[64, 96, 128, 192, 256, 384], resize=0.5, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192, 256, 384], resize=1.0, padding=1.0, separate=False, tr_coverage=0.9, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192, 256, 384, 512], resize=1.0, padding=1.0, separate=False, tr_coverage=0.9, prd_coverage=2.0, out_fun='sigmoid', loss_fun=loss_bce_dice),
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
                ### pair sigmoid ###
                # for origin in origins:
                #     ori_set=ImageSet(cfg, os.path.join(os.getcwd(), args.train_dir), origin, train=True, filter_type='rgb')
                #     for target in targets:
                #         tgt_set=ImageSet(cfg, os.path.join(os.getcwd(), args.train_dir), target, train=True, filter_type='rgb')  # filter_type=cfg.mask_color
                #         pair=ImagePairTrain(cfg, ori_set, tgt_set)
                #         tr, val = pair.get_tr_val_generator()
                #         model.train(cfg, tr, val)
                ### set softmax ###
                for origin in origins:
                    ori_set=ImageSet(cfg, os.path.join(os.getcwd(), args.train_dir), origin, train=True, filter_type='rgb')
                    tgt_set=[]
                    for target in targets:
                        tgt_set.append(ImageSet(cfg, os.path.join(os.getcwd(), args.train_dir), target, train=True, filter_type='rgb'))  # filter_type=cfg.mask_color
                    multi_set=ImagePairMulti(cfg, ori_set, tgt_set)
                    tr, val = multi_set.get_tr_val_generator()
                    model.train(cfg, tr, val)

    if mode != 't':
        for cfg in configs:
            for mod in models:
                model= MyModel(mod, cfg, save=False)
                xls_file = "Result_%s_%s.xlsx" % (args.pred_dir, model.name)
                ## sigmoid
                # for origin in origins:
                #     prd_set=ImageSet(cfg, os.path.join(os.getcwd(), args.pred_dir), origin, train=False)
                #     pair=ImagePairPredict(cfg, prd_set)
                #     res_ind = np.zeros((len(prd_set.images), len(targets)), dtype=np.uint32)
                #     res_grp = np.zeros((len(prd_set.groups), len(targets)), dtype=np.uint32)
                #     for i, target in enumerate(targets):
                #         pair.change_target(target)
                #         res_ind[..., i], res_grp[...,i]=model.predict(pair)
                #     df=pd.DataFrame(res_ind, index=prd_set.images, columns=targets)
                #     to_excel_sheet(df, xls_file, origin) # per slice
                #     if cfg.separate:
                #         df=pd.DataFrame(res_grp, index=prd_set.groups, columns=targets)
                #         to_excel_sheet(df, xls_file, origin + "_sum")
                #         # to_excel_sheet(df.groupby([vc.image_file for vc in prd_set.view_coord]).sum(), xls_file, origin + "_sum") # simple sum

                ## softmax
                for origin in origins:
                    prd_set=ImageSet(cfg, os.path.join(os.getcwd(), args.pred_dir), origin, train=False)
                    multi_set=ImagePairMulti(cfg, prd_set, ','.join(targets))
                    res_ind = np.zeros((len(prd_set.images), len(targets)), dtype=np.uint32)
                    res_grp = np.zeros((len(prd_set.groups), len(targets)), dtype=np.uint32)
                    res_ind, res_grp =model.predict(multi_set.get_prd_generator())
                    df=pd.DataFrame(res_ind, index=prd_set.images, columns=targets)
                    to_excel_sheet(df, xls_file, origin) # per slice
                    if cfg.separate:
                        df=pd.DataFrame(res_grp, index=prd_set.groups, columns=targets)
                        to_excel_sheet(df, xls_file, origin + "_sum")
                        # to_excel_sheet(df.groupby([vc.image_file for vc in prd_set.view_coord]).sum(), xls_file, origin + "_sum") # simple sum