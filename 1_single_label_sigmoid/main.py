import argparse

from image_gen import ImageSet, ImagePairMulti
from util import to_excel_sheet
from model import *

if __name__ == '__main__':
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
                        default='Parenchyma,MildInflammation,SevereInflammation,Normal,ConductingAirway', help='output: targets separated by comma')
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
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 128, 192, 192, 256, 256, 384], kernel_size=(3,3), mask_color="white", separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 128, 192, 192, 256, 256], kernel_size=(3,3), mask_color="white", separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 192, 256, 256], kernel_size=(3,3), mask_color="white", separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 192, 256], kernel_size=(3,3), mask_color="white", separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        ModelConfig((512, 512, 3), (512, 512, 1), model_filter=[64, 96, 128, 192, 256], model_kernel=[3,3], mask_color="white", separate=True, coverage_tr=1.5, coverage_prd=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192], kernel_size=(3,3), mask_color="white", separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 128, 256], kernel_size=(3,3), mask_color="white", separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
    ]
    mode = args.mode[0].lower()
    if mode != 'p':
        for cfg in configs:
            for mod in models:
                model= MyModel(mod, cfg, save=False)
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
            for mod in models:
                model= MyModel(mod, cfg, save=False)
                xls_file = "Result_%s_%s.xlsx" % (args.pred_dir, model.name)
                for origin in origins:
                    multi_set = ImagePairMulti(cfg, os.path.join(os.getcwd(), args.pred_dir), origin, None, is_train=False)
                    res_ind = np.zeros((len(multi_set.img_set.images), 0), dtype=np.uint32)
                    res_grp = np.zeros((len(multi_set.img_set.groups), 0), dtype=np.uint32)
                    if cfg.dep_out==1:
                        for i, target in enumerate(targets):
                            multi_set.change_target([target])
                            ind, grp=model.predict(multi_set)
                            res_ind=np.concatenate((res_ind,ind),axis=-1)
                            res_grp=np.concatenate((res_grp,grp),axis=-1)
                    else:
                        multi_set.change_target(targets)
                        ind, grp = model.predict(multi_set)
                        res_ind = np.concatenate((res_ind, ind), axis=-1)
                        res_grp = np.concatenate((res_grp, grp), axis=-1)
                    df=pd.DataFrame(res_ind, index=multi_set.img_set.images, columns=targets)
                    to_excel_sheet(df, xls_file, origin) # per slice
                    if cfg.separate:
                        df=pd.DataFrame(res_grp, index=multi_set.img_set.groups, columns=targets)
                        to_excel_sheet(df, xls_file, origin + "_sum")
                        # to_excel_sheet(df.groupby([vc.image_file for vc in prd_set.view_coord]).sum(), xls_file, origin + "_sum") # simple sum

