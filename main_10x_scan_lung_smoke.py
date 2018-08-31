import argparse

from model import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train and predict with biomedical images.')
    parser.add_argument('-d', '--dir', dest='dir', action='store',
                        default='10x_scan_lung_smoke', help='work directory, empty->current dir')
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
                        default='*.jpg', help='extension')
    parser.add_argument('-i', '--input', dest='input', type=str,
                        default='Original', help='input: Original')
    parser.add_argument('-o', '--output', dest='output', type=str,
                        default='ConductingAirway,SmallBloodVessel', help='output: targets separated by comma')
    #Background,ConductingAirway,ConnectiveTissue,LargeBloodVessel,RespiratoryAirway,SmallBloodVessel
    args = parser.parse_args()

    script_dir = os.path.realpath(__file__)
    rel_dir = os.path.join(script_dir, args.dir)
    if os.path.exists(args.dir):
        os.chdir(args.dir)
    elif os.path.exists(rel_dir):
        os.chdir(rel_dir)
    else:
        os.chdir(script_dir)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # force cpu
    origins = args.input.split(',')
    targets = args.output.split(',')
    from unet.unetflex import unet1s, unet1d, unet2s,  ca3, ca33, dmp, dca, uuc, utc
    from unet.unetflex import du32, du33
    from unet.unetflex import rn33r, rn33nr, rn131r, rn131nr
    from unet.unetflex import dn13r, dn13nr
    configs = [
        # ModelConfig((1296, 1296, 3), (1296, 1296, 1), num_targets=len(targets), model_filter=[64, 64, 96, 96, 128, 128], model_pool=[2, 2, 3, 3, 3, 3],
        #             model_name=unet1d, model_preconv=c7m3d4, model_downconv=ca33, model_downsamp=dmp, model_upsamp=uuc, model_upconv=ca33, train_rep=10),

        # ModelConfig((512, 512, 3), (512, 512, 1), num_targets=len(targets), model_filter=[64, 64, 96, 96, 128, 128, 128], model_pool=[2, 2, 2, 2, 2, 2, 2],
        #             model_name=unet1d, model_downconv=ca33, model_downsamp=dmp, model_upsamp=uuc, model_upconv=ca33, train_rep=3),

        ModelConfig((512, 512, 3), (512, 512, 1), num_targets=len(targets), model_filter=[96, 96, 96, 96, 96, 96, 96], model_pool=[2, 2, 2, 2, 2, 2, 2],
                    model_name=unet1d, model_downconv=du32, model_downsamp=dmp, model_upsamp=uuc, model_upconv=du33, train_rep=3), # Depp U Net

        ModelConfig((512, 512, 3), (512, 512, 1), num_targets=len(targets), model_filter=[64, 64, 64, 64, 64, 64, 64], model_pool=[2, 2, 2, 2, 2, 2, 2],
                    model_name=unet1d, model_downconv=rn33r, model_downsamp=dmp, model_upsamp=uuc, model_upconv=rn33r, train_rep=3), # Res Net

        ModelConfig((512, 512, 3), (512, 512, 1), num_targets=len(targets), model_filter=[16, 16, 16, 16, 16, 16, 16], model_pool=[2, 2, 2, 2, 2, 2, 2],
                    model_name=unet1d, model_downconv=dn13r, model_downsamp=dmp, model_upsamp=uuc, model_upconv=dn13r, train_rep=3), # Dense Net

    ]
    mode = args.mode[0].lower()
    if mode != 'p':
        for cfg in configs:
            model= MyModel(cfg, save=False)
            print("Network specifications: " + str(cfg))
            for origin in origins:
                multi_set = ImagePair(cfg, os.path.join(os.getcwd(), args.train_dir), origin, targets, is_train=True)
                model.train(cfg, multi_set)

    if mode != 't':
        for cfg in configs:
            model= MyModel(cfg, save=False)
            xls_file = "Result_%s_%s.xlsx" % (args.pred_dir, cfg)
            for origin in origins:
                multi_set = ImagePair(cfg, os.path.join(os.getcwd(), args.pred_dir), origin, targets, is_train=False)
                model.predict(multi_set, xls_file)
