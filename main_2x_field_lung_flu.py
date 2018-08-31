import argparse

from model import *
from unetflex import unet1d, conv33, dmax, upool, conv3, conv331, conv31

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train and predict with biomedical images.')
    parser.add_argument('-d', '--dir', dest='dir', action='store',
                        default='2x_field_lung_flu', help='work directory, empty->current dir')
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
                        default='Parenchyma,SevereInflammation', help='output: targets separated by comma')
    args = parser.parse_args()

    script_dir=os.path.realpath(__file__)
    rel_dir=os.path.join(script_dir, args.dir)
    if os.path.exists(args.dir):
        os.chdir(args.dir)
    elif os.path.exists(rel_dir):
        os.chdir(rel_dir)
    else:
        os.chdir(script_dir)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # force cpu
    origins = args.input.split(',')
    targets = args.output.split(',')
    configs = [
        ModelConfig((512, 512, 3), (512,512, 1), num_targets=len(targets), model_filter=[32, 48, 64, 96, 128, 192, 224, 256],
                    model_name=unet1d, model_poolsize=[2, 2, 2, 2, 2, 2, 2, 2],  # predict_size=1,
                    model_downconv=conv33, model_downsamp=dmax, model_upsamp=upool, model_upconv=conv3),
    ]
    mode = args.mode[0].lower()
    if mode != 'p':
        for cfg in configs:
            model = MyModel(cfg, save=False)
            print("Network specifications: " + str(cfg))
            for origin in origins:
                multi_set = ImagePair(cfg, os.path.join(os.getcwd(), args.train_dir), origin, targets, is_train=True)
                model.train(cfg, multi_set)

    if mode != 't':
        for cfg in configs:
            model = MyModel(cfg, save=False)
            xls_file = "Result_%s_%s.xlsx" % (args.pred_dir, cfg)
            for origin in origins:
                multi_set = ImagePair(cfg, os.path.join(os.getcwd(), args.pred_dir), origin, targets, is_train=False)
                model.predict(multi_set, xls_file)