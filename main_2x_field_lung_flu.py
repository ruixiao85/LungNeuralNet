import argparse

from model import *

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
    from net.unet import UNet, UNet2, SegNet
    nets=[
        # SegNet(num_targets=len(targets), predict_all_inclusive=False),
        # UNet(num_targets=len(targets), predict_all_inclusive=False),
        UNet2(num_targets=len(targets), predict_all_inclusive=False),
    ]
    mode = args.mode[0].lower()
    if mode!='p':
        for net in nets:
            model=Model(net)
            print("Network specifications: "+str(net))
            for origin in origins:
                multi_set=ImagePair(net, os.path.join(os.getcwd(), args.train_dir), origin, targets, is_train=True)
                model.train(multi_set)

    if mode!='t':
        for net in nets:
            model=Model(net)
            for origin in origins:
                multi_set=ImagePair(net, os.path.join(os.getcwd(), args.pred_dir), origin, targets, is_train=False)
                model.predict(multi_set, args.pred_dir)
