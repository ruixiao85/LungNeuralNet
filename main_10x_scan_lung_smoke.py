import argparse
from b1_net_pair import ImageMaskPair
from osio import *

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
                        default='SmallBloodVessel,Background,ConductingAirway,ConnectiveTissue,LargeBloodVessel,RespiratoryAirway', help='output: targets separated by comma')
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
    from c1_unet import UNet2S

    nets = [
        # SegNet(num_targets=len(targets)),
        # SegNetS(num_targets=len(targets)),
        # UNet(num_targets=len(targets)),
        # UNet2(num_targets=len(targets)),
        # UNet(num_targets=len(targets)),
        UNet2S(num_targets=len(targets)),
        # UNet2M(num_targets=len(targets)),
        # UNet2L(num_targets=len(targets)),
        # VggSegNet(num_targets=len(targets)),
        # Refine(num_targets=len(targets))
    ]

    mode = args.mode[0].lower()
    if mode != 'p':
        for net in nets:
            print("Network specifications: " + str(net))
            for origin in origins:
                multi_set = ImageMaskPair(net, os.path.join(os.getcwd(), args.train_dir), origin, targets, is_train=True)
                net.train(multi_set)

    if mode != 't':
        for net in nets:
            for origin in origins:
                multi_set = ImageMaskPair(net,os.path.join(os.getcwd(),args.pred_dir),origin,targets,is_train=False)
                net.predict(multi_set, args.pred_dir)
