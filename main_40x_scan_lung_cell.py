import os
import argparse
from b2_net_multi import ImagePatchPair

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train and predict with biomedical images.')
    parser.add_argument('-d', '--dir', dest='dir', action='store',
                        default='40x_scan_lung_cell', help='work directory, empty->current dir')
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
                        default='LYM,MONO,PMN', help='output: targets separated by comma')
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
    from c2_resnet import MRCNN_ResNet_50, MRCNN_ResNet_101, MRCNN_ResNet_152

    nets = [
        # UNet2m(num_targets=len(targets),dim_in=(768,768,3),dim_out=(768,768,3),filters=[96, 128, 256, 512, 768],out_image=True,
        #        out='sigmoid',indicator='val_pl1mix',loss=loss_pmse,metrics=[pl1mix],
        #        predict_proc=compare_call),
        # UNet2m(num_targets=len(targets),dim_in=(768,768,3),dim_out=(768,768,1),filters=[96, 192, 288, 384, 512],poolings=[2, 2, 2, 2, 2]), #
        # UNet2m(num_targets=len(targets),predict_proc=single_brighten,coverage_tr=1.2,coverage_prd=1.2),
        MRCNN_ResNet_50(num_targets=len(targets))
    ]

    mode = args.mode[0].lower()
    if mode != 'p':
        for net in nets:
            print("Network specifications: " + str(net))
            for origin in origins:
                multi_set=ImagePatchPair(net, os.path.join(os.getcwd(), args.train_dir), origin, targets, is_train=True); net.train(multi_set)

    if mode != 't':
        for net in nets:
            for origin in origins:
                multi_set=ImagePatchPair(net, os.path.join(os.getcwd(), args.pred_dir), origin, targets, is_train=False); net.predict(multi_set, args.pred_dir)
