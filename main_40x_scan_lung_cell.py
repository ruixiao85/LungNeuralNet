import os
import argparse
from b2_net_multi import ImagePatchPair
from keras.backend import clear_session

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train and predict with biomedical images.')
    parser.add_argument('-d', '--dir', dest='dir', action='store',
                        # default='10x_pizz mmp13 ko liver', help='work directory, empty->current dir')
                        default='40x_scan_lung_cell', help='work directory, empty->current dir')
    parser.add_argument('-t', '--train', dest='train_dir', action='store',
                        default='train', help='train sub-directory')
    parser.add_argument('-p', '--pred', dest='pred_dir', action='store',
                        default='pred', help='predict sub-directory')
    parser.add_argument('-m', '--mode', dest='mode', action='store',
                        default='p', help='mode: enter initials from train/test, predict/inference or evaluation (e.g., \'tep\' train->eval->pred)')
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
    from c2_backbones import  MRCNN_Vgg16,MRCNN_Vgg19,MRCNN_Res50,MRCNN_Dense121,MRCNN_Dense169,MRCNN_Dense201,MRCNN_Mobile

    nets = [
        # MRCNN_Vgg16(num_targets=len(targets),target_scale=1.0,coverage_train=3.0,coverage_predict=1.5,dim_in=(768,768,3),dim_out=(768,768,3)),
        # MRCNN_Res50(num_targets=len(targets),target_scale=1.0,coverage_train=3.0,coverage_predict=1.5,dim_in=(768,768,3),dim_out=(768,768,3)),
        MRCNN_Dense169(num_targets=len(targets),target_scale=2.0,coverage_train=3.0,coverage_predict=1.5,dim_in=(768,768,3),dim_out=(768,768,3)),
        # MRCNN_Dense169(num_targets=len(targets),target_scale=4.0,coverage_train=3.0,coverage_predict=1.5,dim_in=(768,768,3),dim_out=(768,768,3)),
    ]
    for m in args.mode.lower():
        if m in ['t','e','p','i']:
            for net in nets:
                print("Network specifications: " + str(net))
                for origin in origins:
                    if m=='t': # train/test
                        net.train(ImagePatchPair(net,os.path.join(os.getcwd(),args.train_dir),origin,targets,is_train=True))
                    elif m=='e': # evaluation mAP
                        net.eval(ImagePatchPair(net,os.path.join(os.getcwd(),args.train_dir),origin,targets,is_train=True))
                    else: #  m=='p' or 'i' # predict/inference
                        net.predict(ImagePatchPair(net,os.path.join(os.getcwd(),args.pred_dir),origin,targets,is_train=False),args.pred_dir)
                clear_session() ; del net.net
        else:
            print("Procedure '%c' not supported, skipped."%m)
