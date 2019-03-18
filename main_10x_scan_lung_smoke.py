import argparse
from b1_net_pair import ImageMaskPair
from keras.backend import clear_session

from c1_vgg import NetU_Vgg
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
                        default='tp', help='mode: enter initials from train/test, predict/inference or evaluation (e.g., \'tep\' train->eval->pred)')
    parser.add_argument('-i', '--input', dest='input', type=str,
                        default='Original', help='input: Original')
    parser.add_argument('-o', '--output', dest='output', type=str,
                        default='Background,ConductingAirway,ConnectiveTissue,LargeBloodVessel,RespiratoryAirway,SmallBloodVessel', help='output: targets separated by comma') #,
    args = parser.parse_args() #Background,ConductingAirway,ConnectiveTissue,LargeBloodVessel,RespiratoryAirway,SmallBloodVessel

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
    from c1_unet import UNet2S, UNet2, UNet2m, UNet2M, UNet
    from postprocess import multi_call
    from module import ca3, ca33, sk, ac3, ac33, bac3, bac33, cba3, cba33, aca3, aca33, ct
    from keras.optimizers import SGD
    nets = [
        # SegNet(num_targets=len(targets),target_scale=1.0),
        # SegNetS(num_targets=len(targets),target_scale=1.0),
        # UNet(num_targets=len(targets),target_scale=1.0),
        # UNet2(num_targets=len(targets),target_scale=1.0),
        NetU_Vgg(num_targets=len(targets),target_scale=1.0,predict_proc=multi_call,save_ind_raw=(False,True)),
        NetU_Vgg(num_targets=len(targets),target_scale=1.0,predict_proc=multi_call,save_ind_raw=(True,True),
              overlay_color=[(0,255,0),]*6,overlay_opacity=[0.8]*4+[0.0,0.8],overlay_textshape_bwif=(False,False,False,False)), #BCCLRS green mask
    ]

    for m in args.mode.lower():
        if m in ['t','p','i']:
            for net in nets:
                print("Network specifications: " + str(net))
                for origin in origins:
                    if m=='t': # train/test
                        net.train(ImageMaskPair(net,os.path.join(os.getcwd(),args.train_dir),origin,targets,is_train=True))
                    else: #  m=='p' or 'i' # predict/inference
                        net.predict(ImageMaskPair(net,os.path.join(os.getcwd(),args.pred_dir),origin,targets,is_train=False),args.pred_dir)
                clear_session(); del net.net
        else:
            print("Procedure '%c' not supported, skipped."%m)
