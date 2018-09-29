import argparse

from image_gen import ImageMaskPair, ImageNoisePair
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
                        default='InflammatoryCell', help='output: targets separated by comma')
    #Background,ConductingAirway,ConnectiveTissue,LargeBloodVessel,RespiratoryAirway,SmallBloodVessel
    #InflammatoryCell
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
    from net.unet import SegNet, SegNetS, UNet, UNetS, UNet2, UNet2S, UNet2M, UNet2L, ResN131S, ResBN131S, UNet2m
    from net.refinenet import Refine
    from net.vgg import VggSegNet
    from net.module import ca1, ca2, ca3, ca3h, cadh, ca33, ca13, cba3, cb3, dmp, dca, uu, ut, uta, sk, ct,\
      du32, cdu33, rn131r, rn131nr, dn13r, dn13nr
    from metrics import loss_pmse, loss_pmae, loss_pmul, loss_padd, pmse, prmse, pmae, pl1mix
    from model import single_call,multi_call,compare_call
    from keras.optimizers import Adam, SGD, RMSprop, Nadam
    nets = [
        # SegNet(num_targets=len(targets)),
        # SegNetS(num_targets=len(targets)),
        # UNet(num_targets=len(targets)),
        # UNet2(num_targets=len(targets)),
        # UNet(num_targets=len(targets)),
        # UNet2S(num_targets=len(targets)),
        # UNet2M(num_targets=len(targets)),
        # UNet2L(num_targets=len(targets)),
        # VggSegNet(num_targets=len(targets)),
        # Refine(num_targets=len(targets))

        # UNet2m(num_targets=len(targets),dim_in=(768,768,3),dim_out=(768,768,3),filters=[96, 128, 256, 512, 768],out_image=True,
        #        out='sigmoid',indicator='val_pl1mix',loss=loss_pmse,metrics=[pl1mix],
        #        predict_proc=compare_call),
        # UNet2m(num_targets=len(targets),dim_in=(768,768,3),dim_out=(768,768,1),filters=[96, 192, 288, 384, 512],poolings=[2, 2, 2, 2, 2]), #
        UNet2m(num_targets=len(targets)), #
    ]

    mode = args.mode[0].lower()
    if mode != 'p':
        for model in [Model(n) for n in nets]:
            print("Network specifications: " + str(model))
            for origin in origins:
                # multi_set = ImageMaskPair(model.net, os.path.join(os.getcwd(), args.train_dir), origin, targets, is_train=True)
                # model.train(multi_set)

                for target in targets:
                    # ImageNoisePair(model.net, os.path.join(os.getcwd(), args.train_dir), origin, [target], is_train=True)
                    # multi_set=ImageMaskPair(model.net,os.path.join(os.getcwd(),args.train_dir),target+"+",[origin],is_train=True,is_reverse=True); model.train(multi_set)
                    multi_set=ImageMaskPair(model.net,os.path.join(os.getcwd(),args.train_dir),target+"+",[target+"-"],is_train=True); model.train(multi_set)

    if mode != 't':
        for model in [Model(n) for n in nets]:
            for origin in origins:
                # multi_set = ImageMaskPair(model.net,os.path.join(os.getcwd(),args.pred_dir),origin,targets,is_train=False)
                # model.predict(multi_set, args.pred_dir)

                for target in targets:
                    # multi_set = ImageMaskPair(model.net,os.path.join(os.getcwd(),args.pred_dir),origin,[target+"+"],is_train=False); model.predict(multi_set, args.pred_dir)
                    multi_set = ImageMaskPair(model.net,os.path.join(os.getcwd(),args.pred_dir),origin,[target+"-"],is_train=False); model.predict(multi_set, args.pred_dir)
