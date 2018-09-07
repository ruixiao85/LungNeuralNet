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
                        default='LargeBloodVessel,SmallBloodVessel', help='output: targets separated by comma')
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
    from net.unet import unet
    from net.icnet import icnet
    from net.module import ca1, ca2, ca3, ca3h, cadh, ca33, ca13, cba3, cb3, dmp, dca, uu, ut, uta, sk, ct,\
      du32, cdu33, rn33r, rn33nr, rn131r, rn131nr, dn13r, dn13nr
    from keras.optimizers import Adam, SGD, RMSprop, Nadam
    configs = [
        # #SegNet zero padding downwards: conv->batchnorm->activation downsample: maxpool upwards: conv->batchnorm (no act) upsamp: upsampling activation on output layer
        # #U-shape 64,128(/2),256(/4),512(/8),512(/8),256(/4),128(/2),64,
        # ModelConfig((768, 768, 3), (768, 768, 1), num_targets=len(targets), model_filter=[64, 128, 256, 512], model_pool=[2, 2, 2], model_name=unet,
        #             train_rep=2, optimizer=Adam(4e-5), predict_all_inclusive=True,
        #             model_preproc=cba3, model_downconv=sk, model_downjoin=sk, model_downsamp=dmp, model_downmerge=sk, model_downproc=cba3,
        #             model_upconv=cb3, model_upjoin=sk, model_upsamp=uu, model_upmerge=sk, model_upproc=cb3, model_postproc=sk),
        #
        # ModelConfig((768, 768, 3), (768, 768, 1), num_targets=len(targets), model_filter=[64, 128, 256, 512, 1024], model_pool=[2, 2, 2, 2, 2], model_name=unet,
        #             train_rep=5, optimizer=Adam(4e-5), predict_all_inclusive=True,
        #             model_preproc=cba3, model_downconv=sk, model_downjoin=sk, model_downsamp=dmp, model_downmerge=sk, model_downproc=cba3,
        #             model_upconv=cb3, model_upjoin=sk, model_upsamp=uu, model_upmerge=sk, model_upproc=sk, model_postproc=cb3),
        #
        # #UNET valid padding 572,570,568->284,282,280->140,138,136->68,66,64->32,30,28->56,54,52->104,102,100->200,198,196->392,390,388 388/572=67.8322% center
        # #UNET same padding 576->288->144->72->36->72->144->288->576 take central 68% =392
        # ModelConfig((768, 768, 3), (768, 768, 1), num_targets=len(targets), model_filter=[64, 128, 256, 512, 1024], model_pool=[2, 2, 2, 2, 2], model_name=unet,
        #             train_rep=2, optimizer=Adam(4e-5), predict_all_inclusive=True,
        #             model_preproc=ca3, model_downconv=ca3, model_downjoin=sk, model_downsamp=dmp, model_downmerge=sk, model_downproc=ca3,
        #             model_upconv=sk, model_upjoin=sk, model_upsamp=uu, model_upmerge=ct, model_upproc=ca33, model_postproc=sk),
        #
        #
        # ModelConfig((1296, 1296, 3), (1296, 1296, 1), num_targets=len(targets), model_filter=[64, 96, 128, 196, 256, 256, 256, 256, 256], model_pool=[2, 2, 2, 2, 3, 3, 3, 3, 3],
        #             model_name=unet, train_rep=3, optimizer=Adam(1e-5), predict_all_inclusive=True, separate=True,
        #             model_preproc=ca3, model_downconv=ca3, model_downjoin=sk, model_downsamp=dmp, model_downmerge=sk, model_downproc=ca3,
        #             model_upconv=sk, model_upjoin=ct, model_upsamp=uu, model_upmerge=ct, model_upproc=ca3, model_postproc=sk),
        #
        # ModelConfig((729, 729, 3), (729, 729, 1), num_targets=len(targets), model_filter=[64, 128, 256, 512, 512, 512, 512], model_pool=[3, 3, 3, 3, 3, 3, 3],
        #             model_name=unet, train_rep=6, optimizer=Adam(1e-5), predict_all_inclusive=True,
        #             model_preproc=ca3, model_downconv=ca3, model_downjoin=sk, model_downsamp=dmp, model_downmerge=sk, model_downproc=ca3,
        #             model_upconv=sk, model_upjoin=ct, model_upsamp=uu, model_upmerge=ct, model_upproc=ca33, model_postproc=sk),
        #
        # ModelConfig((729, 729, 3), (729, 729, 1), num_targets=len(targets), model_filter=[64, 128, 256, 512, 512, 512, 512], model_pool=[3, 3, 3, 3, 3, 3, 3],
        #             model_name=unet, train_rep=6, optimizer=Adam(1e-5), predict_all_inclusive=True,
        #             model_preproc=cadh, model_downconv=ca3, model_downjoin=sk, model_downsamp=dmp, model_downmerge=sk, model_downproc=ca3,
        #             model_upconv=sk, model_upjoin=ct, model_upsamp=uu, model_upmerge=ct, model_upproc=ca33, model_postproc=sk),
        #
        # ModelConfig((1296, 1296, 3), (1296, 1296, 1), num_targets=len(targets), model_filter=[64, 96, 128, 196, 256, 256, 256, 256, 256], model_pool=[2, 2, 2, 2, 3, 3, 3, 3],
        #             model_name=unet, train_rep=2, optimizer=Adam(1e-4), predict_all_inclusive=True,
        #             model_preproc=ca3, model_downconv=sk, model_downjoin=sk, model_downsamp=dmp, model_downmerge=sk, model_downproc=dn13r,
        #             model_upconv=sk, model_upjoin=ct, model_upsamp=uu, model_upmerge=ct, model_upproc=ca3h, model_postproc=sk),
        #
        # ModelConfig((512, 512, 3), (512, 512, 1), num_targets=len(targets), model_filter=[64, 96, 128, 128, 128], model_pool=[2, 2, 2, 2, 2], model_name=unet,
        #             train_rep=6, optimizer=Adam(4e-5), predict_all_inclusive=True,
        #             model_preproc=ca3, model_downconv=ca3, model_downjoin=sk, model_downsamp=dmp, model_downmerge=sk, model_downproc=ca3,
        #             model_upconv=ca3, model_upjoin=ct, model_upsamp=uu, model_upmerge=ct, model_upproc=ca3, model_postproc=ca3),

        # ICNet
        ModelConfig((512, 512, 3), (512, 512, 1), num_targets=len(targets), model_filter=[64, 96, 128, 196, 256, 256, 256, 256, 256], model_pool=[2, 2, 2, 2, 3, 3, 3, 3],
                    model_name=icnet, train_rep=2, optimizer=Adam(1e-4), predict_all_inclusive=True,
                    model_preproc=ca3, model_downconv=sk, model_downjoin=sk, model_downsamp=dmp, model_downmerge=sk, model_downproc=dn13r,
                    model_upconv=sk, model_upjoin=ct, model_upsamp=uu, model_upmerge=ct, model_upproc=ca3h, model_postproc=sk),

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
            for origin in origins:
                multi_set = ImagePair(cfg, os.path.join(os.getcwd(), args.pred_dir), origin, targets, is_train=False)
                model.predict(multi_set, args.pred_dir)
