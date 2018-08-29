import argparse

from image_gen import ImageSet, ImagePair
from util import to_excel_sheet
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
                        default='Background,ConductingAirway,ConnectiveTissue,LargeBloodVessel,RespiratoryAirway,SmallBloodVessel', help='output: targets separated by comma')
    #
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
    from unet.unetflex import unet1s,unet1d,unet2s, conv3, conv33, conv331, conv31, dxmaxpool, uxmergeup
    configs = [
        # ModelConfig((768, 768, 3), (768, 768, 1), overlay_color=len(targets), model_filter=[32, 64, 128, 256, 512],
        #             model_downconv=conv131res, model_downsamp=d2conv131res, model_upsamp=u2trans131res, model_upconv=conv131res),

        # ModelConfig((1296,1296, 3), (1296,1296, 1), overlay_color=len(targets), model_filter=[32, 64, 128, 256, 512], model_poolsize=[2, 2, 2, 2, 2],
        #             model_name=unet1d, model_downconv=conv33, model_downsamp=dxmaxpool, model_upsamp=uxmergeup, model_upconv=conv3),

        # ModelConfig((1296,1296, 3), (1296,1296, 1), overlay_color=len(targets), model_filter=[32, 64, 128, 256, 512], model_poolsize=[2, 2, 2, 2, 2],
        #           model_downconv=conv33, model_downsamp=dxmaxpool, model_upsamp=uxmergeup, model_upconv=conv3),
        #
        # ModelConfig((1296,1296, 3), (1296,1296, 1), overlay_color=len(targets), model_filter=[32, 64, 128, 256, 512], model_poolsize=[3, 3, 3, 3, 3],
        #             model_downconv=conv33, model_downsamp=dxmaxpool, model_upsamp=uxmergeup, model_upconv=conv3),
        #
        # ModelConfig((1296,1296, 3), (1296,1296, 1), overlay_color=len(targets), model_filter=[32, 64, 128, 256, 512], model_poolsize=[2, 2, 3, 3, 3],
        #             model_downconv=conv33, model_downsamp=dxmaxpool, model_upsamp=uxmergeup, model_upconv=conv3),
        #
        # ModelConfig((1296, 1296, 3), (1296, 1296, 1), overlay_color=len(targets), model_filter=[32, 64, 128, 256, 512], model_poolsize=[3, 3, 3, 3, 3],
        #             model_downconv=conv33, model_downsamp=dxmaxpool, model_upsamp=uxmergeup, model_upconv=conv33),


        # ModelConfig((1296, 1296, 3), (1296, 1296, 1), overlay_color=len(targets), model_filter=[24,32,48,64,96,128,192,256], model_poolsize=[2,2,2,2,3,3,3,3],
        #             model_name=unet1d, model_downconv=conv33, model_downsamp=dxmaxpool, model_upsamp=uxmergeup, model_upconv=conv33),
        ModelConfig((1296, 1296, 3), (1296, 1296, 1), num_targets=len(targets), model_filter=[32, 48, 64, 96, 128, 192, 256, 384], model_poolsize=[2, 2, 2, 2, 3, 3, 3, 3],
                    model_name=unet1d, model_downconv=conv33, model_downsamp=dxmaxpool, model_upsamp=uxmergeup, model_upconv=conv3),
        # ModelConfig((1296, 1296, 3), (1296, 1296, 1), overlay_color=len(targets), model_filter=[24,32,48,64,96,128,192,256], model_poolsize=[2,2,2,2,3,3,3,3],
        #             model_downconv=conv33, model_downsamp=dxmaxpool, model_upsamp=uxmergeup, model_upconv=conv3),

        # ModelConfig((1296, 1296, 3), (1296, 1296, 1), overlay_color=len(targets), model_filter=[48,64,96,128,192,256,384,512], model_poolsize=[2,2,2,2,3,3,3,3],
        #             model_name=unet1d, model_downconv=conv33, model_downsamp=dxmaxpool, model_upsamp=uxmergeup, model_upconv=conv33),
        # ModelConfig((1296, 1296, 3), (1296, 1296, 1), overlay_color=len(targets), model_filter=[48,64,96,128,192,256,384,512], model_poolsize=[2,2,2,2,3,3,3,3],
        #             model_name=unet1d, model_downconv=conv33, model_downsamp=dxmaxpool, model_upsamp=uxmergeup, model_upconv=conv3),
        # ModelConfig((1296, 1296, 3), (1296, 1296, 1), overlay_color=len(targets), model_filter=[24,32,48,64,96,128,192,256], model_poolsize=[2,2,2,2,3,3,3,3],
        #             model_name=unet1d, model_downconv=conv331, model_downsamp=dxmaxpool, model_upsamp=uxmergeup, model_upconv=conv331),
        # ModelConfig((1296, 1296, 3), (1296, 1296, 1), overlay_color=len(targets), model_filter=[24,32,48,64,96,128,192,256], model_poolsize=[2,2,2,2,3,3,3,3],
        #             model_name=unet1d, model_downconv=conv331, model_downsamp=dxmaxpool, model_upsamp=uxmergeup, model_upconv=conv31),

        # 1296    1296
        # 648    432
        # 324    144
        # 162    48
        # 81    16
        # ModelConfig((1458, 1458, 3), (1458, 1458, 1), overlay_color=len(targets), model_filter=[32, 48, 64, 96, 128, 198, 256],
        #             model_downconv=conv33, model_downsamp=d3maxpool3, model_upsamp=u3mergeup3, model_upconv=conv3),
        # ModelConfig((768, 768, 3), (768, 768, 1), overlay_color=len(targets), model_filter=[32, 64, 128, 256, 512],
        #             model_downconv=conv33, model_downsamp=d2maxpool2, model_upsamp=u2mergeup2, model_upconv=conv33),
        # ModelConfig((768, 768, 3), (768, 768, 1), overlay_color=len(targets), model_filter=[32, 64, 128, 256, 512],
        #             model_downconv=conv33, model_downsamp=d2maxpool2, model_upsamp=u2mergeup2, model_upconv=conv3),

        # ModelConfig((1458, 1458, 3), (1458, 1458, 1), model_filter=[32, 48, 56, 80, 102, 130, 180], model_name=unet1,
        #             model_downconv=conv3n3n, model_downsamp=d3maxpool3, model_upconv=conv3n, model_upsamp=u3mergeup3),
        #
        # ModelConfig((729,729, 3), (729,729, 1),model_filter=[48, 64, 96, 128, 192, 256],model_name=unet1,
        #             model_downconv=conv33, model_downsamp=d3maxpool3, model_upconv=conv3, model_upsamp=u3mergeup3),
        # ModelConfig((729,729, 3), (729,729, 1),model_filter=[48, 64, 96, 128, 192, 256],model_name=unet1,
        #             model_downconv=conv3n3n, model_downsamp=d3maxpool3, model_upconv=conv3n, model_upsamp=u3mergeup3),


        # ModelConfig((1536,1536, 3), (1536,1536, 1),model_filter=[24, 32, 48, 56, 80, 102, 130, 180],model_name=unet1),
        # ModelConfig((1024,1024, 3), (1024,1024, 1),model_filter=[32, 48, 64, 96, 128, 192, 256, 384],model_name=unet1),
        #
        # ModelConfig((1536,1536, 3), (1536,1536, 1),model_filter=[24, 32, 48, 56, 80, 102, 130, 180],model_name=unet1,
        #             model_downconv=conv3n3n, model_downsamp=d2maxpool2, model_upconv=conv3n, model_upsamp=u2mergeup2),
        # ModelConfig((1024,1024, 3), (1024,1024, 1),model_filter=[32, 48, 64, 96, 128, 192, 256, 384],model_name=unet1,
        #             model_downconv=conv3n3n, model_downsamp=d2maxpool2, model_upconv=conv3n, model_upsamp=u2mergeup2),


        # ModelConfig((2048,2048, 3), (2048,2048, 1),model_filter=[24, 32, 48, 64, 96, 128, 128, 192]),
        # ModelConfig((1536,1536, 3), (1536,1536, 1),model_filter=[32, 48, 64, 96, 128, 192, 256, 384],model_name=unet_pool_up_resd_2f1),
        # ModelConfig((1536,1536, 3), (1536,1536, 1),model_filter=[32, 48, 64, 96, 128, 192, 256, 384],model_name=unet_pool_up_resb_2f1),
        # ModelConfig((1536,1536, 3), (1536,1536, 1),model_filter=[32, 48, 64, 96, 128, 192, 256, 384],model_name=unet_pool_up_resf_2f1),
        # ModelConfig((1536,1536, 3), (1536,1536, 1),model_filter=[32, 48, 64, 96, 128, 192, 256, 384]),
        # ModelConfig((1024,1024, 3), (1024,1024, 1),model_filter=[48, 64, 96, 128, 192, 256, 384],model_name=unet_pool_up_resd_2f1),
        # ModelConfig((1024,1024, 3), (1024,1024, 1),model_filter=[48, 64, 96, 128, 192, 256, 384],model_name=unet_pool_up_resb_2f1),
        # ModelConfig((1024,1024, 3), (1024,1024, 1),model_filter=[48, 64, 96, 128, 192, 256, 384],model_name=unet_pool_up_resf_2f1),
        # ModelConfig((1024,1024, 3), (1024,1024, 1),model_filter=[48, 64, 96, 128, 192, 256, 384]),


        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 128, 192, 192, 256, 256, 384], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 128, 192, 192, 256, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 192, 256, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 192, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 128, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),

        # ModelConfig((512, 512, 3), (512, 512, 1)),
        # ModelConfig((2048, 2048, 3), (2048, 2048, 1), model_filter=[20,28,40,57,81,115,163,231,327,462], image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        # ModelConfig((768, 768, 3), (768, 768, 1), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        # ModelConfig((768, 768, 3), (768, 768, 1), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.2, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        # ModelConfig((512, 512, 3), (512, 512, 1), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.2, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),

        # ModelConfig((768, 768, 3), (768, 768, 1), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        # ModelConfig((512, 512, 3), (512, 512, 1), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        # ModelConfig((768, 768, 3), (768, 768, 1), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.2, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),
        # ModelConfig((512, 512, 3), (512, 512, 1), model_filter=[32, 64, 96, 128, 192, 256, 384, 512], mask_color='green', image_resize=0.2, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0),

        # ModelConfig((768, 768, 3), (768, 768, 6), model_filter=[8, 16, 32, 64, 128, 256, 512, 1024], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0, model_out='softmax', model_loss='categorical_crossentropy'),
        # ModelConfig((768, 768, 3), (768, 768, 6), model_filter=[32, 64, 128, 256, 512, 1024, 2048, 4096], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0, model_out='softmax', model_loss='categorical_crossentropy'),
        # ModelConfig((768, 768, 3), (768, 768, 6), model_filter=[64,80,100,125,156,195,244,305], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0, model_out='softmax', model_loss='categorical_crossentropy'),
        # ModelConfig((768, 768, 3), (768, 768, 6), model_filter=[80,95,113,134,159,189,225,268], mask_color='green', image_resize=0.6, image_padding=1.0, separate=True, coverage_tr=1.5, coverage_prd=2.0, model_out='softmax', model_loss='categorical_crossentropy'),

        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 256], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 192, 256, 256, 384], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024, 1024, 1), filter_size=[64, 96, 128, 192, 256, 256, 384], kernel_size=(3,3), resize=0.6, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 256], kernel_size=(3,1), resize=0.5, padding=1.0, separate=True, tr_coverage=1.5, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),

        # ModelConfig((572, 572, 3), (572, 572, 1), filter_size=[64, 128, 256, 512, 1024], resize=1.0, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # Vanilla U-Net 572 -> 388 (valid padding, Center 67%), pool_up_2f2 with 64, 128, 256, 512, 1024 filters

        # ModelConfig((1060, 1060, 3), (1060,1060, 1), resize=1.0, separate=True, tr_coverage=0.9 prd_coverage=1.4, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024,1024, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), resize=1.0, padding=1.0, separate=True, tr_coverage=0.9, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192], resize=1.0, padding=1.0, separate=False, tr_coverage=0.9, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192, 256], resize=0.5, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024, 1024, 1), filter_size=[64, 96, 128, 192, 256], resize=0.5, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # last-use ModelConfig((512, 512, 3), (512, 512, 1), filter_size=[64, 96, 128, 192, 256], resize=1.0, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024, 1024, 1), filter_size=[32, 48, 64, 96, 128, 192, 256, 384, 512], resize=0.5, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((1024, 1024, 3), (1024, 1024, 1), filter_size=[64, 96, 128, 192, 256, 384], resize=0.5, padding=1.0, separate=True, tr_coverage=1.2, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192, 256, 384], resize=1.0, padding=1.0, separate=False, tr_coverage=0.9, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((768, 768, 3), (768, 768, 1), filter_size=[64, 96, 128, 192, 256, 384, 512], resize=1.0, padding=1.0, separate=False, tr_coverage=0.9, prd_coverage=2.0, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((512, 512, 3), (512, 512, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((256, 256, 3), (256, 256, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, model_out='sigmoid', model_loss=loss_bce_dice),
        # ModelConfig((768, 768, 3), (674, 674, 1), resize=1.0, separate=True, tr_coverage=0.9, prd_coverage=1.4, model_out='sigmoid', model_loss=loss_bce_dice),  # 5 valid
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
