import os
import argparse
import pandas as pd
from datetime import datetime
from PIL import ImageDraw, Image, ImageFont
from skimage.io import imsave, imread
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from image_gen import ImageGenerator
from process_image import scale_input, augment_image_pair, scale_output
from tensorboard_train_val import TensorBoardTrainVal
from model_config import config
from util import get_recursive_rel_path, mk_dir_if_nonexist


def get_train_data(sub_dir, dir_in, dir_out, split_div=4): # index % sp_div==0 to validation set
    wd = os.path.join(os.getcwd(), sub_dir)
    images, total = get_recursive_rel_path(wd, dir_in)
    tgts, _ = get_recursive_rel_path(wd, dir_out)

    images = list(set(images).intersection(tgts))  # image-target pairs only
    total = len(images)  # update # of files
    print("%d image-mask pairs accepted" % total)
    n_val=int(total/split_div)+1
    n_train=total-n_val
    print("Split by division of %d: validation size=%d; train size=%d" % (split_div,n_val,n_train))
    cfg.full=False # avoid similar images
    _val=ImageGenerator('val',cfg,wd,dir_in,dir_out,shuffle=False,img_aug=False)
    _tr=ImageGenerator('tr',cfg,wd,dir_in,dir_out,shuffle=True,img_aug=True)
    if not _val.skip_convert or not _tr.skip_convert:
        _val.recreate_dir()
        _tr.recreate_dir()
        vi, ti =0, 0
        for i, image_name in enumerate(images):
            if i % split_div == 0:
                vi += _val.register_image(image_name) # write=True to crop and save files
            else:
                ti += _tr.register_image(image_name)
            if int(10. * (i + 1) / total) > int(10. * i / total):
                print('Loading %d / %d images [%.0f%%]' % (i + 1, total, 10 * int(10. * (i + 1) / total)))
        print("Successfully added %d for validation and %d for training" % (vi,ti))
    return _tr, _val

def get_predict_data(sub_dir, dir_in):
    wd = os.path.join(os.getcwd(), sub_dir)
    images, total = get_recursive_rel_path(wd, dir_in)
    cfg.full=True # insure coverage
    _img = ImageGenerator('pred',cfg, wd, dir_in, None, shuffle=False, img_aug=False)
    for i, image_name in enumerate(images):
        _img.register_image(image_name)
        if int(10. * (i + 1) / total) > int(10. * i / total):
            print('Loading %d / %d images [%.0f%%]' % (i + 1, total, 10 * int(10. * (i + 1) / total)))
    return _img

def train(_target, num_rep=3, num_epoch=12, cont_train=True):
    weight_file = _target + ".h5"

    if cont_train and os.path.exists(weight_file):
        print("Continue from previous weights")
        model.load_weights(weight_file)

    print('Creating unet and checkpoint...')
    # indicator, trend = 'val_loss', 'min'
    indicator, trend = 'val_dice', 'max'
    print('Fitting unet...')
    for r in range(num_rep):
        print("Training %d/%d for %s" % (r + 1, num_rep, _target))
        # for _i,_ti in enumerate(_tr_img):
        #     imsave("train_aug%d.jpg"% _i, Image.fromarray(_ti.astype(np.uint8), 'RGB'))
        tr.on_epoch_end()
        val.on_epoch_end()
        history = model.fit_generator(tr,validation_data=val,steps_per_epoch=min(100,len(tr.view_coord)),validation_steps=min(30,len(val.view_coord)),
                                      epochs=num_epoch,max_queue_size=1,workers=1,use_multiprocessing=False,shuffle=False,
              callbacks=[
                  ModelCheckpoint(weight_file, monitor=indicator, mode=trend, save_best_only=True),
                  ReduceLROnPlateau(monitor=indicator, mode=trend, factor=0.1, patience=10, verbose=0, min_delta=1e-5, cooldown=0, min_lr=0),
                  EarlyStopping(monitor=indicator, mode=trend, patience=0, verbose=1),
                  TensorBoardTrainVal(log_dir=os.path.join("log", _target), write_graph=True, write_grads=False, write_images=True),
              ]).history
        with open(_target+".csv", "a") as log:
            log.write("\n"+datetime.now().strftime("%Y-%m-%d %H:%M")+" train history:\n")
        pd.DataFrame(history).to_csv(_target+".csv", mode="a")


def predict(_target, _vec, w_whole=True, w_ind=False):
    oc, op, ch, pad, i_sum = cfg.overlay_channel, cfg.overlay_opacity, cfg.call_hardness, cfg.pad, cfg.sum()
    print('Load weights and predicting ...')
    model.load_weights(_target + ".h5")
    tst.on_epoch_end()
    imgs_mask_test = model.predict_generator(tst, verbose=1, batch_size=1)

    target_dir = os.path.join(args.pred_dir, _target)
    print('Saving predicted results [%s] to files...' % _target)
    mk_dir_if_nonexist(target_dir)
    ind_file, whole_file, _ind, _whole, _weight, _val, _sum = None, None, None, None, None, 0, 0
    for i, image in enumerate(imgs_mask_test):
        new_whole_file = os.path.join(target_dir, tst.view_coord[i].file_name.replace(".jpg", ".png"))
        if whole_file is not None and new_whole_file!=whole_file: # export whole_file
            text="%s \n Pixels: %.0f / %.0f Percentage: %.0f%%" % (whole_file, _val, _sum, 100. * _val / _sum)
            print(text)
            if w_whole:  # write wholes image
                _whole = Image.fromarray(((_whole/_weight + 1.0) * 127.).astype(np.uint8), 'RGB')
                draw = ImageDraw.Draw(_whole)
                draw.text((0, 0), text,
                          (255 if oc == 0 else 10, 255 if oc == 1 else 10, 255 if oc == 2 else 10),
                          ImageFont.truetype("arial.ttf", 24))  # font type size)
                mk_dir_if_nonexist(os.path.dirname(whole_file))
                imsave(whole_file, _whole)
            _whole, _weight, _val, _sum = None, None, 0, 0
        whole_file = os.path.join(target_dir, tst.view_coord[i].file_slice)
        ind_file = os.path.join(target_dir, tst.view_coord[i].file_slice)
        if ch==1:  # hard sign
            image = np.rint(image)
        elif 0<ch<1:
            image=(image+np.rint(image)*ch)/(1.0+ch)  # mixed
        i_val=int(np.sum(image[..., 0]))
        _vec[i]=i_val
        _val += i_val
        _sum += i_sum
        text="%s Pixels: %.0f / %.0f Percentage: %.0f%%" % (ind_file, i_val, i_sum, 100. * i_val / i_sum)
        print(text)

        if _whole is None:
            _whole=np.zeros((tst.view_coord[i].origin_row,tst.view_coord[i].origin_col,tst.view_coord[i].origin_dep),dtype=np.float32)
            _weight=np.zeros((tst.view_coord[i].origin_row,tst.view_coord[i].origin_col,tst.view_coord[i].origin_dep),dtype=np.float32)
        _ind=tst[i]
        for c in range(3):
            if c == oc:
                _ind[..., c] = np.tanh(_ind[..., c] + op * image[..., 0])
            else:
                _ind[..., c] = np.tanh(_ind[..., c] - op * image[..., 0])
        _whole[tst.view_coord[i].row_start:tst.view_coord[i].row_end, tst.view_coord[i].col_start:tst.view_coord[i].col_end, ...] += _ind
        _weight[tst.view_coord[i].row_start:tst.view_coord[i].row_end, tst.view_coord[i].col_start:tst.view_coord[i].col_end, ...] += 1.0
        if w_ind:  # write individual image
            _ind = Image.fromarray(((_ind + 1.0) * 127.).astype(np.uint8), 'RGB')
            draw = ImageDraw.Draw(_ind)
            draw.text((0, 0), text.replace("Pixel","\nPixel"),
                      (255 if oc == 0 else 10, 255 if oc == 1 else 10, 255 if oc == 2 else 10),
                      ImageFont.truetype("arial.ttf", 24))  # font type size)
            mk_dir_if_nonexist(os.path.dirname(ind_file))
            imsave(ind_file, _ind)

def append_excel_sheet(_df, _xls, _sheet):
    from openpyxl import load_workbook
    book = load_workbook(_xls)
    writer = pd.ExcelWriter(_xls, engine='openpyxl')
    writer.book = book
    _df.to_excel(writer, sheet_name=_sheet)
    writer.save()
    writer.close()


if __name__ == '__main__':
    # -d "D:\Cel files\2018-07.13 Adam Brenderia 2X LPS CGS" -t "071318 Cleaned 24H post cgs" -p "2018-07.20 Kyle MMP13 Smoke Flu Zander 2X" -o "Paren,InflamMild,InflamSevere"
    # -d "I:/NonParen" -t "Rui Xiao 2017-09-07 (14332)(27)" -p "Rui Xiao 2017-12-05 (15634)(9)" -o "NonParen"
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', '--dir', dest='dir', action='store',
                        default='', help='work directory, empty->current dir')
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
                        default='jpg', help='extension')
    parser.add_argument('-i', '--input', dest='input', type=str,
                        default='Original', help='input: Original')
    parser.add_argument('-o', '--output', dest='output', type=str,
                        default='Paren,InflamMild,InflamSevere', help='output: targets separated by comma')
    args = parser.parse_args()

    os.chdir(os.getcwd() if (args.dir == '') else args.dir)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # force cpu
    targets = args.output.split(',')

    from util_unet import build_compile
    from unet.unet_pool_up_31 import unet_pool_up_5_dure
    from unet.unet_pool_up_valid import unet_pool_up_5_valid
    models = [
        # unet_pool_trans_5,
        # unet_pool_trans_7,
        # unet_conv_trans_5,
        # unet_conv_trans_7,
        # unet_pool_up_5,
        # unet_pool_up_7,
        unet_pool_up_5_dure,
        # unet_pool_up_6_dure,
        # unet_pool_up_7_dure,
        # unet_pool_up_5_valid,
        # unet_vgg_7conv,
    ]
    from unet.util_unet import *
    configs = [
        # config((1024, 1024, 3), (1024,1024, 1), resize=1.0, padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),
        config((512, 512, 3), (512, 512, 1), resize=1.0, padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),
        # config((768, 768, 3), (674, 674, 1), resize=1.0, padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),  # 5 valid
        # config(256, 256, 3, 1, resize=1., padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),
        # config(256, 256, 3, 1, resize=1., padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),
        # config(768, 768, 3, 1, resize=1.0, padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),
        # config(512, 512, 3, 1, resize=0.8, padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),
        # config(256, 256, 3, 1, resize=1., padding=0, full=True, act_fun='elu', out_fun='sigmoid', loss_fun=loss_bce_dice),
        # config(3, 1, 'elu', 'sigmoid', loss_dice),
        # config(3, 1, 'elu', 'sigmoid', 'binary_crossentropy'), # mid
        # config(3, 1, 'elu', 'sigmoid', loss_bce), # no good
        # config(3, 1, 'elu', 'sigmoid', loss_jaccard), # no good
        # config(3, 2, 'relu', 'softmax', 'categorical_crossentropy'),
        # config(3, 2, 'elu', 'softmax'),
        # config(3, 2, 'tanh', 'softmax', 'categorical_crossentropy'),
        # config(3, 2, 'softsign', 'softmax', 'categorical_crossentropy'),  # trouble
        # config(3, 2, 'selu', 'softmax', 'categorical_crossentropy'),  # trouble
        # config(3, 2, 'softplus', 'softmax', 'categorical_crossentropy'), # trouble
    ]
    mode = args.mode[0].lower()
    if mode != 'p':
        for cfg in configs:
            for mod in models:
                model, nn = build_compile(mod, cfg, write=True)
                print("Network specifications: " + nn.replace("_", " "))
                for target in targets:
                    tr, val = get_train_data(args.train_dir, args.input, target)
                    train(target + nn, num_rep=4, num_epoch=20, cont_train=True)

    if mode != 't':
        for cfg in configs:
            tst = get_predict_data(args.pred_dir, args.input)
            for mod in models:
                model, nn = build_compile(mod, cfg)
                res = np.zeros((len(tst.view_coord), len(targets)), np.uint32)
                for x, target in enumerate(targets):
                    predict(target + nn, res[:, x])
                # res_df.to_csv("result_" + args.pred_dir + "_" + nn + ".csv")
                res_df = pd.DataFrame(res,map(str,tst.view_coord), targets)
                xls_file = "Result_" + args.pred_dir + "_" + nn + ".xlsx"
                res_df.to_excel(xls_file, sheet_name = 'Individual')
                append_excel_sheet(res_df.groupby(res_df.index).sum(),xls_file,"Whole")