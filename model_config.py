class ModelConfig:

    def __init__(self, dim_in=(512,512,3), dim_out=(512,512,1), image_format="*.jpg", mask_color="green",  #green/white
                 resize=1., padding=1.0, tr_coverage=0.9, prd_coverage=1.4,
                 filter_size=None, kernel_size=None,
                 act_fun='elu', out_fun='sigmoid', loss_fun='binary_crossentropy',
                 over_ch=2, over_op=0.5, call_hard=1,  # 0-red 1-green 2-blue; red-blue flipped with cv2; opacity; 1: hard(0/1), 0~1: mix, 0: original smooth
                 num_rep=2, num_epoch=12, max_train_step=500, max_vali_step=200, # no limit if None
                 learning_rate=1e-4, valid_split=0.4, img_aug=True, cont_train=True,
                 batch_size=2, shuffle=True, separate=True
                 ):
        self.row_in, self.col_in, self.dep_in = dim_in
        self.row_out, self.col_out, self.dep_out = dim_out
        self.image_format=image_format
        self.mask_color=mask_color
        self.resize= resize
        self.padding= padding
        self.tr_coverage = tr_coverage
        self.prd_coverage = prd_coverage
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.act_fun = act_fun
        self.out_fun = out_fun
        self.loss_fun = loss_fun
        self.overlay_channel, self.overlay_opacity = over_ch, over_op
        self.call_hardness = call_hard
        self.num_rep = num_rep
        self.num_epoch = num_epoch
        self.max_train_step = max_train_step
        self.max_vali_step = max_vali_step
        self.learning_rate = learning_rate
        self.valid_split = valid_split
        self.img_aug = img_aug
        self.shuffle = shuffle
        self.continue_train = cont_train
        self.batch_size = batch_size
        self.separate = separate

    def sum(self):
        return self.row_in * self.col_in

    def __str__(self):
        return '_'.join([
            "%d" % int(0.5 * (self.row_in + self.col_in)),
            # "%d_%d"% (self.row, self.col),
            # "%d_%d"% (self.dep_in, self.dep_out),
            # "%.1f" % self.resize,
            # "%d" % self.pad,
             "%df%d-%d_%dk%s" % (len(self.filter_size), self.filter_size[0],self.filter_size[-1], len(self.kernel_size), ''.join(str(x) for x in self.kernel_size)),
             # self.act_fun, self.out_fun, self.loss_fun if isinstance(self.loss_fun, str) else self.loss_fun.__name__,
            # "%d_%.1f_%.1f" %  (self.overlay_channel, self.overlay_opacity, self.call_hardness)
             ])