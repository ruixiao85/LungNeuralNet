class config:
    def __init__(self):
        self.row, self.col = None, None
        self.dep_in, self.dep_out = None, None
        self.resize, self.pad = None, None
        self.full = None
        self.act_fun = None
        self.out_fun = None
        self.loss_fun = None
        self.overlay_channel, self.overlay_opacity = None, None
        self.call_hardness = None


    def __init__(self, row=512, col=512, dep_in=3, dep_out=1, resize=1., padding=0, full=True,
                 act_fun='elu', out_fun='sigmoid', loss_fun='binary_crossentropy',
                 over_ch=2, over_op=0.6, call_hard=0):  # 0-red 1-green 2-blue; opacity; 1: hard(0/1), 0~1: mix, 0: original smooth
        self.row, self.col = row, col
        self.dep_in, self.dep_out = dep_in, dep_out
        self.resize, self.pad = resize, padding
        self.full = full
        self.act_fun = act_fun
        self.out_fun = out_fun
        self.loss_fun = loss_fun
        self.overlay_channel, self.overlay_opacity = over_ch, over_op
        self.call_hardness = call_hard

    def sum(self):
        return self.row*self.col

    def __str__(self):
        return '_'.join([
            "%d" % int(0.5*(self.row+self.col)),
            # "%d_%d"% (self.row, self.col),
            # "%d_%d"% (self.dep_in, self.dep_out),
            "%.1f" % self.resize,
            # "%d" % self.pad,
            # "%r" % self.full,  # not suitable for unet saving
             # self.act_fun, self.out_fun, self.loss_fun if isinstance(self.loss_fun, str) else self.loss_fun.__name__,
            # "%d_%.1f_%.1f" %  (self.overlay_channel, self.overlay_opacity, self.call_hardness)
             ])