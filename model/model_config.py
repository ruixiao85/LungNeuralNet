class config:
    def __init__(self):
        self.__init__(dep_in=3, dep_out=2, act_fun='elu', out_fun='softmax', loss_fun='categorical_crossentropy')

    def __init__(self, dep_in, dep_out, act_fun=None, out_fun=None, loss_fun=None):
        self.dep_in, self.dep_out = dep_in, dep_out
        self.act_fun = act_fun
        self.out_fun = out_fun
        self.loss_fun = loss_fun
        self.aug = True
        self.overlay_channel, self.overlay_opacity = 2, 0.6  # 0-red 1-green 2-blue
        # self.overlay_channel, self.overlay_opacity = 1, 1.0  # 0-red 1-green 2-blue
        self.call_hardness = 0  # 1: hard(0/1), 0~1: mix, 0: original smooth

    def __str__(self):
        return '_'.join([self.act_fun, self.out_fun, self.loss_fun if isinstance(self.loss_fun, str) else self.loss_fun.__name__])