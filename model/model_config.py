class config:
    def __init__(self):
        self.__init__(dep_in=3, dep_out=2, act_fun='elu', out_fun='softmax', loss_fun='categorical_crossentropy')

    def __init__(self, dep_in, dep_out, act_fun, out_fun, loss_fun=None):
        self.dep_in, self.dep_out = dep_in, dep_out
        self.act_fun = act_fun
        self.out_fun = out_fun
        self.loss_fun = loss_fun
        self.overlay_channel = 1  # 0-red 1-green 2-blue
        self.overlay_opacity = 1.0  # opacity

    def __str__(self):
        return '_'.join([self.act_fun, self.out_fun, self.loss_fun if isinstance(self.loss_fun, str) else self.loss_fun.__name__])