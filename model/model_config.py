class config:
    def __init__(self):
        self.dep_in, self.dep_out=3,2
        self.act_fun = 'elu'
        self.out_fun = 'softmax'
        self.loss_fun = 'categorical_crossentropy'
        self.overlay_channel = 2  # blue

    def __init__(self, dep_in, dep_out, act_fun, out_fun, loss_fun=None):
        self.__init__(dep_in, dep_out, act_fun, out_fun, loss_fun)

    def __str__(self):
        return '_'.join([self.act_fun, self.out_fun, self.loss_fun if isinstance(self.loss_fun, str) else self.loss_fun.__name__])