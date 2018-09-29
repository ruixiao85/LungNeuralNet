
from keras.engine.saving import model_from_json

from net.basecfg import Config


class Net(Config):
    # 'relu6'  # min(max(features, 0), 6)
    # 'crelu'  # Concatenates ReLU (only positive part) with ReLU (only the negative part). Note that this non-linearity doubles the depth of the activations
    # 'elu'  # Exponential Linear Units exp(features)-1, if <0, features
    # 'selu'  # Scaled Exponential Linear Rectifier: scale * alpha * (exp(features) - 1) if < 0, scale * features otherwise.
    # 'softplus'  # log(exp(features)+1)
    # 'softsign' features / (abs(features) + 1)

    # 'mean_squared_error' 'mean_absolute_error'
    # 'binary_crossentropy'
    # 'sparse_categorical_crossentropy' 'categorical_crossentropy'

    #     model_out = 'softmax'   model_loss='categorical_crossentropy'
    #     model_out='sigmoid'    model_loss=[loss_bce_dice] 'binary_crossentropy' "bcedice"
    def __init__(self, dim_in=None, dim_out=None, feed=None, act=None, out=None, loss=None, metrics=None, optimizer=None, indicator=None,
                 filename=None, **kwargs):

        super(Net,self).__init__(**kwargs)
        self.row_in, self.col_in, self.dep_in=dim_in or (512,512,3)
        self.row_out, self.col_out, self.dep_out=dim_out or (512,512,1)
        from metrics import jac, dice, dice67, dice33, acc, acc67, acc33, loss_bce_dice, custom_function_keras
        custom_function_keras()  # leakyrelu, swish
        self.feed=feed or 'tanh'
        self.act=act or 'elu'
        self.out=out or ('sigmoid' if self.dep_out==1 else 'softmax')
        self.loss=loss or (
            loss_bce_dice if self.dep_out==1 else 'categorical_crossentropy')  # 'binary_crossentropy'
        self.metrics=metrics or ([jac, dice, dice67, dice33] if self.dep_out==1 else [acc, acc67, acc33])
        from keras.optimizers import Adam
        self.optimizer=optimizer or Adam(1e-5)
        self.indicator=indicator if indicator is not None else ('val_dice' if self.dep_out==1 else 'val_acc')  # indicator to maximize
        self.net=None # abstract -> instatiate in subclass
        self.filename=filename

    @classmethod
    def from_json(cls, filename):  # load model from json
        my_net=cls(filename=filename)
        with open(filename+".json", 'r') as json_file:
            my_net.net=model_from_json(json_file.read())

    def save_net(self):
        json_net=(self.filename if self.filename is not None else str(self)) + ".json"
        with open(json_net, "w") as json_file:
            json_file.write(self.net.to_json())

    def compile_net(self, print_summary=True):
        self.net.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics)
        if print_summary:
            self.net.summary()

    def __str__(self):
        return self.filename # should not be call unless loaded from json

    @staticmethod
    def cap_lim_join(lim,*text):
        test_list=[t.capitalize()[:lim] for t in text]
        return ''.join(test_list)
