from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adadelta, SGD, RMSprop, Adagrad, Adam, Adamax
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import logging

# Setup logging
logger = logging.getLogger(__name__)

class BaseMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden=1000, n_deep=4,
                 l1_norm=0, l2_norm=0, drop=0,
                 early_stop=True, max_epoch=5000,
                 patience=200, learning_rate=None,
                 optimizer='Adadelta', activation='tanh',
                 verbose=0):
        self.max_epoch = max_epoch
        self.early_stop = early_stop
        self.n_hidden = n_hidden
        self.n_deep = n_deep
        self.l1_norm = l1_norm
        self.l2_norm = l2_norm
        self.drop = drop
        self.patience = patience
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.activation = activation

    def fit(self, X, y, **kwargs):
        y = np.squeeze(y)
        if len(y.shape) == 1:  # One class
            self.n_class = 1
            self.n_label = len(np.unique(y))
            if self.n_label == 1:
                logger.error('Label does not have more than 1 unique element')
            elif self.n_label == 2:  # two labels
                out_dim = 1
            else:  # more than two labels
                out_dim = self.n_label
        else:  # More than one class
            self.n_class = y.shape[1]
            self.n_label = [len(np.unique(y[:,ii])) for ii in range(self.n_class)]
            out_dim = self.n_class

        if hasattr(self, 'model'):
            self.reset_model()
        else:
            self.build_model(X.shape[1], out_dim)
            
        if self.verbose:
            temp = [layer.units for layer in self.model.layers if isinstance(layer, Dense)]
            print('Model:{}'.format(temp))
            print('l1: {}, drop: {}, lr: {}, patience: {}'.format(
                self.l1_norm, self.drop, self.learning_rate,
                self.patience))

        return self

    def build_model(self, in_dim, out_dim, n_class=1):
        self.model = build_model(in_dim, out_dim=out_dim,
                               n_hidden=self.n_hidden, l1_norm=self.l1_norm,
                               l2_norm=self.l2_norm,
                               n_deep=self.n_deep, drop=self.drop,
                               learning_rate=self.learning_rate,
                               optimizer=self.optimizer,
                               activation=self.activation,
                               n_class=self.n_class)
        self.w0 = self.model.get_weights()
        return self

    def predict_proba(self, X):
        proba = self.model.predict(X, verbose=self.verbose)
        if proba.shape[1] == 1:
            proba = np.array(proba).reshape((X.shape[0], -1))
            temp = (1 - proba.sum(axis=1)).reshape(X.shape[0], -1)
            proba = np.hstack((temp, proba))
        return proba

    def predict(self, X):
        prediction = self.model.predict(X, verbose=self.verbose)
        if self.n_class == 1:
            return np.round(prediction[:, 1])
        else:
            return np.round(prediction)

def build_model(in_dim, out_dim=1, n_hidden=100, l1_norm=0.0,
                l2_norm=0, n_deep=5, drop=0.1,
                learning_rate=0.1, optimizer='Adadelta',
                activation='tanh', n_class=1):
    model = Sequential()
    # Input layer
    model.add(Dense(
        units=n_hidden,
        input_dim=in_dim,
        kernel_initializer='uniform',
        activation=activation,
        kernel_regularizer=l1_l2(l1=l1_norm, l2=l2_norm)))

    # Hidden layers
    for _ in range(n_deep - 1):
        model.add(Dropout(drop))
        model.add(Dense(
            units=n_hidden,
            kernel_initializer='uniform',
            activation=activation))

    # Output layer
    if out_dim == 1:
        final_activation = activation
    elif n_class == 1:
        final_activation = 'softmax'
    else:
        final_activation = 'sigmoid'

    model.add(Dense(units=out_dim,
                   kernel_initializer='uniform',
                   activation=final_activation))

    # Optimizer selection
    optimizer_classes = {
        'Adadelta': Adadelta,
        'SGD': SGD,
        'RMSprop': RMSprop,
        'Adagrad': Adagrad,
        'Adam': Adam,
        'Adamax': Adamax
    }
    
    optimizer_class = optimizer_classes.get(optimizer, Adadelta)
    opt = optimizer_class(learning_rate=learning_rate) if learning_rate else optimizer_class()

    # Compile model
    loss = 'binary_crossentropy' if out_dim == 1 else 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=opt)

    return model
