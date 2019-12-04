import utils
from tcn import TCN
from tcn import compiled_tcn
from tensorflow.keras import backend as K, Model, Input, optimizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda
from tensorflow.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_x_y(size=1000):
    '''import numpy as np
    pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
    x_train = np.zeros(shape=(size, timesteps, 1))
    y_train = np.zeros(shape=(size, 1))
    x_train[pos_indices, 0] = 1.0
    y_train[pos_indices, 0] = 1.0
    return x_train, y_train'''
    import pandas as pd
    df = pd.read_csv("../../data/training.csv", header=None)
    data, label = df[0].values, df[1].values
    x_train, x_val, y_train, y_val = utils.train_test_split(data, label, 0.01, seed=1)
    return x_train, x_val, y_train, y_val


def train(model, x_train, x_val, y_train, y_val, max_len=102400, batch_size=1, shuffle=True):
    ear = EarlyStopping(monitor='loss', patience=50)
    mcp = ModelCheckpoint("./tcn.h5", monitor="loss", save_best_only=False, save_weights_only=False)
    history = model.fit_generator(
        utils.data_generator(x_train, y_train, max_len, batch_size, shuffle),
        steps_per_epoch=len(x_train),
        epochs=1,
        verbose=1,
        callbacks=[ear, mcp]
        # validation_data=utils.data_generator(x_val, y_val, max_len, batch_size, shuffle),
        # validation_steps=len(x_val)
    )
    return history, model


def predict(model, x_test, y_test, max_len=102400, batch_size=1, verbose=1):
    pred = model.predict_generator(
        utils.data_generator(x_test, y_test, max_len, batch_size, shuffle=False),
        steps=len(x_test),
        verbose=verbose
        )
    return pred

''',
        num_feat=1,
        num_classes=10,
        nb_filters=20,
        kernel_size=6,
        dilations=[2 ** i for i in range(9)],
        nb_stacks=1,
        max_len=None,  # x_train[0:1].shape[1],
        use_skip_connections=True'''

batch_size, timesteps, input_dim = None, None, 1
i = Input(batch_shape=(batch_size, timesteps, input_dim))
o = TCN(return_sequences=False, dilations=[2 ** i for i in range(9)], kernel_size=6, nb_filters=20, nb_stacks=1)(i)  # The TCN layers are here.
o = Dense(1)(o)
m = Model(inputs=[i], outputs=[o])
m.compile(optimizer='adam', loss='mse')
m.summary()

# m = compiled_tcn(return_sequences=False, num_feat=1, num_classes=10, nb_filters=20, kernel_size=6, dilations=[2 ** i for i in range(9)], nb_stacks=1, max_len=None, use_skip_connections=True)
x_train, x_val, y_train, y_val = get_x_y()
x_train = x_val
y_train = y_val
history, model = train(m, x_train, x_val, y_train, y_val)
#model = load_model("D:\\tcn.h5")
pred = predict(model, x_val, y_val)

from sklearn import metrics
confidence = 50
Y = y_val
acc = metrics.accuracy_score(Y, pred > (confidence / 100))
bacc = metrics.balanced_accuracy_score(Y, pred > (confidence / 100))
cm = metrics.confusion_matrix(Y, pred > (confidence / 100))
print('[Acc: ' + str(acc)[:6] + "] [Balanced Acc: " + str(bacc)[:6] + ']')
tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]
print("%5s & %5s & %5s & %5s" % (str(tn), str(fp), str(fn), str(tp)))

TPR = (tp / (tp + fn)) * 100
FPR = (fp / (fp + tn)) * 100
FNR = (fn / (fn + tp)) * 100
fpr_auc, tpr_auc, thds_auc = metrics.roc_curve(Y, pred, drop_intermediate=False)
auc = metrics.roc_auc_score(Y, pred)
# print("Overall ROC AUC Score     : {:0.2f}".format(auc))  # , fpr, tpr)
print("TIER-1", "TPR: {:0.5f}".format(TPR), "FPR: {:0.5f}".format(FPR))
