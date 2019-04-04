import os
os.environ["DATA_FOLDER"] = "/home/biodisi/input/"

import keras

from awx_core.layers import *
from utils.parser import *
from utils import new_datasets

from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve


train, val, test = initialize_dataset('borat_yeast_GO', new_datasets)

clf = keras.models.Sequential([
    keras.layers.Dense(
        500,
        activation='tanh',
        kernel_regularizer=keras.regularizers.l1_l2(l2=0, l1=0),
        name='dense_1'
    ),
    keras.layers.GaussianNoise(0.1),
    keras.layers.Dense(
        500,
        activation='tanh',
        kernel_regularizer=keras.regularizers.l1_l2(l2=0, l1=0),
        name='dense_2'
    ),
    keras.layers.GaussianNoise(0.1),
    keras.layers.Dense(
        500,
        activation='tanh',
        kernel_regularizer=keras.regularizers.l1_l2(l2=0, l1=0),
        name='dense_3'
    ),
    keras.layers.GaussianNoise(0.1),
    AWX(
        A=train.A, 
        n_norm=1, 
        activation='sigmoid', 
        kernel_regularizer=keras.regularizers.l1_l2(l1=0, l2=1e-6), 
        name='AWX'
    )
])

clf.compile(
    keras.optimizers.Adam(lr=1e-5),
    loss='binary_crossentropy',
    metrics=['binary_crossentropy']
)


clf.fit(
    train.X,
    train.Y,

    validation_data=[
        val.X,
        val.Y,
    ],
    epochs=2000, 
    batch_size=32,
    initial_epoch=0,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', mode='auto', ),
    ],
    verbose=2
)


Y_prob_train = clf.predict(train.X)
Y_prob_test = clf.predict(test.X)


print  average_precision_score(test.Y[:,test.Y.sum(0)!=0], Y_prob_test[:,test.Y.sum(0)!=0], average='micro')