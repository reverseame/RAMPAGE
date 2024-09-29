from RAMPAGE.Classifier import Classifier
from RAMPAGE.Result import Result
from RAMPAGE.DataElement import DataElement

from common.resultCommon import ResultCommon
from common.commonData import CommonData

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
import tensorflow as tf
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy

class LSTM_example(Classifier):

    model = None
    epochs = CommonData.epochs
    batch_size = CommonData.batch_size
    model_name = "LSTM_example"
    maxlen = CommonData.maxlen

    save_file = "./classifiers/models/"+model_name+".keras"

    def __init__(self) -> None:

        self.model = Sequential (name=self.model_name)
        self.model.add(Embedding(256,128,input_length=CommonData.maxlen))
        self.model.add(LSTM(units=128, unroll=True))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, weight_decay=0.001)
        
        self.times = 0
        
        self.model.compile(loss ='binary_crossentropy',
        optimizer=adam, metrics=CommonData.metrics)

        #self.model.summary()
        
    def train(self, train:set, validation:set):

        x = []
        y = []

        for dataelement in train:
            dataVec = []
            for character in dataelement.domain:
                dataVec.append(ord(character) - 33) # 33 is the first printeable char (is !)
            x.append(dataVec)
            if dataelement.isDGA:
                y.append(1)
            else:
                y.append(0)

        x_val = []
        y_val = []

        for dataelement in validation:
            dataVec = []
            for character in dataelement.domain:
                    dataVec.append(ord(character) - 33) # 33 is the first printeable char (is !)
            x_val.append(dataVec)
            if dataelement.isDGA:
                y_val.append(1)
            else:
                y_val.append(0)

        x = pad_sequences(x, padding='post', maxlen=self.maxlen)

        x = numpy.array(x, dtype=float)
        y = numpy.array(y, dtype=float)

        x_val = pad_sequences(x_val, padding='post', maxlen=self.maxlen)

        x_val = numpy.array(x_val, dtype=float)
        y_val = numpy.array(y_val, dtype=float)

        checkpoint=ModelCheckpoint(self.save_file, monitor='val_accuracy',
                                   verbose=CommonData.verbose, save_best_only=True,
                                   save_weights_only=False, mode='auto')

        history = self.model.fit(x, y, epochs=self.epochs,
                                 verbose = CommonData.verbose,
                                 validation_data=(x_val, y_val),
                                 batch_size=self.batch_size,
                                 callbacks=[checkpoint])

    def test(self, test:set) -> Result:
        
        best_model = load_model(self.save_file)

        x_test = []
        y_test = []

        for dataelement in test:
            dataVec = []
            for character in dataelement.domain:
                    dataVec.append(ord(character) - 33) # 33 is the first printeable char (is !)
            x_test.append(dataVec)
            if dataelement.isDGA:
                y_test.append(1)
            else:
                y_test.append(0)

        x_test = pad_sequences(x_test, padding='post', maxlen=self.maxlen)

        x_test = numpy.array(x_test, dtype=float)
        y_test = numpy.array(y_test, dtype=float)
        
        scores = best_model.evaluate(x_test, y_test, batch_size=10)
        #print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

        return ResultCommon(scores[1]*100, scores[2]*100, scores[3]*100, scores[4], scores[5], scores[6], scores[7], scores[8])
