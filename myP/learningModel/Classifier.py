import tensorflow
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras import Model
import numpy
import cv2

from myP.settings import batchSize
from learningModel.datasetManager import data_manager

imageSize = 128


class Classifier(Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = [
            Conv2D(64, 2, activation='relu'),
            Dropout(0.3),
            Conv2D(32, 2, activation='relu'),
            Dropout(0.3),
            Flatten(),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class ClassifierWoker():
    modelPath = "learningModel/savedModel/Classifier"

    def __init__(self):
        self.model = Classifier()
        try:
            self.model.load_weights(self.modelPath)
        except:
            pass
        self.model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(),
            loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )
        self.trainDataReady = False

    def getData(self):
        data_manager.retrieveData()
        self.trainData, self.trainLabel = data_manager.getDataset()
        for index in range(len(self.trainData)):
            self.trainData[index] = self.resize(self.trainData[index])
        if len(self.trainLabel) > 0:
            self.trainData = numpy.float16(self.trainData)
            self.trainLabel = numpy.float16(self.trainLabel)
            self.trainDataReady = True
        else:
            print("no data")

    def resize(self, img):
        return cv2.resize(img, (imageSize, imageSize),
                          interpolation=cv2.INTER_AREA)

    def train(self, epochs=1):
        for _ in range(epochs):
            self.trainDataReady = False
            self.getData()
            if self.trainDataReady:
                self.model.fit(
                    self.trainData,
                    self.trainLabel,
                    batch_size=batchSize,
                    epochs=1,
                )
                self.model.save(self.modelPath)

    def predict(self, path):
        img = data_manager.preprocessData(path)
        img = self.resize(img)
        logit = self.model.predict(numpy.float16([img]))
        prediction = float(tensorflow.nn.sigmoid(logit)[0])
        print(logit, prediction)
        return prediction
