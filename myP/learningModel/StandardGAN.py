import time
import tensorflow
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Reshape, Dropout
from tensorflow.keras import Model
import numpy
import cv2

from myP.settings import STATICFILES_DIRS, batchSize
from learningModel.datasetManager import data_manager

imageSize = 256
featureVectorLength = 100

class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = [
            Dense(imageSize*imageSize),
            BatchNormalization(),
            LeakyReLU(),
            Reshape((imageSize//8, imageSize//8, 64)),
            Conv2DTranspose(32, 3, strides=(2, 2),
                            padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(16, 3, strides=(2, 2),
                            padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(3, 3, strides=(2, 2), padding='same',
                            use_bias=False, activation='sigmoid')
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = [
            Conv2D(32, 2, activation='relu'),
            Dropout(0.3),
            Conv2D(64, 2, activation='relu'),
            Dropout(0.3),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class StandardGANWoker():
    GPath = "learningModel/savedModel/StandardGAN/Generator"
    DPath = "learningModel/savedModel/StandardGAN/Discriminator"

    def __init__(self):
        self.G = Generator()
        self.D = Discriminator()
        try:
            self.G.load_weights(self.GPath)
            self.D.load_weights(self.DPath)
        except:
            pass
        self.trainDataReady = False
        self.cross_entropy = tensorflow.keras.losses.BinaryCrossentropy(
            from_logits=True)
        self.GOptimizer = tensorflow.keras.optimizers.Adam()
        self.DOptimizer = tensorflow.keras.optimizers.Adam()

    def getDLoss(self, real_output, fake_output):
        real_loss = self.cross_entropy(
            tensorflow.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(
            tensorflow.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def getGLoss(self, fake_output):
        return self.cross_entropy(tensorflow.ones_like(fake_output), fake_output)

    @tensorflow.function
    def train_step(self, images, GIteration=1, DIteration=1):
        for _ in range(DIteration):
            with tensorflow.GradientTape() as D_tape:
                noise = tensorflow.random.normal([batchSize, featureVectorLength])
                generated_images = self.G(noise, training=True)
                real_output = self.D(images, training=True)
                fake_output = self.D(generated_images, training=True)
                D_loss = self.getDLoss(real_output, fake_output)
                gradients_of_D = D_tape.gradient(D_loss, self.D.trainable_variables)
                self.DOptimizer.apply_gradients(zip(gradients_of_D, self.D.trainable_variables))
        for _ in range(GIteration):
            with tensorflow.GradientTape() as G_tape:
                noise = tensorflow.random.normal([batchSize, featureVectorLength])
                generated_images = self.G(noise, training=True)
                fake_output = self.D(generated_images, training=True)
                G_loss = self.getGLoss(fake_output)
                gradients_of_G = G_tape.gradient(G_loss, self.G.trainable_variables)
                self.GOptimizer.apply_gradients(zip(gradients_of_G, self.G.trainable_variables))

    def train(self, epochs=1):
        for epoch in range(epochs):
            self.trainDataReady = False
            self.getData()
            if self.trainDataReady:
                start = time.time()
                for image_batch in self.trainData:
                    self.train_step(image_batch)
                print('Time for epoch {} is {} sec'.format(
                    epoch + 1, time.time()-start))
                self.G.save(self.GPath)
                self.D.save(self.DPath)

    def getData(self):
        data_manager.retrieveData()
        trainDataLable = data_manager.getDataset()
        self.trainData = []
        for index in range(len(trainDataLable[0])):
            if trainDataLable[1][index] == 1:
                img = cv2.resize(trainDataLable[0][index], (imageSize, imageSize),
                          interpolation=cv2.INTER_AREA)
                self.trainData.append(img)
        if len(self.trainData) > 0:
            self.trainData = tensorflow.data.Dataset.from_tensor_slices(
                self.trainData).shuffle(batchSize).batch(batchSize)
            self.trainDataReady = True
        else:
            print("no data")

    def generateImg(self, num=1):
        img = self.generator(tensorflow.random.normal(
            [num, featureVectorLength]))
        img = img * 255
        for index in range(num):
            cv2.imwrite(STATICFILES_DIRS[2] + "/generated" +
                        str(index)+".jpg", numpy.array(img[index, :, :, :]))
