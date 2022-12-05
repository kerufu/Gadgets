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

class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = [
            Conv2D(32, 2, activation='relu'),
            Dropout(0.3),
            Conv2D(64, 2, activation='relu'),
            Dropout(0.3),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(featureVectorLength)
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class DiscriminatorOnEncoder(Model):
    def __init__(self):
        super(DiscriminatorOnEncoder, self).__init__()
        self.model = [
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
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


class DiscriminatorOnDecoder(Model):
    def __init__(self):
        super(DiscriminatorOnDecoder, self).__init__()
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


class CAAEWoker():
    EPath = "learningModel/savedModel/CAAE/Encoder"
    DOEPath = "learningModel/savedModel/CAAE/DiscriminatorOnEncoder"
    DPath = "learningModel/savedModel/CAAE/Decoder"
    DODPath = "learningModel/savedModel/CAAE/DiscriminatorOnDecoder"

    def __init__(self):
        self.E = Encoder()
        self.DOE = DiscriminatorOnEncoder()
        self.D = Decoder()
        self.DOD = DiscriminatorOnDecoder()
        try:
            self.E.load_weights(self.EPath)
            self.DOE.load_weights(self.DOEPath)
            self.D.load_weights(self.DPath)
            self.DOD.load_weights(self.DODPath)
        except:
            pass
        self.trainDataReady = False
        self.cross_entropy = tensorflow.keras.losses.BinaryCrossentropy(
            from_logits=True)
        self.mse = tensorflow.keras.losses.MeanSquaredError()
        self.EOptimizer = tensorflow.keras.optimizers.Adam()
        self.DOEOptimizer = tensorflow.keras.optimizers.Adam()
        self.DOptimizer = tensorflow.keras.optimizers.Adam()
        self.DODOptimizer = tensorflow.keras.optimizers.Adam()

    def getELoss(self, DOE_fake_output, input_image, D_output):
        discriminator_loss = self.cross_entropy(tensorflow.ones_like(DOE_fake_output), DOE_fake_output)
        image_loss = self.mse(input_image, D_output)
        return discriminator_loss + image_loss

    def getDOELoss(self, DOE_real_output, DOE_fake_output):
        real_loss = self.cross_entropy(
            tensorflow.ones_like(DOE_real_output), DOE_real_output)
        fake_loss = self.cross_entropy(
            tensorflow.zeros_like(DOE_fake_output), DOE_fake_output)
        return real_loss + fake_loss

    def getDLoss(self, DOD_fake_output, input_image, D_output):
        discriminator_loss = self.cross_entropy(tensorflow.ones_like(DOD_fake_output), DOD_fake_output)
        image_loss = self.mse(input_image, D_output)
        return discriminator_loss + image_loss

    def getDODLoss(self, DOD_real_output, DOD_fake_output):
        real_loss = self.cross_entropy(
            tensorflow.ones_like(DOD_real_output), DOD_real_output)
        fake_loss = self.cross_entropy(
            tensorflow.zeros_like(DOD_fake_output), DOD_fake_output)
        return real_loss + fake_loss

    @tensorflow.function
    def train_step(self, images, conditions, EIteration=1, DIteration=1, DOEIteration=1, DODIteration=1):

        for _ in range(DOEIteration):
            with tensorflow.GradientTape() as DOE_tape:
                noise = tensorflow.random.normal([batchSize, featureVectorLength])
                encoded_feature_vector = self.E(images, training=True)
                DOE_real_output = self.DOE(noise, training=True)
                DOE_fake_output = self.DOE(encoded_feature_vector, training=True)
                DOE_loss = self.getDOELoss(DOE_real_output, DOE_fake_output)
                gradients_of_DOE = DOE_tape.gradient(DOE_loss, self.DOE.trainable_variables)
                self.DOEOptimizer.apply_gradients(zip(gradients_of_DOE, self.DOE.trainable_variables))
        for _ in range(EIteration):
            with tensorflow.GradientTape() as E_tape:
                encoded_feature_vector = self.E(images, training=True)
                DOE_fake_output = self.DOE(encoded_feature_vector, training=True)
                conditional_encoded_feature_vector = tensorflow.concat([encoded_feature_vector, conditions], 1)
                decoded_images = self.D(conditional_encoded_feature_vector, training=True)
                E_loss = self.getELoss(DOE_fake_output, images, decoded_images)
                gradients_of_E = E_tape.gradient(E_loss, self.E.trainable_variables)
                self.EOptimizer.apply_gradients(zip(gradients_of_E, self.E.trainable_variables))
        for _ in range(DODIteration):
            with tensorflow.GradientTape() as DOD_tape:
                encoded_feature_vector = self.E(images, training=True)
                conditional_encoded_feature_vector = tensorflow.concat([encoded_feature_vector, conditions], 1)
                decoded_images = self.D(conditional_encoded_feature_vector, training=True)
                DOD_real_output = self.DOD(images, training=True)
                DOD_fake_output = self.DOD(decoded_images, training=True)
                DOD_loss = self.getDODLoss(DOD_real_output, DOD_fake_output)
                gradients_of_DOD = DOD_tape.gradient(DOD_loss, self.DOD.trainable_variables)
                self.DODOptimizer.apply_gradients(zip(gradients_of_DOD, self.DOD.trainable_variables))
        for _ in range(DIteration):
            with tensorflow.GradientTape() as D_tape:
                encoded_feature_vector = self.E(images, training=True)
                conditional_encoded_feature_vector = tensorflow.concat([encoded_feature_vector, conditions], 1)
                decoded_images = self.D(conditional_encoded_feature_vector, training=True)
                DOD_fake_output = self.DOD(decoded_images, training=True)
                D_loss = self.getDLoss(DOD_fake_output, images, decoded_images)
                gradients_of_D = D_tape.gradient(D_loss, self.D.trainable_variables)
                self.DOptimizer.apply_gradients(zip(gradients_of_D, self.D.trainable_variables))

    def train(self, epochs=1):
        for epoch in range(epochs):
            self.trainDataReady = False
            self.getData()
            if self.trainDataReady:
                start = time.time()
                for batch in self.trainData:
                    images, conditions = batch
                    conditions = tensorflow.expand_dims(tensorflow.cast(conditions, dtype = tensorflow.float32), axis=1)
                    self.train_step(images, conditions)
                print('Time for epoch {} is {} sec'.format(
                    epoch + 1, time.time()-start))
                self.E.save(self.EPath)
                self.DOE.save(self.DOEPath)
                self.D.save(self.DPath)
                self.DOD.save(self.DODPath)

    def getData(self):
        data_manager.retrieveData()
        trainDataLable = data_manager.getDataset()
        self.trainData = []
        condition = []
        for index in range(len(trainDataLable[0])):
            img = cv2.resize(trainDataLable[0][index], (imageSize, imageSize),
                        interpolation=cv2.INTER_AREA)
            self.trainData.append(img)
            condition.append(trainDataLable[1][index])
        if len(self.trainData) > 0:
            self.trainData = tensorflow.data.Dataset.from_tensor_slices(
                (self.trainData, condition)).shuffle(batchSize).batch(batchSize)
            self.trainDataReady = True
        else:
            print("no data")

    def generateImg(self, num=1):
        img = self.D(tensorflow.random.normal(
            [num, featureVectorLength+1]))
        img = img * 255
        for index in range(num):
            cv2.imwrite(STATICFILES_DIRS[2] + "/generated" +
                        str(index)+".jpg", numpy.array(img[index, :, :, :]))
