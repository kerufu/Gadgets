import time
import tensorflow
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Reshape, Dropout
from tensorflow.keras import Model
import numpy
import cv2

from myP.settings import STATICFILES_DIRS, batchSize
from learningModel.datasetManager import data_manager

imageSize = 256


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
                            use_bias=False, activation='tanh')
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


class pGenerationWoker():
    generatorPath = "learningModel/savedModel/pGeneration/Generator"
    discriminatorPath = "learningModel/savedModel/pGeneration/Discriminator"

    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        try:
            self.generator.load_weights(self.generatorPath)
            self.discriminator.load_weights(self.discriminatorPath)
        except:
            pass
        self.trainDataReady = False
        self.cross_entropy = tensorflow.keras.losses.BinaryCrossentropy(
            from_logits=True)
        self.generatorOptimizer = tensorflow.keras.optimizers.Adam()
        self.discriminatorOptimizer = tensorflow.keras.optimizers.Adam()

    def getDiscriminatorLoss(self, real_output, fake_output):
        real_loss = self.cross_entropy(
            tensorflow.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(
            tensorflow.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def getGeneratorLoss(self, fake_output):
        return self.cross_entropy(tensorflow.ones_like(fake_output), fake_output)

    @tensorflow.function
    def train_step(self, images):
        # for _ in range(5):
        #     with tensorflow.GradientTape() as disc_tape:
        #         noise = tensorflow.random.normal([batchSize, 100])
        #         generated_images = self.generator(noise, training=True)
        #         real_output = self.discriminator(images, training=True)
        #         fake_output = self.discriminator(generated_images, training=True)
        #         disc_loss = self.getDiscriminatorLoss(real_output, fake_output)
        #         gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        #         self.discriminatorOptimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        # for _ in range(2):
        #     with tensorflow.GradientTape() as gen_tape:
        #         noise = tensorflow.random.normal([batchSize, 100])
        #         generated_images = self.generator(noise, training=True)
        #         fake_output = self.discriminator(generated_images, training=True)
        #         gen_loss = self.getGeneratorLoss(fake_output)
        #         gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        #         self.generatorOptimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        noise = tensorflow.random.normal([batchSize, 100])
        with tensorflow.GradientTape() as gen_tape, tensorflow.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.getGeneratorLoss(fake_output)
            disc_loss = self.getDiscriminatorLoss(real_output, fake_output)
            gradients_of_generator = gen_tape.gradient(
                gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables)
            self.generatorOptimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminatorOptimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, epochs=2):

        for epoch in range(epochs):
            self.trainDataReady = False
            self.getData()
            if self.trainDataReady:
                start = time.time()
                for image_batch in self.trainData:
                    self.train_step(image_batch)
                print('Time for epoch {} is {} sec'.format(
                    epoch + 1, time.time()-start))
                self.generator.save(self.generatorPath)
                self.discriminator.save(self.discriminatorPath)

    def resize(self, img):
        return cv2.resize(img, (imageSize, imageSize),
                          interpolation=cv2.INTER_AREA)

    def getData(self):
        data_manager.retrieveData()
        trainDataLable = data_manager.getDataset()
        self.trainData = []
        for index in range(len(trainDataLable[0])):
            if trainDataLable[1][index] == 1:
                img = self.resize(trainDataLable[0][index])
                self.trainData.append(img)
        if len(self.trainData) > 0:
            self.trainData = tensorflow.data.Dataset.from_tensor_slices(
                self.trainData).shuffle(batchSize).batch(batchSize)
            self.trainDataReady = True
        else:
            print("no data")

    def generateImg(self, num=1):
        img = self.generator(tensorflow.random.normal(
            [num, 100]), training=False)
        img = img * 127.5 + 127.5
        for index in range(num):
            cv2.imwrite(STATICFILES_DIRS[2] + "/generated" +
                        str(index)+".jpg", numpy.array(img[index, :, :, :]))
