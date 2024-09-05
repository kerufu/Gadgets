import time
import random
from termcolor import cprint
import pickle
import os

import tensorflow as tf
import numpy as np

import setting

class encoder(tf.keras.Model):
    def __init__(self):
        super(encoder, self).__init__()
        self.model = [
            tf.keras.layers.Dense(128, activation='selu',
                                  kernel_regularizer=tf.keras.regularizers.L1L2()
                                  ),
            tf.keras.layers.Dense(64, activation='selu',
                                  kernel_regularizer=tf.keras.regularizers.L1L2()
                                  ),
            tf.keras.layers.Dense(32, activation='selu',
                                  kernel_regularizer=tf.keras.regularizers.L1L2(),
                                  ),
            tf.keras.layers.Dense(setting.embedding_dimension,
                                  activity_regularizer=tf.keras.regularizers.L1L2(),
                                  kernel_regularizer=tf.keras.regularizers.L1L2()
                                  )
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    
class decoder(tf.keras.Model):
    def __init__(self):
        super(decoder, self).__init__()
        self.model = [
            tf.keras.layers.Dense(32, activation='selu',
                                  kernel_regularizer=tf.keras.regularizers.L1L2()
                                  ),
            tf.keras.layers.Dense(64, activation='selu',
                                  kernel_regularizer=tf.keras.regularizers.L1L2()
                                  ),
            tf.keras.layers.Dense(128, activation='selu',
                                  kernel_regularizer=tf.keras.regularizers.L1L2()
                                  ),
            tf.keras.layers.Dense(setting.num_embedding_variable,
                                  kernel_regularizer=tf.keras.regularizers.L1L2()
                                  )
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x

class discriminator(tf.keras.Model):
    def __init__(self):
        super(discriminator, self).__init__()
        self.model = [
            tf.keras.layers.Dense(32, activation='selu',
                                  kernel_regularizer=tf.keras.regularizers.L1L2()
                                  ),
            tf.keras.layers.Dense(1,
                                  kernel_regularizer=tf.keras.regularizers.L1L2())
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    
class ae_classifier(tf.keras.Model):
    def __init__(self):
        super(ae_classifier, self).__init__()
        self.model = [
            tf.keras.layers.Dense(32, activation='selu',
                                  kernel_regularizer=tf.keras.regularizers.L1L2()),
            tf.keras.layers.Dense(1,
                                  kernel_regularizer=tf.keras.regularizers.L1L2())
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    
class autoencoder_woker():
    
    def __init__(self, dsw, enable_discriminator=False, ae_iteration=1, c_iteration=1, dis_iteration=1):
        self.enable_discriminator = enable_discriminator
        self.ae_iteration = ae_iteration
        self.c_iteration = c_iteration
        self.dis_iteration = dis_iteration
        self.dsw = dsw

        self.e = encoder()
        self.d = decoder()
        self.c = ae_classifier()
        if self.enable_discriminator:
            self.dis = discriminator()
        try:
            self.e.load_weights(setting.encoder_path)
            self.d.load_weights(setting.decoder_path)
            self.c.load_weights(setting.ae_classifier_path)
            if self.enable_discriminator:
                self.dis.load_weights(setting.discriminator_path)
            print("Saved audoencoder weight loaded")
        except:
            print("Saved audoencoder weight not found")

        self.encoder_optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
        self.decoder_optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
        self.classifier_optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
        if self.enable_discriminator:
            self.discriminator_optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)

        self.reconstruction_loss = tf.keras.losses.MeanSquaredError()
        self.classifying_loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True)
        if self.enable_discriminator:
            self.discriminator_loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True)

        self.train_ae_metric = tf.keras.metrics.MeanSquaredError()
        self.test_ae_metric = tf.keras.metrics.MeanSquaredError()
        self.train_c_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)
        self.test_c_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)
        if self.enable_discriminator:
            self.train_d_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)
            self.test_d_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)

        self.false_positive = tf.keras.metrics.FalsePositives(thresholds=0)
        self.precision = tf.keras.metrics.Precision(thresholds=0)
        self.recall = tf.keras.metrics.Recall(thresholds=0)

    @tf.function
    def train_step(self, batch):
        embedding, non_embedding, label = batch["e"], batch["n"], batch["l"]
        for _ in range(self.ae_iteration):
            with tf.GradientTape() as e_tape:
                with tf.GradientTape() as d_tape:
                    features = self.e(embedding, training=True)
                    decoder_output = self.d(tf.concat([features, non_embedding], 1), training=True)
                
                    r_loss = self.reconstruction_loss(embedding, decoder_output)
                    e_loss = r_loss + tf.add_n(self.e.losses)

                    c_output = self.c(features, training=True)
                    e_loss += setting.ae_classifier_weight * self.classifying_loss(label, c_output)

                    if self.enable_discriminator:
                        dis_output = self.dis(features, training=True)
                        e_loss += setting.discriminator_weight * self.discriminator_loss(tf.ones_like(dis_output), dis_output)

                    d_loss = r_loss + tf.add_n(self.d.losses)

            gradients_of_e = e_tape.gradient(e_loss, self.e.trainable_variables)
            self.encoder_optimizer.apply_gradients(zip(gradients_of_e, self.e.trainable_variables))

            gradients_of_d = d_tape.gradient(d_loss, self.d.trainable_variables)
            self.decoder_optimizer.apply_gradients(zip(gradients_of_d, self.d.trainable_variables))

            self.train_ae_metric.update_state(embedding, decoder_output)

        for _ in range(self.c_iteration):
            with tf.GradientTape() as c_tape:
                features = self.e(embedding, training=True)
                c_output = self.c(features, training=True)
                c_loss = self.classifying_loss(label, c_output)
                c_loss += tf.add_n(self.c.losses)

            gradients_of_c = c_tape.gradient(c_loss, self.c.trainable_variables)
            self.classifier_optimizer.apply_gradients(zip(gradients_of_c, self.c.trainable_variables))

            self.train_c_metric.update_state(label, c_output)
        
        if self.enable_discriminator:
            for _ in range(self.dis_iteration):
                with tf.GradientTape() as dis_tape:
                    features = self.e(embedding, training=True)
                    noise = tf.random.normal([setting.batch_size, setting.embedding_dimension])
                    noise_output = self.dis(noise, training=True)
                    encoded_output = self.dis(features, training=True)

                    dis_loss = self.discriminator_loss(tf.ones_like(noise_output), noise_output)
                    dis_loss += self.discriminator_loss(tf.zeros_like(encoded_output), encoded_output)
                    dis_loss += tf.add_n(self.dis.losses)

                gradients_of_dis = dis_tape.gradient(dis_loss, self.dis.trainable_variables)
                self.discriminator_optimizer.apply_gradients(zip(gradients_of_dis, self.dis.trainable_variables))

                self.train_d_metric.update_state(tf.ones_like(noise_output), noise_output)
                self.train_d_metric.update_state(tf.zeros_like(encoded_output), encoded_output)

        return decoder_output

    @tf.function
    def test_step(self, batch):
        embedding, non_embedding, label = batch["e"], batch["n"], batch["l"]
        features = self.e(embedding, training=False)
        decoder_output = self.d(tf.concat([features, non_embedding], 1), training=False)
        self.test_ae_metric.update_state(embedding, decoder_output)

        c_output = self.c(features, training=True)
        self.test_c_metric.update_state(label, c_output)

        self.false_positive.update_state(label, c_output)
        self.precision.update_state(label, c_output)
        self.recall.update_state(label, c_output)

        if self.enable_discriminator:
            noise = tf.random.normal([setting.batch_size, setting.embedding_dimension])
            noise_output = self.dis(noise, training=False)
            encoded_output = self.dis(features, training=False)
            
            self.test_d_metric.update_state(tf.ones_like(noise_output), noise_output)
            self.test_d_metric.update_state(tf.zeros_like(encoded_output), encoded_output)
        
        return decoder_output

    def train(self, epochs=1):

        embedding_data = self.dsw.get_embedding_data()
        non_embedding_data = self.dsw.get_non_embedding_data()

        legitimate_lable = non_embedding_data[:,-1].astype(int)

        train_dataset = tf.data.Dataset.from_tensor_slices({
            "e": embedding_data[:setting.train_dataset_size],
            "n": tf.cast(non_embedding_data[:setting.train_dataset_size,:-1],tf.float32),
            "l": legitimate_lable[:setting.train_dataset_size]
            })
        test_dataset = tf.data.Dataset.from_tensor_slices({
            "e": embedding_data[setting.train_dataset_size:],
            "n": tf.cast(non_embedding_data[setting.train_dataset_size:,:-1],tf.float32),
            "l": legitimate_lable[setting.train_dataset_size:]
            })

        for epoch in range(epochs):
            start = time.time()

            self.train_ae_metric.reset_state()
            self.test_ae_metric.reset_state()
            self.train_c_metric.reset_state()
            self.test_c_metric.reset_state()
            if self.enable_discriminator:
                self.train_d_metric.reset_state()
                self.test_d_metric.reset_state()

            self.false_positive.reset_state()
            self.precision.reset_state()
            self.recall.reset_state()

            train_dataset = train_dataset.shuffle(train_dataset.cardinality())
            test_dataset = test_dataset.shuffle(test_dataset.cardinality())

            for batch in train_dataset.batch(setting.batch_size):
                train_input = batch["e"]
                train_output = self.train_step(batch)
            for batch in test_dataset.batch(setting.batch_size):
                test_input = batch["e"]
                test_output = self.test_step(batch)
                
            cprint('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start), 'red')

            print("Train AutoEncoder Loss:" + str(self.train_ae_metric.result().numpy()))
            print("Test AutoEncoder Loss:" + str(self.test_ae_metric.result().numpy()))

            tf.print("Train True Sample: ", train_input[0, :])
            tf.print("Train Decoded Sample: ", train_output[0, :])

            tf.print("Test True Sample: ", test_input[0, :])
            tf.print("Test Decoded Sample: ", test_output[0, :])

            print("Train Classifier Accuracy:" + str(self.train_c_metric.result().numpy()))
            print("Test Classifier Accuracy:" + str(self.test_c_metric.result().numpy()))

            print("Test Classifier False Positive Rate:" + str(self.false_positive.result().numpy()/setting.test_dataset_size))
            print("Test Classifier Precision:" + str(self.precision.result().numpy()))
            print("Test Classifier Recall:" + str(self.recall.result().numpy()))

            if self.enable_discriminator:
                print("Train Discriminator Accuracy:" + str(self.train_d_metric.result().numpy()))
                print("Test Discriminator Accuracy:" + str(self.test_d_metric.result().numpy()))

            self.e.save(setting.encoder_path)
            self.d.save(setting.decoder_path)
            self.c.save(setting.ae_classifier_path)
            if self.enable_discriminator:
                self.dis.save(setting.discriminator_path)

    def test(self, indexes):

        self.train_ae_metric.reset_state()
        self.test_ae_metric.reset_state()
        self.train_c_metric.reset_state()
        self.test_c_metric.reset_state()
        if self.enable_discriminator:
            self.train_d_metric.reset_state()
            self.test_d_metric.reset_state()

        self.false_positive.reset_state()
        self.precision.reset_state()
        self.recall.reset_state()

        indexes = random.choices(indexes, k=setting.test_dataset_size)
        embedding_data = self.dsw.get_embedding_data()[indexes]
        non_embedding_data = self.dsw.get_non_embedding_data()[indexes]
        legitimate_lable = non_embedding_data[:,-1].astype(int)
        non_embedding_data = non_embedding_data[:,:-1]

        features = self.e(embedding_data, training=False)
        decoder_output = self.d(tf.concat([features, non_embedding_data], 1), training=False)
        self.test_ae_metric.update_state(embedding_data, decoder_output)

        c_output = self.c(features, training=True)
        self.test_c_metric.update_state(legitimate_lable, c_output)

        self.false_positive.update_state(legitimate_lable, c_output)
        self.precision.update_state(legitimate_lable, c_output)
        self.recall.update_state(legitimate_lable, c_output)

        if self.enable_discriminator:
            noise = tf.random.normal([setting.test_dataset_size, setting.embedding_dimension])
            noise_output = self.dis(noise, training=False)
            encoded_output = self.dis(features, training=False)
            
            self.test_d_metric.update_state(tf.ones_like(noise_output), noise_output)
            self.test_d_metric.update_state(tf.zeros_like(encoded_output), encoded_output)

        sample_index = random.randint(0, setting.test_dataset_size-1)
        embedding_sample = embedding_data[sample_index, :]
        non_embedding_sample = non_embedding_data[sample_index, :]
        embedding_sample = tf.reshape(embedding_sample, shape=(1, setting.num_embedding_variable))
        non_embedding_sample = tf.reshape(non_embedding_sample, shape=(1, setting.num_non_embedding_variable-1))
        non_embedding_sample = tf.cast(non_embedding_sample, tf.float32) 
        sample_features = self.e(embedding_sample, training=False)
        decoded_sample = self.d(tf.concat([sample_features, non_embedding_sample], 1), training=False)

        print("Test AutoEncoder Loss:" + str(self.test_ae_metric.result().numpy()))
        tf.print("Test True Sample: ", embedding_sample)
        tf.print("Test Decoded Sample: ", decoded_sample)
        print("Test Classifier Accuracy:" + str(self.test_c_metric.result().numpy()))

        print("Test Classifier False Positive Rate:" + str(self.false_positive.result().numpy()/setting.test_dataset_size))
        print("Test Classifier Precision:" + str(self.precision.result().numpy()))
        print("Test Classifier Recall:" + str(self.recall.result().numpy()))

        if self.enable_discriminator:
            print("Test Discriminator Accuracy:" + str(self.test_d_metric.result().numpy()))
            tf.print("Test Noise Mean: ", tf.math.reduce_mean(noise))
            tf.print("Test Encoded Features Mean: ", tf.math.reduce_mean(features))
            tf.print("Test Noise STD: ", tf.math.reduce_std(noise))
            tf.print("Test Encoded Features STD: ", tf.math.reduce_std(features))

    def anormal_analysis(self, renew_indexes=False):
        anormal_indexes = None
        normal_indexes = None
        losses_mean, losses_std = None, None
        if renew_indexes or (not os.path.isfile(setting.anormal_indexes_path)) or (not os.path.isfile(setting.normal_indexes_path)):

            embedding_data = self.dsw.get_embedding_data()
            non_embedding_data = self.dsw.get_non_embedding_data()[:,:-1]

            full_dataset = tf.data.Dataset.from_tensor_slices({
                "e": embedding_data,
                "n": tf.cast(non_embedding_data,tf.float32)
                })
            full_dataset = full_dataset.batch(1)

            @tf.function
            def step(sample):
                embedding, non_embedding = sample["e"], sample["n"]
                features = self.e(embedding, training=False)
                decoder_output = self.d(tf.concat([features, non_embedding], 1), training=False)
                return self.reconstruction_loss(embedding, decoder_output)

            losses = []
            for sample in full_dataset:
                losses.append(step(sample))
            
            losses = np.array(losses)
            losses_mean = np.mean(losses)
            losses_std = np.std(losses)
            threshold = losses_mean + losses_std * setting.anormal_deviate
            anormal_indexes = np.where(losses>=threshold)[0]
            normal_indexes = np.where(losses<threshold)[0]
            with open(setting.anormal_indexes_path, 'wb') as fp:
                pickle.dump(anormal_indexes, fp)
            with open(setting.normal_indexes_path, 'wb') as fp:
                pickle.dump(normal_indexes, fp)
            
            with open(setting.loss_mean_std_path, 'wb') as fp:
                pickle.dump([losses_mean, losses_std], fp)

        else:
            with open(setting.anormal_indexes_path, 'rb') as fp:
                anormal_indexes = pickle.load(fp)
            with open(setting.normal_indexes_path, 'rb') as fp:
                normal_indexes = pickle.load(fp)
            with open(setting.loss_mean_std_path, 'rb') as fp:
                losses_mean, losses_std = pickle.load(fp)
        
        return anormal_indexes, normal_indexes, losses_mean, losses_std



    