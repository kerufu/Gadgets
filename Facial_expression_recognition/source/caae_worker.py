import time

import tensorflow as tf
import cv2
import numpy as np
from termcolor import cprint

import setting
from model import encoder, decoder, encoder_discriminator, decoder_discriminator, classifier

class caae_worker():

    def __init__(self, ae_iteration=1, ed_iteration=1, dd_iteration=1, c_iteration=1):
        self.ae_iteration = ae_iteration
        self.ed_iteration = ed_iteration
        self.dd_iteration = dd_iteration
        self.c_iteration = c_iteration

        self.e = encoder()
        self.d = decoder()
        self.ed = encoder_discriminator()
        self.dd = decoder_discriminator()
        self.c = classifier()
        
        try:
            self.e.load_weights(setting.encoder_path)
            self.d.load_weights(setting.decoder_path)
            self.ed.load_weights(setting.encoder_discriminator_path)
            self.dd.load_weights(setting.decoder_discriminator_path)
            self.c.load_weights(setting.classifier_path)
            print("caae model weight loaded")
        except:
            print("caae model weight not found")

        self.e_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)
        self.d_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)
        self.ed_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)
        self.dd_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)

        self.mse = tf.keras.losses.MeanSquaredError()
        self.bfce = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True, label_smoothing=setting.label_smoothing_ratio)
        self.cfce = tf.keras.losses.CategoricalFocalCrossentropy(from_logits=True, label_smoothing=setting.label_smoothing_ratio)

        self.ae_train_metric = tf.keras.metrics.MeanSquaredError()
        self.ed_train_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)
        self.dd_train_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)
        self.c_train_metric = tf.keras.metrics.CategoricalAccuracy()

        self.ae_test_metric = tf.keras.metrics.MeanSquaredError()
        self.ed_test_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)
        self.dd_test_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)
        self.c_test_metric = tf.keras.metrics.CategoricalAccuracy()
   
    def get_e_loss(self, input_image, output_image, ed_fake, dd_fake, one_hot, c_pred):
        loss = self.mse(input_image, output_image)
        loss += self.bfce(tf.ones_like(ed_fake), ed_fake) * setting.caae_discriminator_weight
        loss += self.bfce(tf.ones_like(dd_fake), dd_fake) * setting.caae_discriminator_weight
        loss += self.cfce(one_hot, c_pred)
        return loss
    
    def get_d_loss(self, input_image, output_image, dd_fake):
        loss = self.mse(input_image, output_image)
        loss += self.bfce(tf.ones_like(dd_fake), dd_fake) * setting.caae_discriminator_weight
        return loss  + tf.add_n(self.d.losses)
    
    def get_ed_loss(self, target, output):
        return self.bfce(target, output) + tf.add_n(self.ed.losses)
    
    def get_dd_loss(self, target, output):
        return self.bfce(target, output) + tf.add_n(self.dd.losses)
    
    def get_c_loss(self, one_hot, c_pred):
        return self.cfce(one_hot, c_pred)
    
    @tf.function
    def train_encoder_discriminator(self, batch):
        image = batch["data"]
        with tf.GradientTape() as ed_tape_true:
            noise = tf.random.uniform([setting.batch_size, setting.feature_size], minval=-1, maxval=1)
            
            ed_true = self.ed(noise, training=True)

            ed_loss_true = self.get_ed_loss(tf.ones_like(ed_true), ed_true)

        ed_gradient = ed_tape_true.gradient(ed_loss_true, self.ed.trainable_variables)
        self.ed_opt.apply_gradients(zip(ed_gradient, self.ed.trainable_variables))

        with tf.GradientTape() as ed_tape_fake:
            features = self.e(image)
            
            ed_fake = self.ed(features, training=True)

            ed_loss_fake = self.get_ed_loss(tf.zeros_like(ed_fake), ed_fake)

        ed_gradient = ed_tape_fake.gradient(ed_loss_fake, self.ed.trainable_variables)
        self.ed_opt.apply_gradients(zip(ed_gradient, self.ed.trainable_variables))

        self.ed_train_metric.update_state(tf.ones_like(ed_true), ed_true)
        self.ed_train_metric.update_state(tf.zeros_like(ed_fake), ed_fake)

    @tf.function
    def train_decoder_discriminator(self, batch):
        image, condition = batch["data"], batch["condition_label"]
        with tf.GradientTape() as dd_tape_true:
            dd_true = self.dd(image, training=True)

            dd_loss_true = self.get_dd_loss(tf.ones_like(dd_true), dd_true)

        dd_gradient = dd_tape_true.gradient(dd_loss_true, self.dd.trainable_variables)
        self.dd_opt.apply_gradients(zip(dd_gradient, self.dd.trainable_variables))

        with tf.GradientTape() as dd_tape_fake:
            features = self.e(image)
            decoded_image = self.d(tf.concat([features, tf.cast(condition, tf.float32)], 1))

            dd_fake = self.dd(decoded_image, training=True)

            dd_loss_fake = self.get_dd_loss(tf.zeros_like(dd_fake), dd_fake)
        
        dd_gradient = dd_tape_fake.gradient(dd_loss_fake, self.dd.trainable_variables)
        self.dd_opt.apply_gradients(zip(dd_gradient, self.dd.trainable_variables))

        self.dd_train_metric.update_state(tf.ones_like(dd_true), dd_true)
        self.dd_train_metric.update_state(tf.zeros_like(dd_fake), dd_fake)

    @tf.function
    def train_autocoder(self, batch):
        image, condition, one_hot = batch["data"], batch["condition_label"], batch["one_hot_coding_label"]
        with tf.GradientTape() as e_tape:
            with tf.GradientTape() as d_tape:
                features = self.e(image, training=True)
                decoded_image = self.d(tf.concat([features, tf.cast(condition, tf.float32)], 1), training=True)
                ed_fake = self.ed(features)
                dd_fake = self.dd(decoded_image)
                c_pred = self.c(features)

                e_loss = self.get_e_loss(image, decoded_image, ed_fake, dd_fake, one_hot, c_pred)
                d_loss = self.get_d_loss(image, decoded_image, dd_fake)

        e_gradient = e_tape.gradient(e_loss, self.e.trainable_variables)
        self.e_opt.apply_gradients(zip(e_gradient, self.e.trainable_variables))
        d_gradient = d_tape.gradient(d_loss, self.d.trainable_variables)
        self.d_opt.apply_gradients(zip(d_gradient, self.d.trainable_variables))

        self.ae_train_metric.update_state(image, decoded_image)
    
    @tf.function
    def train_classifier(self, batch):
        image, one_hot = batch["data"], batch["one_hot_coding_label"]
        with tf.GradientTape() as c_tape:
            features = self.e(image)
            c_pred = self.c(features, training=True)

            c_loss = self.get_c_loss(one_hot, c_pred)
        
        c_gradient = c_tape.gradient(c_loss, self.c.trainable_variables)
        self.c_opt.apply_gradients(zip(c_gradient, self.c.trainable_variables))

        self.c_train_metric.update_state(one_hot, c_pred)

    @tf.function
    def test_step(self, batch):
        image, condition, one_hot = batch["data"], batch["condition_label"], batch["one_hot_coding_label"]
        noise = tf.random.uniform([setting.batch_size, setting.feature_size], minval=-1, maxval=1)

        features = self.e(image)
        decoded_image = self.d(tf.concat([features, tf.cast(condition, tf.float32)], 1))
        ed_true = self.ed(noise)
        ed_fake = self.ed(features)
        dd_true = self.dd(image)
        dd_fake = self.dd(decoded_image)
        c_pred = self.c(features)

        self.ed_test_metric.update_state(tf.ones_like(ed_true), ed_true)
        self.ed_test_metric.update_state(tf.zeros_like(ed_fake), ed_fake)
        self.dd_test_metric.update_state(tf.ones_like(dd_true), dd_true)
        self.dd_test_metric.update_state(tf.zeros_like(dd_fake), dd_fake)
        self.ae_test_metric.update_state(image, decoded_image)
        self.c_test_metric.update_state(one_hot, c_pred)

        return decoded_image
    
    def train(self, epoch, train_dataset, validation_dataset):
        train_dataset = train_dataset.shuffle(train_dataset.cardinality()//setting.shuffle_buffer_size_divider, reshuffle_each_iteration=True).batch(setting.batch_size)
        validation_dataset = validation_dataset.batch(setting.batch_size)

        for epoch_num in range(epoch):
            start = time.time()
            self.ae_train_metric.reset_state()
            self.ed_train_metric.reset_state()
            self.dd_train_metric.reset_state()
            self.c_train_metric.reset_state()

            self.ae_test_metric.reset_state()
            self.ed_test_metric.reset_state()
            self.dd_test_metric.reset_state()
            self.c_test_metric.reset_state()
            
            for batch in train_dataset:
                for _ in range(self.ed_iteration):
                    self.train_encoder_discriminator(batch)
                for _ in range(self.dd_iteration):
                    self.train_decoder_discriminator(batch)
                for _ in range(self.ae_iteration):
                    self.train_autocoder(batch)
                for _ in range(self.c_iteration):
                    self.train_classifier(batch)
                
            for batch in validation_dataset:
                image = batch["data"][0, :]
                decoded_image = self.test_step(batch)[0, :]

            if self.ae_iteration:
                self.e.save(setting.encoder_path)
                self.d.save(setting.decoder_path)
            if self.ed_iteration:
                self.ed.save(setting.encoder_discriminator_path)
            if self.dd_iteration:
                self.dd.save(setting.decoder_discriminator_path)
            if self.c_iteration:
                self.c.save(setting.classifier_path)

            cv2.imwrite(setting.sample_image, np.array((image+1)*127.5))
            cv2.imwrite(setting.sample_decoded_image, np.array((decoded_image+1)*127.5))

            cprint('Time for epoch {} is {} sec'.format(epoch_num + 1, time.time()-start), 'red')

            print("Train AutoEncoder Loss: " + str(self.ae_train_metric.result().numpy()))
            print("Test AutoEncoder Loss: " + str(self.ae_test_metric.result().numpy()))

            print("Train Encoder Discriminator Accuracy: " + str(self.ed_train_metric.result().numpy()))
            print("Test Encoder Discriminator Accuracy: " + str(self.ed_test_metric.result().numpy()))

            print("Train Decoder Discriminator Accuracy: " + str(self.dd_train_metric.result().numpy()))
            print("Test Decoder Discriminator Accuracy: " + str(self.dd_test_metric.result().numpy()))

            print("Train Classifier Accuracy: " + str(self.c_train_metric.result().numpy()))
            print("Test Classifier Accuracy: " + str(self.c_test_metric.result().numpy()))
