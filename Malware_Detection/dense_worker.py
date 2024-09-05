import tensorflow as tf

import setting


class dense_classifier(tf.keras.Model):
    def __init__(self):
        super(dense_classifier, self).__init__()
        self.model = [
            tf.keras.layers.Dense(32, activation='selu',
                                  kernel_regularizer=tf.keras.regularizers.L1L2()
                                  ),
            tf.keras.layers.Dense(1,
                                  kernel_regularizer=tf.keras.regularizers.L1L2()
                                  )
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class dense_worker():

    def __init__(self, dsw) -> None:
        self.c = dense_classifier()
        self.dsw = dsw

        try:
            self.c.load_weights(setting.dense_classifier_path)
            print("Saved dense classifier weight loaded")
        except:
            print("Saved dense classifier weight not found")
        self.c.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
                       loss=tf.keras.losses.BinaryFocalCrossentropy(
                           from_logits=True),
                       metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0),
            tf.keras.metrics.FalsePositives(thresholds=0),
            tf.keras.metrics.Precision(thresholds=0),
            tf.keras.metrics.Recall(thresholds=0)
        ]
        )

    def train(self, epoch, indexes):
        dataset_size = len(indexes)
        embedding_data = self.dsw.get_embedding_data()[indexes]
        non_embedding_data = self.dsw.get_non_embedding_data()[indexes]

        feature = tf.concat([embedding_data, non_embedding_data[:, :-1]], 1)
        legitimate_lable = non_embedding_data[:, -1].astype(int)

        train_dataset_size = int(
            dataset_size*setting.dense_train_dataset_ratio)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (feature[:train_dataset_size], legitimate_lable[:train_dataset_size]))
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (feature[train_dataset_size:], legitimate_lable[train_dataset_size:]))

        train_dataset = train_dataset.shuffle(
            setting.dataset_size, reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(setting.batch_size)
        test_dataset = test_dataset.shuffle(
            setting.dataset_size, reshuffle_each_iteration=True)
        test_dataset = test_dataset.batch(setting.batch_size)

        self.c.fit(train_dataset, epochs=epoch, validation_data=test_dataset)
        self.c.save(setting.dense_classifier_path)

    def test(self, indexes):
        embedding_data = self.dsw.get_embedding_data()[indexes]
        non_embedding_data = self.dsw.get_non_embedding_data()[indexes]
        legitimate_lable = non_embedding_data[:, -1].astype(int)
        non_embedding_data = non_embedding_data[:, :-1]
        feature = tf.concat([embedding_data, non_embedding_data], 1)

        self.c.evaluate(feature, legitimate_lable)
