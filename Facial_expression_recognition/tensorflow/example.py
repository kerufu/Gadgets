import cv2
import tensorflow as tf
import numpy as np
from os.path import dirname, join
import base64
from tensorflow.keras.regularizers import l2
import java

tf_model_path = join(dirname(__file__), "face_expression_model")

# HOG
std_width = 50
winSize = (std_width, std_width)
blockSize = (20, 20)
blockStride = (10, 10)
cellSize = (5, 5)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64

class CNN_model(tf.keras.layers.Layer):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.model = [
            tf.keras.layers.Reshape((std_width, std_width, 1)),
            tf.keras.layers.Conv2D(8, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu')
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x

class HOG_model(tf.keras.layers.Layer):

    def __init__(self):
        super(HOG_model, self).__init__()
        self.model = [
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(4, activation='relu')
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x

class Landmark_model(tf.keras.layers.Layer):

    def __init__(self):
        super(Landmark_model, self).__init__()
        self.model = [
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(8, activation='relu')
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x

class faceExpressionDetectorModel(tf.keras.Model):
    def __init__(self):
        super(faceExpressionDetectorModel, self).__init__()
        self.CNN_module = CNN_model()
        self.HOG_module = HOG_model()
        self.Landmark_module = Landmark_model()
        self.neutral_output_module = [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.00001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001))
        ]
        self.smile_output_module = [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.00001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001))
        ]

    def call(self, inputs):
        face_image, HOG, landmark = inputs
        feature_CNN = self.CNN_module(face_image)
        feature_HOG = self.HOG_module(HOG)

        feature_landmark = self.Landmark_module(landmark)
        neutral_feature = tf.keras.layers.concatenate([feature_CNN, feature_HOG, feature_landmark])
        smile_feature = tf.keras.layers.concatenate([feature_CNN, feature_HOG, feature_landmark])

        for layer in self.neutral_output_module:
            neutral_feature = layer(neutral_feature)

        for layer in self.smile_output_module:
            smile_feature = layer(smile_feature)

        return [neutral_feature, smile_feature]

class FaceExpressionDetector:
    initialized = False
    def initialize(self):
        self.tf_model = faceExpressionDetectorModel()
        self.tf_model.load_weights(tf_model_path).expect_partial()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
        self.get_hog = cv2.HOGDescriptor(
                winSize, blockSize, blockStride, cellSize, nbins,
                derivAperture, winSigma, histogramNormType,
                L2HysThreshold, gammaCorrection, nlevels)
        self.initialized = True

    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def run(self, encodedImage, landmark):
        if not self.initialized:
            self.initialize()

        print("already initialized")
        decoded_data = base64.b64decode(encodedImage)
        np_data = np.fromstring(decoded_data,np.uint8)
        image = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, dsize=(std_width, std_width), interpolation=cv2.INTER_CUBIC)
        gray = self.clahe.apply(gray)

        landmark = np.array(landmark)

        hog_features = self.get_hog.compute(gray)

        inputs = [np.array([gray]).astype(np.float32),
                  np.array([hog_features.flatten()]).astype(np.float32),
                  np.array([landmark.flatten()]).astype(np.float32)]

        result = self.tf_model.predict_on_batch(inputs)
        res = [java.jfloat(float(result[0].numpy()[0][0])), java.jfloat(float(result[1].numpy()[0][0]))]
        return res

def getFaceExpressionDetector():
    return FaceExpressionDetector()
