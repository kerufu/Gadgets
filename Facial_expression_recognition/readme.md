Explore application of GAN in classification, experiment in CAAE and WGAN 

Reason:
1. GAN based data augmentation, inject randomness to middle layer feature directly
2. Inspired by knowledge distillation, replacing kernal regularization and dropout (redundant neurals)

Test procedure:
1. Disable discriminator and scale dowm model until overfitting and test acc significantly degenerates
2. Gradually add weight and iteration on discriminators to mitigate overfitting

Test result:

CAAE works better than WGAN, possibly because:
1. Wasserstein loss is compatible with cross entropy from logit only if batch normalized, but it would impair the effectiveness of feature discriminator
2. Didn't find paper/blog about smoothing label with Wasserstein loss 
2. More information loss without auto encoder architecture

All models are trained for 50 epoches.

For full size model: 65% acc

For scaled_down model: Keep only 2 small conv2d layer in encoder. The model and settings are in branch "scale_down_model". 

Without support modules: 50.2% Acc, significantly overfitting very soon and degenerate to around 41% Acc.

With support modules: 52.6% Acc, not significantly overfitting yet.

Model settings:

the encoded features are forced to proximate uniform distribution (-1, 1), the conditional label (expression class) are in one hot coding, e.g. 1, -1 , -1, ...

data are normalized into range (-1, 1), train on the whole train dataset and use validation dataset as test dataset

Setup guideline

dataset: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset?resource=download

put the dataset in ./dataset

Docker usage:

To build image: docker build . -t tensorflow_opencv

To create container and start: docker run --name fer -e PYTHONUNBUFFERED=1 -it --gpus all tensorflow_opencv

To copy files from/to container: check the .sh scripts

To set up GPU on windows docker:
1. upgrade nvidia driver
2. wsl --install in administrator mode
3. enable wsl2 and ubuntu in docker
4. follow the instruction: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#install-guide

Hints:
1. Shuffle and batch tf.data.Dataset carefully
2. "selu" ≈ "BatchNormalization" + "leaky_relu", don't use "selu" + "BatchNormalization"
3. Gradient penalty or label smoothing when discriminator is unstable
4. Maxpooling for classification cnn, stride for generation cnn
5. Allocate more than one dense layer in classifier, otherwise it doesn't have sufficient capability to learn the distribution of the augmented feature
6. KL divergence may replace feature discriminator, not fully verified
7. I used Flatten() bacause this model is not for image generation, and GANs are simply supporting the training of classifier. Reshape conditional label instead of flatten image if model is for image generation

More reference:

https://medium.com/@KNuggies/tensorflow-with-gpu-on-windows-with-wsl-and-docker-75fb2edd571f

https://github.com/soumith/ganhacks

https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/

https://davidleonfdez.github.io/gan/2022/05/17/gan-convergence-stability.html
