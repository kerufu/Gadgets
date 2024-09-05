Dataset: https://www.kaggle.com/datasets/dasarijayanth/pe-header-data

General idea: Build a model to encod the PE header of malware and legal ware, and classify the encoded features by classifier. 

Mere classification can be done via dense model, using conditional auto encoder to detecting unprecedented malware (inspired by the Kaspersky material, use auto encoder to cluster)

Reason for AE: to learn the proximity between before and after minor modifications

Reason for CAAE, not standard AE: 
1. GAN based data augmentation
2. symbolic header values (version, et.al) are conditional labels, and numerical header values are for embedding.
Information contained in conditional labels are passed intermediately to encoder and encoded feature through back propagation

Auto encoder to meansure the commonality in pe header (reconstruction loss)
high commonality, low reconstruction loss: legitimate classifying using AE_classifier
Low commonality, high reconstruction loss: classifying by the specially trained classifier (using abnormal data point in dataset, e.g. sample_loss >= mean_loss + std_loss * ratio)

1. Analyze the dataset
    a. select features to embedding (num of types, variance)
    b. Log dataset
    c. Normalizing
    d. balancing or using BinaryFocalCrossentropy
2. Embedding the data into low dimensional features vector
    a. PCA (for comparision)
    b. Auto encoder
        i. Sparsity from activation regulation
        ii. Randomness from discriminator https://arxiv.org/abs/1702.08423
3. Choose a fully connected classifier to classify the embedded features
    a. classifier trained together with AE
    b. classifier trained on anormal data points

Result:
![alt text](https://github.com/kerufu/Malware_Detection/blob/main/result.png)

Potential improvement:
1. More formal feature selection
    https://scikit-learn.org/stable/modules/feature_selection.html
2. Validate various models
3. Utilizing more knowledge on malware
4. Utilizing non embedding header information more effectively
