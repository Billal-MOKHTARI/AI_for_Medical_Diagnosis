# Data Exploration & Image Pre-processing

## 1. Exploration
### 1.1 Data Types and Null Values Check
### 1.2 Unique IDs Check
### 1.3 Data Labels
### 1.4 Data Visualization
### 1.5 Investigating a Single Image
### 1.6 Investigating Pixel Value Distribution
## 2. Image Preprocessing in Keras
### 2.1 Standardization


## 1. Import Packages and Functions
## 2. Load the Datasets
### 2.1 Loading the Data
### 2.2 Preventing Data Leakage
### 2.3 Preparing Images
By using **tensorflow.keras.preprocessing.image.ImageDataGenerator** object :
- Standardization (samplewise_center, smplewise_std_normalization)
- Transform to 3D channel images if the CNN architecture requires that
- Shuffle the images
- Set the image size
- Small batch sizes (8 for example)

#### Notice
We should build different generators for training and test, because we don't want that our training set to have any information about the test set. 
- We create train generator which uses his own batch statistics.
- We create validation and test set by using the statistics **computed from the training set**:
    1. We create augmented data on the training set with sample size (different of batch size).
    2. We fit a generator (with standardization parameters) on one single image of previous augmented data.
    3. We apply the generator to augment the validation and the test set.


## Some methods :

```python
"""Create an image generator"""
generator = ImageDataGenerator()

"""Takes data & label arrays, generates batches of augmented data."""
generator.flow()

"""Takes the dataframe and the path to a directory + generates batches.
The generated batches contain augmented/normalized data."""
generator.flow_from_dataframe()

"""Takes the path to a directory & generates batches of augmented data."""
generator.flow_from_directory()

"""Fits the data generator to some sample data.
This computes the internal data stats related to the data-dependent transformations, based on an array of sample data"""
generator.fit(data_sample)

"""Assume we have generated augmented data, the method next allows us to get the next batch"""
augmented_data.next()

"""Get the index-th example of the generated data"""
augmented_data.__getitem(index)

"""Get the labels of the generated data"""
augmented_data.labels
```

## 3. Model Development
### 3.1 Addressing Class Imbalance
### 3.2 DenseNet121
We add the following layers :
1. A `GlobalAveragePooling2D` layer to get the average of the last convolution layers from DenseNet121.
2. A `Dense` layer with `sigmoid` activation to get the prediction logits for each of our classes.
## 4. Training
```python
model.fit_generator(train_generator, validation_data, steps_per_epoch, validation_steps, epochs)
```

### 4.1 Training on the Larger Dataset
Few useful keras callbacks :
1. You can use `ModelCheckpoint` callback to monitor your model's `val_loss` metric and keep a snapshot of your model at the point. 
2. You can use the `TensorBoard` to use the Tensorflow Tensorboard utility to monitor your runs in real-time. 
3. You can use the `ReduceLROnPlateau` to slowly decay the learning rate for your model as it stops getting better on a metric such as `val_loss` to fine-tune the model in the final steps of training.
4. You can use the `EarlyStopping` callback to stop the training job when your model stops getting better in it's validation loss. You can set a `patience` value which is the number of epochs the model does not improve after which the training is terminated. This callback can also conveniently restore the weights for the best metric at the end of training to your model.

You can read about these callbacks and other useful Keras callbacks [here](https://keras.io/callbacks/).

## 5. Prediction and Evaluation
```python
predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))
```

### 5.1 ROC Curve and AUROC
```python
auc_rocs = util.get_roc_curve(labels, predicted_vals, test_generator)
```

Interresting papers :
- [CheXNet](https://arxiv.org/abs/1711.05225)
- [CheXpert](https://arxiv.org/pdf/1901.07031.pdf)
- [ChexNeXt](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002686)

### 5.2 Visualizing Learning with GradCAM 
One of the challenges of using deep learning in medicine is that the complex architecture used for neural networks makes them much harder to interpret compared to traditional machine learning models (e.g. linear models). 

One of the most common approaches aimed at increasing the interpretability of models for computer vision tasks is to use Class Activation Maps (CAM). 
- Class activation maps are useful for understanding where the model is "looking" when classifying an image. 

In this section we will use a [GradCAM's](https://arxiv.org/abs/1610.02391) technique to produce a heatmap highlighting the important regions in the image for predicting the pathological condition. 
- This is done by extracting the gradients of each predicted class, flowing into our model's final convolutional layer. Look at the `util.compute_gradcam` which has been provided for you in `util.py` to see how this is done with the Keras framework. 

It is worth mentioning that GradCAM does not provide a full explanation of the reasoning for each classification probability. 
- However, it is still a useful tool for "debugging" our model and augmenting our prediction so that an expert could validate that a prediction is indeed due to the model focusing on the right regions of the image.

```python
df = pd.read_csv("data/nih/train-small.csv")
IMAGE_DIR = "data/nih/images-small/"

# only show the labels with top 4 AUC
labels_to_show = np.take(labels, np.argsort(auc_rocs)[::-1])[:4]
```
