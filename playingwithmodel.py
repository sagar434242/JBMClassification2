import glob
from random import shuffle
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras import optimizers


def getLabel(filePaths):
	labels = []
	for img in filePaths:
		if 'Healthy' in img:
			labels.append(0)
		elif 'defects' in img:
			labels.append(1)

	dataZip = list(zip(filePaths, labels))
	shuffle(dataZip)
	filePaths, labels = zip(*dataZip)
	return filePaths, labels


def split_trainTest(imgsAll, labelAll, splitRatio = 0.25):
	dataZip = list(zip(imgsAll, labelAll))
	shuffle(dataZip)
	imgsAll, labelAll = zip(*dataZip)
	splitPoint = int(len(imgsAll)*splitRatio)

	trainImgs = imgsAll[:int(len(imgsAll) - splitPoint)]
	trainLabel = labelAll[:int(len(imgsAll) - splitPoint)]
	testImgs = imgsAll[int(len(imgsAll)-splitPoint):]
	testLabel = labelAll[int(len(imgsAll)-splitPoint):]

	return trainImgs, testImgs, trainLabel, testLabel

images, labels = getLabel(glob.glob('C:\\Users\\sbhure\\Desktop\\JBM\\JBMClassification2\\*\\*.jpg'))
# split the images into train and test sets.
X_train, X_test, y_train, y_test = split_trainTest(images, labels,  splitRatio=0.25)
display(print(f'SHape of training dataset {len(X_train)} and shape of validation set is {len(X_test)}'))

IMAGE_SIZE = (224, 224)

train_imgs = [img_to_array(load_img(img, target_size=IMAGE_SIZE)) for img in X_train]
train_imgs = np.array(train_imgs)
train_labels = list(y_train)
validation_imgs = [img_to_array(load_img(img, target_size=IMAGE_SIZE)) for img in X_test]
validation_imgs = np.array(validation_imgs)
validation_labels = list(y_test)
# Scale the images as Deep learning models tend to work good with smaller input

print('Train dataset shape:', train_imgs.shape, 
      '\tValidation dataset shape:', validation_imgs.shape)
train_imgs_scaled = train_imgs.astype('float32')
validation_imgs_scaled  = validation_imgs.astype('float32')
train_imgs_scaled /= 255
validation_imgs_scaled /= 255

batch_size = 32
num_classes = 2
epochs = 30
input_shape = (224, 224, 3)
img_width, img_height = 224, 224

# train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
#                                    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
#                                    horizontal_flip=True, fill_mode='nearest')

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_imgs_scaled, train_labels, batch_size=batch_size)
val_generator = val_datagen.flow(validation_imgs_scaled, validation_labels, batch_size=batch_size//2)


from keras.models import Model
import keras
from keras.applications.inception_v3 import InceptionV3

inception = InceptionV3(weights='imagenet', include_top=False, 
                             input_shape=(224, 224, 3))

output = inception.layers[-1].output
output = keras.layers.Flatten()(output)
incept_model = Model(inception.input, output)

incept_model.trainable = False
for layer in incept_model.layers:
    layer.trainable = False
    
# import pandas as pd
# pd.set_option('max_colwidth', -1)
# layers = [(layer, layer.name, layer.trainable) for layer in incept_model.layers]
# display(pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']))
    
def get_bottleneck_features(model, input_imgs):
    features = model.predict(input_imgs, verbose=0)
    return features
    
train_features_vgg = get_bottleneck_features(incept_model, train_imgs_scaled)
validation_features_vgg = get_bottleneck_features(incept_model, validation_imgs_scaled)

print('Train Bottleneck Features:', train_features_vgg.shape, 
      '\tValidation Bottleneck Features:', validation_features_vgg.shape)



from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
from keras import backend as K

input_shape = incept_model.output_shape[1]

model = Sequential()
model.add(InputLayer(input_shape=(input_shape,)))
model.add(Dense(524, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation=K.tanh))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
sgd = optimizers.RMSprop(lr=1e-5)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.summary()


history = model.fit(x=train_features_vgg, y=train_labels,
                    validation_data=(validation_features_vgg, validation_labels),
                    batch_size=batch_size,  
                    epochs=150,
                    verbose=1)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Model Performance', fontsize=18)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,151))
ax1.plot(epoch_list, history.history['acc'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_acc'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, 151, 10))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, 151, 10))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

model.save('JBM_Classification.h5')
f.savefig('JBM_Classification.png')

