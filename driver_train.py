import numpy as np
import pandas as pd
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import os
print(os.listdir("imgs/"))
from PIL import Image

class Configuration:
    def __init__(self):
        self.epochs = 10
        self.batch_size = 32
        self.maxwidth =0
        self.maxheight=0
        self.minwidth = 35000
        self.minheight = 35000
        self.imgcount=0
        self.img_width_adjust = 240
        self.img_height_adjust= 180
        #Kaggle
        self.data_dir = "imgs/train/"

config = Configuration()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('imgs/train/c0/img_4013.jpg')
imgplot = plt.imshow(img)
img.shape
plt.show()

def findPictureDims(path):
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg"):
                config.imgcount+=1
                filename = os.path.join(subdir, file)
                image = Image.open(filename)
                width, height = image.size
                if width < config.minwidth:
                    config.minwidth = width
                if height < config.minheight:
                    config.minheight = height
                if width > config.maxwidth:
                    config.maxwidth = width
                if height > config.maxheight:
                    config.maxheight = height
    return

def listDirectoryCounts(path):
    d = []
    for subdir, dirs, files in os.walk(path,topdown=False):
        filecount = len(files)
        dirname = subdir
        d.append((dirname,filecount))
    return d

def SplitCat(df):
    for index, row in df.iterrows():
        directory=row['Category'].split('/')
        if directory[3]!='':
            directory=directory[3]
            df.at[index,'Category']=directory
        else:
            df.drop(index, inplace=True)
    return

#dirCount=listDirectoryCounts(config.data_dir)
#categoryInfo = pd.DataFrame(dirCount, columns=['Category','Count'])
#SplitCat(categoryInfo)
#categoryInfo=categoryInfo.sort_values(by=['Category'])
#print(categoryInfo.to_string(index=False))

findPictureDims(config.data_dir)
print("Minimum Width:\t",config.minwidth, "\tMinimum Height:",config.minheight)
print("Maximum Width:\t",config.maxwidth, "\tMaximum Height:",config.maxheight, "\tImage Count:\t",config.imgcount)

def build_model():
    inputs = Input(shape=(config.img_width_adjust,config.img_height_adjust,3), name="input")
    
    #Convolution 1
    conv1 = Conv2D(128, kernel_size=(3,3), activation="relu", name="conv_1")(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool_1")(conv1)

    #Convolution 2
    conv2 = Conv2D(64, kernel_size=(3,3), activation="relu", name="conv_2")(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="pool_2")(conv2)
    
    #Convolution 3
    conv3 = Conv2D(32, kernel_size=(3,3), activation="relu", name="conv_3")(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), name="pool_3")(conv3)
    
    #Convolution 4
    conv4 = Conv2D(16, kernel_size=(3,3), activation="relu", name="conv_4")(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2), name="pool_4")(conv4)
    
    #Fully Connected Layer
    flatten = Flatten()(pool4)
    fc1 = Dense(1024, activation="relu", name="fc_1")(flatten)
    
    #output
    output=Dense(10, activation="softmax", name ="softmax")(fc1)
    
    # finalize and compile
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    return model

def setup_data(train_data_dir, val_data_dir, img_width=config.img_width_adjust, img_height=config.img_height_adjust, batch_size=config.batch_size):
    
    train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.8) # set validation split
    

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')
    
    validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')
        #Note uses training dataflow generator
    return train_generator, validation_generator

def fit_model(model, train_generator, val_generator, batch_size, epochs):
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        verbose=1)
    return model

def eval_model(model, val_generator, batch_size):
    scores = model.evaluate_generator(val_generator, steps=val_generator.samples // batch_size)
    print("Loss: " + str(scores[0]) + " Accuracy: " + str(scores[1]))


train_generator, val_generator = setup_data(config.data_dir, config.data_dir, batch_size=config.batch_size)

model = build_model()
print (model.summary())

model = fit_model(model, train_generator, val_generator,
                  batch_size=config.batch_size,
                  epochs=config.epochs)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

