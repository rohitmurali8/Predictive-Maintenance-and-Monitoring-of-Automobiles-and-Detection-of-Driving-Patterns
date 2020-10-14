from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from keras.models import model_from_json
import cv2
from imutils import build_montages
from imutils import paths
import random
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import SGD
import os

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

imagePaths = list(paths.list_images("C:/Data/Software/imgs/test"))
#random.shuffle(imagePaths)
imagePaths = imagePaths[75:76]
 
# initialize our list of results
results = []
'''img = cv2.imread("img_33.jpg")
cv2.imshow("Image", img)
img = np.array(img)
img = np.reshape(img,(7,240,180,3))'''
for p in imagePaths:
	# load our original input image
    orig = cv2.imread(p)
 
	# pre-process our image by converting it from BGR to RGB channel
	# ordering (since our Keras mdoel was trained on RGB ordering),
	# resize it to 64x64 pixels, and then scale the pixel intensities
	# to the range [0, 1]
    image = cv2.resize(orig, (180, 240))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
 
	# make predictions on the input image
    pred = loaded_model.predict(image)
    pred = pred.argmax(axis=1)[0]
    print(pred)
    if pred == 0:
        print("Driver not distracted")
    if pred == 1 or pred == 3:
        print("Driver using phone")
    if pred == 2:
        print("Driver talking on phone")
    if pred == 4:
        print("Driver talking on phone")
    if pred == 5:
        print("Driver fiddling with dashboard")
    if pred == 6:
        print("Driver drinking juice")
    if pred == 7 or pred == 9:
        print("Driver looking away")
    if pred == 8:
        print("Driver fidgeting specks/hair/face")
    results.append(orig)

montage = build_montages(results, (128, 128), (4, 4))[0]
'''def test_gen(batch_size = 100):
    path = "C:\\Data\\Software\\imgs\\test"
    a = 0
    list1 = []
    for image in os.listdir(path):
        if a < batch_size:
            img = load_img(path + '/' + image, target_size = (180,240))
            img = img_to_array(img)
            list1.append(img)
            a = a + 1
        else:
            yield np.array(list1)
            list1 = []
        
test_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.8) # set validation split
    

test_generator = test_datagen.flow_from_directory(
        "C:\\Data\\Software\\imgs\\test",
        target_size=(240, 180),
        batch_size=64,
        class_mode='categorical',
        subset='training')
generator = datagen.flow_from_directory(
        'data/test',
        target_size=(150, 150),
        batch_size=16,
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels

probabilities = model.predict_generator(generator, 2000)

loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
test_accuracy = loaded_model.evaluate_generator(test_gen(), steps = 797)
print(test_accuracy)'''
# show the output montage
cv2.imshow("Results", montage)
cv2.waitKey(0)
