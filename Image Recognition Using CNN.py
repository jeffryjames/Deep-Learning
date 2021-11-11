from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten    
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape = (64,64,3,),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(optimizer="adam",loss= "binary_crossentropy",metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale= 1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
                
val_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory("D:/Python/DL/Day 12/Code/Dataset/train", 
                                                    target_size=(64,64),
                                                    batch_size=8,
                                                    class_mode="binary")

val_set = train_datagen.flow_from_directory("D:/Python/DL/Day 12/Code/Dataset/val", 
                                                    target_size=(64,64),
                                                    batch_size=8,
                                                    class_mode="binary")

model.fit(training_set,
                    steps_per_epoch=10,
                    epochs=50,
                    validation_data= val_set,
                    validation_steps=2)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")

#TESTING

from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

def classify(img_file):
    img_name = img_file
    test_image = image.load_img(img_name, target_size = (64, 64))

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    if result[0][0] == 1:
        prediction = 'Thanos'
    else:
        prediction = 'Joker'
    print(prediction,img_name)


import os
path = 'D:/Python/DL/Day 12/Code/Dataset/test'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
     if '.jpeg' in file:
       files.append(os.path.join(r, file))

for f in files:
   classify(f)
   print('\n')