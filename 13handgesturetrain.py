from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#designing CNN
model = Sequential()# sequential-> starting model from the scratch
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(256,256,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) #designing more maxpooling layer for finding difference between images
model.add(Flatten())
model.add(Dense(units=150,activation='relu'))
model.add(Dense(units=6,activation='softmax'))#6 classes->0,1,2,3,4,5, output layer
#compiling the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#preprocessing the model
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=12.,
                                   width_shift_range=8.2,
                                   height_shift_range=8.2,
                                   zoom_range=8.15,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255) # pre-processing scaling settings

#training the model
training_set = train_datagen.flow_from_directory('Dataset2/train',
                                                 target_size=(256,256),
                                                 color_mode='grayscale',
                                                 batch_size=8,
                                                 classes = ['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                                 class_mode='categorical')
#validation model
val_set = val_datagen.flow_from_directory('Dataset2/val',
                                          target_size=(256,256),
                                          color_mode='grayscale',
                                          batch_size=8,
                                          classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                          class_mode='categorical')

# callback_list = [
#     EarlyStopping(monitor='val_loss',patience=10),
#     ModelCheckpoint(filepath="model.h6",monitor="val_loss",save_best_only=True,verbose=1)
#     ]
#fitting the model
model.fit_generator(training_set,
                    steps_per_epoch=10,# sample training imgs/batch_size
                    epochs=50,
                    validation_data=val_set,
                    validation_steps=8,
                    # callbacks=callback_list
                    )     # this lines will do training, also checks for callbacks, if it occurs it will terminate the code by saving model
