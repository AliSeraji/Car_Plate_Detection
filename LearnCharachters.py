import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array,load_img,ImageDataGenerator
from keras.utils import to_categorical
from keras.layers.core import Dense,Dropout
from keras.layers import AveragePooling2D,Flatten
from keras.applications import MobileNetV2
from keras import Model,Input
from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

class LearnCharacterReading():
    def __init__(self):
        self.datasetPaths= glob.glob("dataset_characters/**/*.jpg")
        self.inputData=[]
        self.lables=[]
        self.BATCH_SIZE=64
        self.EPOCHS = 40
        self.trainXYtestXY=[]

    def loadImages(self):
        print("loading models...")
        for path in self.datasetPaths:
            label=path.split(os.path.sep)[-2]
            image=load_img(path,target_size=(105,105))
            image=img_to_array(image)
            self.inputData.append(image)
            self.lables.append(label)
        self.inputData=np.array(self.inputData,dtype="float16")
        self.lables=np.array(self.lables)
        print("Find {:d} images with {:d} classes".format(len(self.inputData),len(set(self.lables))))

    def convertToOneHotEncodingLables(self):
        """here we just perform a one-hot encoding to our data for training"""
        print("converting to one-hot encoding...")
        lbs=preprocessing.LabelEncoder()
        lbs.fit(self.lables)
        self.lables=lbs.transform(self.lables)
        catLabels=to_categorical(self.lables)
        np.save('licenceplate_character_classification.npy',lbs.classes_)
        (trainX, testX, trainY, testY) =train_test_split(self.inputData,catLabels, test_size=0.10,
                                                            stratify=catLabels, random_state=42)
        self.trainXYtestXY.append(trainX)
        self.trainXYtestXY.append(testX)
        self.trainXYtestXY.append(trainY)
        self.trainXYtestXY.append(testY)
        """generate data for augmentation to acheive better training"""
        generatedImages =ImageDataGenerator(rotation_range=5,
                              width_shift_range=0.05,
                              height_shift_range=0.05,
                              shear_range=0.05,
                              zoom_range=0.05,
                              fill_mode="nearest"
                              )
        print("one-hot encoding is done!")
        return catLabels,generatedImages                                                    

    def createModel(self,lr,decay, training,outputShape):
        """create a basic model with MobileNet and giving pretrained weigths"""
        print("creating the model...")
        baseModel = MobileNetV2(weights="imagenet", 
                            include_top=False,
                            input_tensor=Input(shape=(105, 105, 3)))

        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(256, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(outputShape, activation="softmax")(headModel)
        
        model = Model(inputs=baseModel.input, outputs=headModel)
        
        if training:
            # define trainable lalyer
            for layer in baseModel.layers:
                layer.trainable = True
                
            # compile model
            optimizer = Adam(lr=lr, decay = decay)
            model.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=["accuracy"])    
        print("the model is created!")    
        return model
    
    def trainModel(self,model,generatedImages):
        print("training")
        checkpoint=[
                EarlyStopping(monitor='val_loss', patience=20, verbose=0),
                ModelCheckpoint(filepath="licenceplate_character_classification_weight.h5", verbose=1, save_weights_only=True)          
            ]
        result = model.fit(generatedImages.flow(self.trainXYtestXY[0], self.trainXYtestXY[2], batch_size=self.BATCH_SIZE), 
                   steps_per_epoch=len(self.trainXYtestXY[0]) // self.BATCH_SIZE, 
                   validation_data=(self.trainXYtestXY[1], self.trainXYtestXY[3]), 
                   validation_steps=len(self.trainXYtestXY[1]) // self.BATCH_SIZE, 
                   epochs=self.EPOCHS, callbacks=checkpoint)
        jsonModel = model.to_json()
        with open("Network_character_recognizer.json", "w") as json_file:
            json_file.write(jsonModel)
        print("done training and saving network!")
        fig = plt.figure(figsize=(14,5))
        grid=gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
        fig.add_subplot(grid[0])
        plt.plot(result.history['accuracy'], label='training accuracy')
        plt.plot(result.history['val_accuracy'], label='val accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        fig.add_subplot(grid[1])
        plt.plot(result.history['loss'], label='training loss')
        plt.plot(result.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()


        