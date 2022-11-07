from sklearn import preprocessing
import numpy as np
import cv2 as cv
from tensorflow.keras.models import model_from_json,load_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



class CharacterReader():
    def __init__(self):
        netFile=open('Network_character_recognizer.json','r')
        print("loading neural network...")
        createdModel=netFile.read()
        print("neural network loaded!")
        print("loading models...")
        self.model=model_from_json(createdModel)
        print("models are loaded!")
        netFile.close()
        print("loading weights...")
        self.model.load_weights("licenceplate_character_classification_weight.h5")
        print("weights are loaded!")
        self.labels=preprocessing.LabelEncoder()
        print("loading character classes...")
        self.labels.classes_=np.load("licenceplate_character_classification.npy")
        print("character classes are loaded successfully!")
        
    def predictCharacter(self,charIm):
        charIm=cv.resize(charIm,(105,105))
        charIm=np.stack((charIm,)*3,axis=-1)
        prediction=self.labels.inverse_transform([np.argmax(self.model.predict(charIm[np.newaxis,:]))])
        return prediction
    
    def performPrediction(self,cropedCharacters):
        #fig = plt.figure(figsize=(15,3))
        cols = len(cropedCharacters)
        #if cols==0:
        #    return
        #grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)
        finalPrediction=""
        for i,char in enumerate(cropedCharacters):
            title=np.array2string(self.predictCharacter(char))
            plt.title('{}'.format(title.strip("'[]"),fontsize=20))
            finalPrediction+=title.strip("'[]")
            #plt.axis(False)
            #plt.imshow(char,cmap='gray')
            #plt.show()
        print("final result is::"+finalPrediction)
        
        
        
          