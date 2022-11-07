import cv2
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt


class CharacterSegmentaion():
    def __init__(self,licencePlates):
        self.licencePlates=licencePlates
        self.dilatedImg,self.lpImg,self.binaryImg=self.processLicencePlate()

    def processLicencePlate(self):
        LpImg=[]
        binaryImage=[]
        dilatedImg=[]
        i=0
        if(len(self.licencePlates)):
            for licensePlt in self.licencePlates:
                """Convert result to 8-bit"""
                img=np.array(licensePlt[0])
                LpImg.append(cv2.convertScaleAbs(img,alpha=(255.0))) 
                """convert to grayScale and blur the image to reduce noise"""
                grayScale = cv2.cvtColor(LpImg[i], cv2.COLOR_BGR2GRAY)
                blurImg = cv2.GaussianBlur(grayScale,(7,7),0)
                """inverse threshold binary Image"""
                binaryImage.append(cv2.threshold(blurImg,180,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]) 
                
                """dilate the image to increase the white area"""
                kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                dilatedImg.append(cv2.morphologyEx(binaryImage[i], cv2.MORPH_DILATE, kernel3)) 
                #show result
                #plt.figure(figsize=(10,5))
                #plt.imshow(dilatedImg[i],cmap='binary_r')
                #plt.show()
                i+=1
            return dilatedImg,LpImg,binaryImage

    def sortContours(self,contours):
        i=0
        boundingBoxes=[cv2.boundingRect(c) for c in contours]
        (contours,boundingBoxes)=zip(*sorted(zip(contours,boundingBoxes),key=lambda b: b[1][i], reverse=False))
        return contours

    def findCharacters(self,binaryImage,LpImg):
        allLiceneces=[]
        for i in range(len(binaryImage)):
            contours,_=cv2.findContours(binaryImage[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            characterImg=[]
            charWidth,charHeight=45,90
            for char in self.sortContours(contours):
                (x, y, w, h) = cv2.boundingRect(char)
                ratio=h/w
                if 0.3<=ratio<=5:
                    if h/LpImg[i].shape[0]>=0.5:
                        """draw box around the characters"""
                        cv2.rectangle(LpImg[i], (x, y), (x + w, y + h), (0, 255,0), 2)
                        """crop charachters from the plate image(the dilated image)"""
                        currentChar= self.dilatedImg[i][y:y+h,x:x+w] 
                        currentChar=cv2.resize(currentChar,dsize=(charWidth,charHeight))
                        _,currentChar=cv2.threshold(currentChar,220,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        characterImg.append(currentChar)
            fig = plt.figure(figsize=(10,4))
            plt.imshow(LpImg[i])
            plt.show()
            print("Detect {} letters...".format(len(characterImg)))
            allLiceneces.append(characterImg) 
            #self.showFoundCharacters(characterImg)
        return allLiceneces            

    def showFoundCharacters(self,charachterImg):
        fig = plt.figure(figsize=(14,4))
        grid = gridspec.GridSpec(ncols=len(charachterImg),nrows=1,figure=fig)
        for i in range(len(charachterImg)):
            fig.add_subplot(grid[i])
            plt.axis(False)
            plt.imshow(charachterImg[i],cmap="gray")
            plt.show()





