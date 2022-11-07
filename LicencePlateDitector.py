import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json
import glob



class LicencePLateDitector():
    def __init__(self):
        self.wpod_net_path ="wpod-net.json"
        self.wpod_net = self.load_model(self.wpod_net_path)
        self.licencePlates=self.findPlate(self.getImagePath())

    def load_model(self,path):
        try:
            path = splitext(path)[0]
            with open('%s.json' % path, 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json, custom_objects={})
            model.load_weights('%s.h5' % path)
            print("Loading model successfully...")
            return model
        except Exception as e:
            print(e)
            
    def preprocess_image(self,image_path,resize=False):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        if resize:
            img = cv2.resize(img, (224,224))
        return img

    def getImagePath(self):
        image_paths = glob.glob("car_plate_examples/*.jpg")
        print("Found %i images..."%(len(image_paths)))
        return image_paths

    def findPlateCoordinates(self,image_path):
        Dmax = 608
        Dmin = 288
        vehicle = self.preprocess_image(image_path)
        ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
        side = int(ratio * Dmin)
        bound_dim = min(side, Dmax)
        _ , LpImg, _, cor = detect_lp(self.wpod_net, vehicle, bound_dim, lp_threshold=0.5)
        return LpImg, cor

    def findPlate(self,image_paths):
        test_image = image_paths[0]
        licensePlates=[]
        for i in range(len(image_paths)):
            test_image = image_paths[i]
            #print("img name::",basename(test_image))
            LpImg,cor = self.findPlateCoordinates(test_image)
            #print("Detect %i plate(s) in"%len(LpImg),splitext(basename(test_image))[0])
            #print("Coordinate of plate(s) in image: \n", cor)
            licensePlates.append(LpImg)
            self.cordinate=cor
        return licensePlates

    def drawPlateBox(self,image, thickness=3):
        cor=self.cordinate
        pts=[]  
        x_coordinates=cor[0][0]
        y_coordinates=cor[0][1]
        # store the top-left, top-right, bottom-left, bottom-right 
        # of the plate license respectively
        for i in range(4):
            pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
        
        pts = np.array(pts,np.int32)
        pts = pts.reshape((-1,1,2))
        #vehicle_image = self.preprocess_image(image_path,True)
        cv2.polylines(image,[pts],True,(0,255,0),thickness)
        return image

    def showPlates(self,test_image):
        plt.figure(figsize=(8,8))
        plt.axis(False)
        plt.imshow(self.drawPlateBox(test_image,self.cordinate))
        plt.show()

"""
def runProgram():
    licenseDitector=LicencePLateDitector()
runProgram()    
"""