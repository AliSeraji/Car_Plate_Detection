from LearnCharachters import LearnCharacterReading
import threading
import time

INIT_LR = 1e-4
EPOCHS = 40


def main():
    learner=LearnCharacterReading()
    learner.loadImages()
    catLabels,generatedImages=learner.convertToOneHotEncodingLables()
    model=learner.createModel(INIT_LR,INIT_LR/EPOCHS,True,catLabels.shape[1])
    learner.trainModel(model,generatedImages)
    print('done')
    

main()