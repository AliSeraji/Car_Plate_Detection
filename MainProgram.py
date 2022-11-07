from LicencePlateDitector import LicencePLateDitector
from CharacterSegmention import CharacterSegmentaion
from CharacterReader import CharacterReader

def main():
    reader=CharacterReader()
    licenseDitector=LicencePLateDitector()
    charSegment=CharacterSegmentaion(licenseDitector.licencePlates)
    charImgs=charSegment.findCharacters(charSegment.binaryImg,charSegment.lpImg)
    for licence in charImgs:
        reader.performPrediction(licence)
    
    
main()