import argparse
from lab1_addnoise import addNoise
from lab1_autoencoderTest import testAutoencoder
from lab1_linearinterpolation import linearInterpolator


# prompts user for input and ensures it falls in valid range
def get_valid_index():
    while True:
        try:
            index = int(input("Please enter an integer between 0 and 59999: "))
            if 0 <= index <= 59999:
                return index
            else:
                raise ValueError("Input must be between 0 and 59999")
        except ValueError as e:
            print(f"Error: {e}")


def get_valid_index2(index1):
    while True:
        try:
            index2 = int(input("Please enter a new integer between 0 and 59999: "))
            if index2 == index1:
                raise ValueError("New integer cannot be same as old integer")
            elif 0 <= index2 <= 59999:
                return index2
            else:
                raise ValueError("Input must be between 0 and 59999")
        except ValueError as e:
            print(f"Error: {e}")


# apply argument parser to enable running script via required command
parser = argparse.ArgumentParser(description='Calling Lab 1 with desired model file')
parser.add_argument('-l', '--modelPath', type=str, default="MLP.8.pth", help='name of model file')
args = parser.parse_args()
modelPath = args.modelPath

# get first index
index = get_valid_index()

# demonstrate step 4 functionality
autoencoderObject = testAutoencoder(modelPath, index)
autoencoderObject.runAutoencoder(modelPath, index)

# demonstrate step 5 functionality
addNoiseObject = addNoise(modelPath, index)
addNoiseObject.applyNoise(modelPath, index)

# get second index for linear interpolation and demonstrate step 6 functionality
index2 = get_valid_index2(index)
linearInterpolationObject = linearInterpolator(modelPath, 8, index, index2)
linearInterpolationObject.interpolator(modelPath, 8, index, index2)
