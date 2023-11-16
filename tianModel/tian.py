import os
from math import *

here = os.path.dirname(os.path.abspath(__file__))

handDirectory = here + '\\tianDatasets\\handwrittenData'
typedDirectory = here + '\\tianDatasets\\typedData'

testImagesDirectory = handDirectory + '\\t10k-images.idx3-ubyte'
testLabelsDirectory = handDirectory + '\\t10k-labels.idx1-ubyte'
trainImagesDirectory = handDirectory + '\\train-images.idx3-ubyte'
trainLabelsDirectory = handDirectory + '\\train-labels.idx1-ubyte'

def bytesToInts(byte_data):
    return int.from_bytes(byte_data, 'big')

def importImages(filename, maxImages=None):
    images = []
    with open(filename, 'rb') as file:
        _ = file.read(4)
        numImages = bytesToInts(file.read(4))
        if maxImages:
            numImages = maxImages
        numRows = bytesToInts(file.read(4))
        numColumns = bytesToInts(file.read(4))
        for imgIndex in (range(numImages)):
            image = []
            for rowIndex in (range(numRows)):
                row = []
                for columnIndex in (range(numColumns)):
                    pixel = file.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images

def importLabels(filename, maxLabels=None):
    labels = []
    with open(filename, 'rb') as file:
        _ = file.read(4)
        numLabels = bytesToInts(file.read(4))
        if maxLabels:
            numLabels = maxLabels
        for labelIndex in (range(numLabels)):
            label = file.read(1)
            labels.append(label)
    return labels

def flattenList(list):
    return [pixel for sublist in list for pixel in sublist]

def extract_features(X):
    return [flattenList(sample) for sample in X]

def dist(x, y):
    return sum (
        [
            (bytesToInts(x_i) - bytesToInts(y_i)) ** 2
            for x_i, y_i in zip(x,y)

        ]
    ) ** 0.5


def getTrainingDistance(X_train, testSamples):
    return (dist(train_sample, testSamples) for train_sample in X_train)

def getMostFrequent(list):
    return max(list, key=list.count)


def knn(X_train, y_train, X_test, k=3):
    y_pred = []

    for testidx, sample in enumerate(X_test):
        trainingDistances = getTrainingDistance(X_train, sample)
        sortedDistance = [
            pair[0]
            for pair in sorted(enumerate(trainingDistances), key=lambda x: x[1])
        ]
        candidates = [
            bytesToInts(y_train[idx])
            for idx in sortedDistance[:k]
        ]
        top_candidate = getMostFrequent(candidates)
        y_pred.append(top_candidate)
    return y_pred


def main():
    X_train = importImages(trainImagesDirectory, 1000)
    y_train = importLabels(trainLabelsDirectory, 1000)
    X_test = importImages(testImagesDirectory, 5)
    y_test = importLabels(testLabelsDirectory, 5)

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    y_pred = knn(X_train, y_train, X_test ,3)

    print(y_pred)
    accuracy = sum([
        int(y_pred_i == bytesToInts(y_test_i)) 
        for y_pred_i, y_test_i 
        in zip(y_pred, y_test)
                    ]) / len(y_test)
    print(f'Predicted labels: {y_pred}')
    print(f'Accuracy: {accuracy}%')
if __name__ == '__main__':
    main()