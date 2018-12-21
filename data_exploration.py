import cv2.cv2 as cv2
import pandas as pd


# function to explore the data and print useful information
def explore():
    # reads training labels from csv file and inputs them in DataFrame
    labels = pd.read_csv('../data/train/train.csv')

    # helpful information of the data
    print('\nNumber of images:', len(labels['Image']))

    print('\nDescription:\n', labels.describe())

    print('\nMode id:', labels['Id'].mode()[0])

    # shows the 16 most common labels
    print('\nMost frequent ids:\n', labels['Id'].value_counts()[:16])

    # reads every image then gets their shape and creates a DataFrame of shapes
    image_shapes = pd.Series([cv2.imread('../data/train/' + im).shape for im in labels['Image']])
    
    # shows the 16 most common image shapes
    print('\nMost frequent image shapes:\n', image_shapes.value_counts()[:16])


if __name__ == '__main__':
    explore()
