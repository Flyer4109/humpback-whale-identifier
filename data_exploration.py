import cv2.cv2 as cv2
import pandas as pd
import matplotlib.pyplot as plt


# function to explore the data and print useful information
def explore():
    # reads training labels from csv file and inputs them in DataFrame
    labels = pd.read_csv('../data/train/train.csv')

    # helpful information of the data
    print('\nNumber of images:', len(labels['Image']))

    print('\nDescription:\n', labels.describe())

    print('\nMode id:', labels['Id'].mode()[0])

    # shows the 20 most common labels
    print('\nMost frequent ids:\n', labels['Id'].value_counts()[:20])

    # reads every image then gets their shape and creates a DataFrame of shapes
    image_shapes = pd.Series([cv2.imread('../data/train/' + im).shape for im in labels['Image']])

    # shows the 20 most common image shapes
    print('\nMost frequent image shapes:\n', image_shapes.value_counts()[:20])

    # Series that stores the counts of each label in order
    label_counts = labels['Id'].value_counts()

    # Series that stores the counts of each image resolution in order
    image_shape_counts = image_shapes.value_counts()

    # useful bar charts to help visualise data
    label_counts[1:20].plot(kind='bar', title='The 20 whales with the most images')
    plt.ylabel('Count')
    plt.show()

    label_counts[-20:].plot(kind='bar', title='Sample of whales with only one image')
    plt.ylabel('Count')
    plt.show()

    image_shape_counts[:20].plot(kind='bar', title='The 20 most common image resolutions')
    plt.ylabel('Count')
    plt.show()

    image_shape_counts[-20:].plot(kind='bar', title='Sample of image resolutions with only one image')
    plt.ylabel('Count')
    plt.show()


if __name__ == '__main__':
    explore()
