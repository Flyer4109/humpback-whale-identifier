import cv2.cv2 as cv2
import pandas as pd
import matplotlib.pyplot as plt


# function to explore the data and print useful information
def explore():
    # reads training labels from csv file and inputs them in DataFrame
    labels = pd.read_csv('../data/train/train.csv')

    # reads every image then gets their shape and creates a DataFrame of shapes
    image_shapes = pd.Series([cv2.imread('../data/train/' + im).shape for im in labels['Image']])

    # Series that stores the counts of each label in order
    label_counts = labels['Id'].value_counts()

    # Series that stores the counts of each image resolution in order
    image_shape_counts = image_shapes.value_counts()

    # description of labels includes: count, unique, top, and freq
    labels_desc = labels.describe()

    # variable to store if all data images are RGB
    is_rgb = True

    # iterate through image shapes
    for image_shape in image_shape_counts.index:
        # if the third number is not three then it is not RGB
        if image_shape[2] != 3:
            # all images are not RGB
            is_rgb = False

    # helpful information of the data
    print('-' * 35 + '\n~ Description of labels ~')

    print(labels_desc)

    print('-' * 35)
    print('Number of images:', len(labels['Image']))
    print('Number of labels:', labels_desc['Id']['unique'])
    print('Most common label:', labels_desc['Id']['top'], '(' + str(label_counts[0]) + ')')
    print(labels_desc['Id']['top'], 'takes up ' + str(round((label_counts[0]/len(labels['Image']) * 100), 2)) +
          '% of all images')

    # prints whether all images are RGB
    if is_rgb:
        print('All images have 3 channels (RGB)')
    else:
        print('Images vary in number of channels')

    print('-' * 35)

    # shows the 20 most common labels
    print('~ Most frequent ids ~')
    print(label_counts[:20])
    print('-' * 35)

    # shows the 20 most common image shapes
    print('~ Most frequent image shapes ~')
    print(image_shape_counts[:20])
    print('-' * 35)

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
