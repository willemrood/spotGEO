import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

import itertools

import random

from skimage import measure
# %% Define functions
def read_image(path):
    root = 'D:/Python/Projects/spotGEO/data/'
    fullpath = root+path
    return plt.imread(fullpath)

def read_annotation_file(path):
    root = 'D:/Python/Projects/spotGEO/data/'
    fullpath = root+path
    with open(fullpath) as annotation_file:
        annotation_list = json.load(annotation_file)
    # Transform list of annotations into dictionary
    annotation_dict = {}
    for annotation in annotation_list:
        sequence_id = annotation['sequence_id']
        if sequence_id not in annotation_dict:
            annotation_dict[sequence_id] = {}
        annotation_dict[sequence_id][annotation['frame']] = annotation['object_coords']
    return annotation_dict


random.seed(0)

def random_different_coordinates(coords, size_x, size_y, pad):
    """ Returns a random set of coordinates that is different from the provided coordinates, 
    within the specified bounds.
    The pad parameter avoids coordinates near the bounds."""
    good = False
    while not good:
        good = True
        c1 = random.randint(pad + 1, size_x - (pad + 1))
        c2 = random.randint(pad + 1, size_y -( pad + 1))
        for c in coords:
            if c1 == c[0] and c2 == c[1]:
                good = False
                break
    return (c1,c2)

def extract_neighborhood(x, y, arr, radius):
    """ Returns a 1-d array of the values within a radius of the x,y coordinates given """
    return arr[(x - radius) : (x + radius + 1), (y - radius) : (y + radius + 1)].ravel()

def check_coordinate_validity(x, y, size_x, size_y, pad):
    """ Check if a coordinate is not too close to the image edge """
    return x >= pad and y >= pad and x + pad < size_x and y + pad < size_y

def generate_labeled_data(image_path, annotation, nb_false, radius):
    """ For one frame and one annotation array, returns a list of labels 
    (1 for true object and 0 for false) and the corresponding features as an array.
    nb_false controls the number of false samples
    radius defines the size of the sliding window (e.g. radius of 1 gives a 3x3 window)"""
    features,labels = [],[]
    im_array = read_image(image_path)
    # True samples
    for obj in annotation:
        obj = [int(x + .5) for x in obj] #Project the floating coordinate values onto integer pixel coordinates.
        # For some reason the order of coordinates is inverted in the annotation files
        if check_coordinate_validity(obj[1],obj[0],im_array.shape[0],im_array.shape[1],radius):
            features.append(extract_neighborhood(obj[1],obj[0],im_array,radius))
            labels.append(1)
    # False samples
    for i in range(nb_false):
        c = random_different_coordinates(annotation,im_array.shape[1],im_array.shape[0],radius)
        features.append(extract_neighborhood(c[1],c[0],im_array,radius))
        labels.append(0)
    return np.array(labels),np.stack(features,axis=1)

def generate_labeled_set(annotation_array, path, sequence_id_list, radius, nb_false):
    # Generate labeled data for a list of sequences in a given path
   
    labels,features = [],[]
    for seq_id in sequence_id_list:
        for frame_id in range(1,6):
            d = generate_labeled_data(f"{path}{seq_id}/{frame_id}.png",
                                    annotation_array[seq_id][frame_id],
                                    nb_false,
                                    radius)
            labels.append(d[0])
            features.append(d[1])
    return np.concatenate(labels,axis=0), np.transpose(np.concatenate(features,axis=1))

def classify_image(im, model, radius):
    n_features=(2*radius+1)**2 #Total number of pixels in the neighborhood
    feat_array=np.zeros((im.shape[0],im.shape[1],n_features))
    for x in range(radius+1,im.shape[0]-(radius+1)):
        for y in range(radius+1,im.shape[1]-(radius+1)):
            feat_array[x,y,:]=extract_neighborhood(x,y,im,radius)
    all_pixels=feat_array.reshape(im.shape[0]*im.shape[1],n_features)
    pred_pixels=model.predict(all_pixels).astype(np.bool_)
    pred_image=pred_pixels.reshape(im.shape[0],im.shape[1])
    return pred_image
# %% Test functions
example_image=read_image('train/1/1.png')

print(example_image.shape)
plt.imshow(example_image, cmap='gray')
plt.axis('off')
plt.show()

# %%

train_annotation=read_annotation_file('train_anno.json')
print(train_annotation[1][1])
print(train_annotation[1][3])
# %%
radius=3
train_labels, train_features = generate_labeled_set(train_annotation,'train/', range(1,101), radius, 10)
print(train_labels.shape)
print(train_labels)
print(train_features.shape)
# %%
plt.subplot(1,2,1)
print(train_labels[0])
plt.imshow(train_features[0].reshape((7,7)), cmap='gray',vmin=0,vmax=1)
plt.axis('off')

plt.subplot(1,2,2)
print(train_labels[5])
plt.imshow(train_features[5].reshape((7,7)), cmap='gray',vmin=0,vmax=1)
plt.axis('off')
plt.show()

#%%
RF = RandomForestClassifier(n_estimators=500, max_depth=30, random_state=0)
RF.fit(train_features, train_labels)

#%%
test_labels, test_features = generate_labeled_set(train_annotation,'train/', range(101,106), radius, 500)
print(test_labels.shape)
print(test_labels)
print(test_features.shape)
#%%

pred_labels = RF.predict(test_features)
print(classification_report(pred_labels,test_labels))
print(confusion_matrix(pred_labels,test_labels))
print("Kappa =",cohen_kappa_score(pred_labels,test_labels))


#%%
sequence_id, frame_id = 102, 1
target_image = plt.imread(f"D:/Python/Projects/spotGEO/data/train/{sequence_id}/{frame_id}.png")
plt.imshow(target_image, cmap='gray')
plt.axis('off')
plt.show()


#%%
pred_image=classify_image(target_image, RF, radius)






#%%
plt.figure(figsize=(15,10))
plt.imshow(pred_image, interpolation='None')
plt.axis('off')
plt.show()















