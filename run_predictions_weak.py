import os
import numpy as np
import json
from PIL import Image

def normalize(v):
    '''
    This function returns the normalized vector v
    '''
    norm = np.linalg.norm(v)
    if norm ==0:
        return v
    return v/norm

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''

    # define the number of rows, columns and channels of the image and template
    (I_rows,I_cols,I_channels) = np.shape(I)
    (T_rows,T_cols,T_channels) = np.shape(T)

    # normalize the template
    T_red = normalize(T[...,0]).flatten()
    T_green = normalize(T[...,1]).flatten()
    T_blue = normalize(T[...,2]).flatten()

    # create an empty heatmap
    heatmap = np.zeros((I_rows-T_rows,I_cols-T_cols))

    # convolve the template with the image
    # loop through the image
    for r in range(I_rows-T_rows):
        for c in range(I_cols-T_cols):

            # select a region of the image with the same size as the template
            d = I[r:r+T_rows,c:c+T_cols,:]

            # normalize the selected region of the image
            d_red = normalize(d[...,0]).flatten()
            d_green = normalize(d[...,1]).flatten()
            d_blue = normalize(d[...,2]).flatten()

            red_score = np.inner(d_red,T_red)
            green_score = np.inner(d_green,T_green)
            blue_score = np.inner(d_blue,T_blue)

            avg_score = (red_score+blue_score+green_score)/3
                
            heatmap[r][c] = avg_score

    return heatmap


def predict_boxes(heatmap, T, reach = 5):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []
    
    rows, cols = np.shape(heatmap)
    box_height, box_width, T_channels = np.shape(T)

    # create a new array that is just the heatmap with a border of all 0s so we don't have to deal with edge cases
    new_heatmap = np.zeros([rows+(2*reach),cols+(2*reach)])
    new_heatmap[reach:-reach,reach:-reach] = heatmap

    # loop through each inner element of the new heatmap with borders of 0s
    for r in range(reach,rows+reach):
        for c in range(reach,cols+reach):
            mid = new_heatmap[r][c]
            # check to see if the value exceeds the threshold
            if mid >= 0.8:
                # check if the value is the max value of the square
                if mid == np.amax(new_heatmap[r-reach:r+reach+1,c-reach:c+reach+1]):
                    output.append([r-reach,c-reach, r-reach+box_height, c-reach+box_width, mid])

    return output

def get_circle_template(size, color):
    '''
    This function defines a square kernel matrix  of size (2*size+1) X (2*size+1)
    for a circle of radius size * 3/4 centered in the square with the color given
    by the [R,G,B] value of color with a black background
    '''
    
    # first check if size is a valid number
    if size < 1:
        print('size cannot be < 1')
        return
    
    # define the size of the square kernel matix
    T = np.zeros((2*size+1,2*size+1,3))

    for r in range(2*size+1):
        for c in range(2*size+1):
            if np.sqrt((r-size)**2+(c-size)**2) < size*3/4:
                T[r][c] = color
    
    return T 

def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    color = [255,200,100]
    sizes = [10]
    output = []

    for size in sizes:
        T = get_circle_template(size, color)

        heatmap = compute_convolution(I, T)
        output.extend(predict_boxes(heatmap, T))

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output


# set the current path
path = os.getcwd()

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = r'C:\Users\amora\Documents\Caltech\EE 148\HW1\data\RedLights2011_Medium'

# load splits: 
split_path = path + '\\data\\hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = path + '\\data\\hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''

preds_train = {}
for i in range(len(file_names_train)):
    print(file_names_train[i])

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):
        print(file_names_test[i])
        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
