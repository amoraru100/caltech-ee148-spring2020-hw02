import os
import numpy as np
import json
from PIL import Image

def add_bounding_boxes(I, bounding_boxes):
    # set the color of the boxes [R,G,B]
    box_color = [0,0,255]

    for box in bounding_boxes:
        tl_row = box[0]
        tl_column = box[1]
        br_row = box[2]
        br_column = box[3]

        # draw the top of the box
        I[tl_row,tl_column:br_column] = box_color

        # draw the left side of the box
        I[tl_row:br_row,tl_column] = box_color

        # draw the right side of the box
        I[tl_row:br_row,br_column] = box_color

        # draw the bottom of the box
        I[br_row,tl_column:br_column] = box_color

    return I

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

# set the current path
path = os.getcwd()

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = r'C:\Users\amora\Documents\Caltech\EE 148\HW1\data\RedLights2011_Medium'

# set a path for saving the visualizations:
visuals_path = path + '\\Visuals\\'

'''
Create the visuals
'''

# define the filenames of the image(s) you want to use to get the heatmaps
file_names = ['RL-044.jpg']

for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.array(I)
    
    # define the color and size for the templates
    color = [255,200,100]
    sizes = [5,10,15]
    output = []

    for size in sizes:
        # make each template
        T = get_circle_template(size, color)

        # save each template
        image = Image.fromarray(T.astype(np.uint8))
        image.save(visuals_path + 'template_' + str(size) + '.jpg')

        # make each heatmap
        heatmap = compute_convolution(I, T)

        # convert the heatmap into a greyscale image
        heatmap_image = np.expand_dims(heatmap, axis = 2)
        heatmap_image = np.repeat(heatmap_image,3, axis = 2)*255

        # save each heatmap
        image = Image.fromarray(heatmap_image.astype(np.uint8))
        image.save(visuals_path + '\\heatmap_' + str(size) + '_' + file_names[i])
        