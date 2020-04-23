import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    
    # calculate the area of each box
    area_1 = (box_1[2]-box_1[0])*(box_1[3]-box_1[1])
    area_2 = (box_2[2]-box_2[0])*(box_2[3]-box_2[1])

    # find the top left and bottom right corners of the intersection
    mins = np.amin([box_1,box_2],axis = 0)
    maxes = np.amax([box_1,box_2],axis = 0)

    i = np.concatenate((maxes[:2],mins[2:]), axis = None)
    
    # calculate the area of the intersection
    if (i[2] >= i[0]) and (i[3] >= i[1]):
        area_i = (i[2]-i[0])*(i[3]-i[1])
    else:
        area_i = 0

    # calculate the area of the union
    area_u = area_1 + area_2 - area_i

    # calculate iou
    iou = area_i/area_u

    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''

    # define counters for the number of true positives, false poistives, and false negatives
    TP = 0
    FP = 0
    FN = 0

    # loop through all of the files
    for pred_file, pred in preds.items():
        # get the ground truths for each file
        gt = gts[pred_file]

        # get the number of ground truths for each file
        n_gt = len(gt)

        # set a counter for the number of predictions and true positives for the file
        n_preds = 0
        tp = 0

        # loop through the ground truths
        for i in range(len(gt)):
            # set a flag to check if we have matched a prediction to a ground truth
            match = False
            # loop through the predictions
            for j in range(len(pred)):
                # check if the confidence score of the predictions exceeds the threshold
                if (pred[j][4] >= conf_thr):
                    # increment the number of predictions
                    n_preds += 1

                    # compute the iou
                    iou = compute_iou(pred[j][:4], gt[i])

                    # check if the iou exceeds the threshold
                    if iou >= iou_thr and match == False:
                        # increment the number of true positives
                        tp += 1
                        # set the flag that we have found a match for the gound truth
                        match = True

        # calculate the number of false positives for the file
        fp = n_preds - tp

        # calculate the number of false negatives for the file
        fn = n_gt - tp

        # update the overall counters
        TP += tp
        FP += fp
        FN += fn

    return TP, FP, FN

# set the current path
path = os.getcwd()

# set a path for predictions and annotations:
preds_path = path + '\\data\\hw02_preds'
gts_path = path + '\\data\\hw02_annotations'

# load splits:
split_path = path + '\\data\\hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)

# define the iou thresholds
ious = [0.25, 0.50, 0.75]
# loop through the iou thresholds
for iou in ious:
    # For a fixed IoU threshold, vary the confidence thresholds for the training set.
    # first get the confidence thresholds
    confidence_thrs = []
    for fname, preds in preds_train.items():
        for pred in preds:
            confidence_thrs.append(pred[4])

    # sort the confidence thresholds
    confidence_thrs.sort()

    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou, conf_thr=conf_thr)

    # get the total number of predictions and ground truths
    n_preds_train = tp_train + fp_train
    n_gt_train = tp_train + fn_train

    precision_train = tp_train/n_preds_train
    recall_train = tp_train/n_gt_train
    
    # Plot the PR curve for the training set
    plt.figure(1)
    plt.plot(recall_train,precision_train, label = 'IoU = ' + str(iou))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Training Set')
    plt.legend()

if done_tweaking:
    # define the iou thresholds
    ious = [0.25, 0.50, 0.75]
    # loop through the iou thresholds
    for iou in ious:
        # For a fixed IoU threshold, vary the confidence thresholds for the training set.
        # first get the confidence thresholds
        confidence_thrs = []
        for fname, preds in preds_test.items():
            for pred in preds:
                confidence_thrs.append(pred[4])

        # sort the confidence thresholds
        confidence_thrs.sort()

        tp_test = np.zeros(len(confidence_thrs))
        fp_test = np.zeros(len(confidence_thrs))
        fn_test = np.zeros(len(confidence_thrs))
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=iou, conf_thr=conf_thr)

        # get the total number of predictions and ground truths
        n_preds_test = tp_test + fp_test
        n_gt_test = tp_test + fn_test

        # calculate the precision and recall
        precision_test = tp_test/n_preds_test
        recall_test = tp_test/n_gt_test

        # Plot the PR curve for the test set
        plt.figure(2)
        plt.plot(recall_test,precision_test, label = 'Iou = ' + str(iou))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Test Set')
        plt.legend()
    
# Show the plot(s)
plt.show()
