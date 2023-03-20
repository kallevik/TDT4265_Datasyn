import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # REMEMBER EDGE CASE OF DIV BY ZERO

    # Compute intersection SNITT
    xA = max(prediction_box[0], gt_box[0]) 
    yA = max(prediction_box[1], gt_box[1])
    xB = min(prediction_box[2], gt_box[2]) 
    yB = min(prediction_box[3], gt_box[3])  

    Area_common = max(0, xB - xA) * max(0, yB - yA)
    
    # Compute union
    Area_pred = (prediction_box[2] - prediction_box[0] ) * (prediction_box[3] - prediction_box[1] )
    Area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    Area_union = float(Area_pred + Area_gt - Area_common)

    iou = Area_common / Area_union

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    #If no positive preditions
    #1 since no false positive to reduce precision
    if num_tp + num_fp == 0:
        return 1.0

    precision = num_tp / (num_tp + num_fp)

    return precision


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    #If no positive predicitons
    #0 since no true positive to capture
    if num_tp + num_fn == 0:
        return 0.0
    
    recall = num_tp /(num_tp + num_fn)

    return recall


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # YOUR CODE HERE

    # Find all possible matches with a IoU >= iou threshold
    # Sort all matches on IoU in descending order
    # Find all matches with the highest IoU threshold

    #Get 
    num_pred_boxes = prediction_boxes.shape[0]
    num_gt_boxes = gt_boxes.shape[0]
    #print(num_gt_boxes, num_pred_boxes)
    
    #Test if empty arguments 
    if num_gt_boxes == 0 or num_gt_boxes == 0:
        return np.array([]), np.array([])
   
    # Create a matrix to hold the IoU scores for all pairs of boxes
    iou_matrix = np.zeros((num_pred_boxes, num_gt_boxes)) 

    # Compute the IoU scores for each pair of boxes
    for i in range(num_pred_boxes):
        for j in range(num_gt_boxes):
            iou_matrix[i, j] = calculate_iou(prediction_boxes[i], gt_boxes[j])
    #print(iou_matrix)
    
    # Create arrays to hold the matched boxes
    matched_prediction_boxes = []
    matched_gt_boxes = []

    # Iterate through the IoU matrix in decreasing order of IoU scores
    while True:
        max_iou = np.max(iou_matrix)
        if max_iou < iou_threshold:
            break

        # Find the indices of the box pair with the highest IoU score
        max_iou_pred_index, max_iou_gt_index = np.argwhere(iou_matrix == max_iou)[0]

        # Remove the matched boxes from consideration in the future
        # so we can iterate through the highest values in decending order
        iou_matrix[max_iou_pred_index, :] = 0
        iou_matrix[:, max_iou_gt_index] = 0


        # Add the matched boxes to the output arrays       
        matched_prediction_boxes.append(prediction_boxes[max_iou_pred_index])
        matched_gt_boxes.append(gt_boxes[max_iou_gt_index])

    return np.array(matched_prediction_boxes), np.array(matched_gt_boxes)



def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    num_pred_boxes = prediction_boxes.shape[0]
    num_gt_boxes = gt_boxes.shape[0]
    tp, fp, fn = 0, 0, 0

    if num_pred_boxes > 0 and num_gt_boxes > 0:
        # get all box matches
        pred_matches, gt_matches = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
        num_pred_matches = pred_matches.shape[0]
        num_gt_matches = gt_matches.shape[0]

        # calculate true positives
        tp = num_pred_matches

        # calculate false positives
        fp = num_pred_boxes - num_pred_matches

        # calculate false negatives
        fn = num_gt_boxes - num_gt_matches

    elif num_pred_boxes > 0 and num_gt_boxes == 0:
        # if no gt boxes, all predicted boxes are false positives
        fp = num_pred_boxes

    elif num_pred_boxes == 0 and num_gt_boxes > 0:
        # if no predicted boxes, all gt boxes are false negatives
        fn = num_gt_boxes

    return {"true_pos": tp, "false_pos": fp, "false_neg": fn}


def calculate_precision_recall_all_images(all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    tp, fp, fn = 0, 0, 0
    for i in range(len(all_prediction_boxes)):

        pred_boxes = all_prediction_boxes[i]
        gt_boxes = all_gt_boxes[i]
        matched_pred_boxes, matched_gt_boxes = get_all_box_matches(pred_boxes, gt_boxes, iou_threshold)

        # Update the counts of true positives, false positives, and false negatives
        tp += len(matched_pred_boxes)
        fp += len(pred_boxes) - len(matched_pred_boxes)
        fn += len(gt_boxes) - len(matched_gt_boxes)

    precision = calculate_precision(tp, fp, fn)
    recall = calculate_recall(tp, fp, fn)
    return precision, recall

def get_precision_recall_curve(all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!
    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]

        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]

        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]
            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE

    #print(len(all_prediction_boxes)) #2
    precisions = [] 
    recalls = []
    for threshold in confidence_thresholds: # 0.0 0.001 0.002 ... 0.9999
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for i in range(len(all_prediction_boxes)): #0 1 
           
            # Get predicted and ground truth boxes
            pred_boxes = all_prediction_boxes[i] #b1, b2
            gt_boxes = all_gt_boxes[i]           #b2, b2
            scores = confidence_scores[i]        # s, s
            
            # Get the indices of the predicted boxes that have confidence scores
            # above the threshold
            indices = np.where(scores >= threshold)[0] # [0 1 2 3] ... [3]
            if len(indices) == 0:
                continue
            print("indices: ", indices)
            print("threshold:", threshold)
            for i in indices:
                print("score > threshold: ",scores[i])

            
            # Match each predicted box to a ground truth box
            matches = get_all_box_matches(pred_boxes[indices], gt_boxes, iou_threshold)
            print(pred_boxes[indices])
            # Compute number of true/false positives/negatives
            true_positives += len(matches)
            false_positives += len(indices) - len(matches)
            false_negatives += len(gt_boxes) - len(matches)

        print(true_positives, false_positives, false_negatives)
        if true_positives == 0 and false_positives == 0 and false_negatives == 0:
            break
        
        # Compute precision and recall
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        #precision = calculate_precision(true_positives, false_positives, false_negatives)
        #recall = calculate_recall(true_positives, false_positives, false_negatives)
        
        # Append to list of precisions and recalls
        precisions.append(precision)
        recalls.append(recall)
    


    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)


    average_precisions = []
    for cls in range(len(precisions)):
        # Pad precision and recall arrays with 0 and 1 respectively at both ends
        precisions_pad = np.concatenate([[0], precisions[cls], [0]])
        recalls_pad = np.concatenate([[0], recalls[cls], [1]])

        # Compute the area under the curve using the trapezoidal rule
        area = np.trapz(precisions_pad, recalls_pad)

        # Calculate the average precision at each recall level
        ap = 0
        for level in recall_levels:
            mask = recalls_pad >= level
            if np.any(mask):
                ap += np.max(precisions_pad[mask]) * (recalls_pad[mask][-1] - recalls_pad[mask][0])
        ap /= len(recall_levels)

        average_precisions.append(ap)

    # Take the mean of average precision values for all classes
    mean_average_precision = np.mean(average_precisions)

    return mean_average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))

def main():
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)

if __name__ == "__main__":
    main()
