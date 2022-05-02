import bisect
import numpy as np
from evalutils.evalutils import score_detection
import os, json

#... gather gt and predictions, extract probs, count tissue area
target_fps=[10, 20, 50, 100, 200, 300]
probs = np.sort(np.unique(probs))
thresholds = probs[::-1] #from big to small, important for bisect to get the selected tprs

tps, fns, fps = compute_scores(gts, preds, dist_thresh=8, thresholds=thresholds)
tprs = [tp/(tp+fn) for tp,fn in zip(tps, fns)]
av_fps = [fp/area_mm for fp in fps]
froc, sel_tprs = compute_froc_score(tprs, av_fps, target_fps)

def compute_scores(gt_coords, pred_coords, dist_thresh, thresholds):
    """ computes the overall tps, fns, fps per threshold given a list of gt and predictions for multiple rois
    gt_coords:[[x,y]],  pred_coords: [[x,y,prob]]
    """
    n_thresh = len(thresholds)

    tps = np.zeros((n_thresh)); fns = np.zeros((n_thresh)); fps = np.zeros((n_thresh));
    for i,thresh in enumerate(thresholds):
        for gt, pred in zip(gt_coords, pred_coords):
            if len(pred)==0:
                fns[i]+=len(gt) #no tp or fp
            elif len(gt)==0:
                fps[i]+=np.sum(pred[:,2]>=thresh) #no fn or tp
            else:
                thresh_pred = pred[np.where(pred[:,2]>=thresh)[0],:2]
                det_score = score_detection(ground_truth=gt, predictions=thresh_pred, radius=dist_thresh)
                tps[i]+=det_score.true_positives
                fns[i]+=det_score.false_negatives
                fps[i]+=det_score.false_positives
    return tps, fns, fps

def compute_froc_score(tprs:list, fps:list, target_fps:list, interpolate_edges=True):
    """
    Compute the average sensitivity at predefined false positive rates.

    Args:
        tprs (list): List of true positive rates for different thresholds.
        fps (list): List of (average) false positives for different thresholds.
        target_fps (list): Target fps for score calculation.

    Returns:
        float: Computed FROC score.
    """
    if interpolate_edges:
        #If starts after 0, add 0-entry
        if fps[0]!=0:
            fps.insert(0, 0)
            tprs.insert(0,0)

        #if ends before one of the target fps, add missing (horizontal interpolation)
        for t_fp in target_fps:
            if t_fp > max(fps):
                fps.append(t_fp)
                tprs.append(tprs[-1])

    n_thresh = len(tprs)
    print('computing froc score with %d thresholds and these average fps: %s' % (n_thresh, str(target_fps)))

    target_sensitivities = []
    for target_fp in target_fps:
        #old
        # target_index = bisect.bisect_left(fps, target_fp) #fps[:i]<target_fp -> takes x>=target_fp
        target_index = bisect.bisect_right(fps, target_fp)-1 #fps[:i]<=target_fp -> takes x<=target_fp
        target_index = min(target_index, n_thresh - 1)
        target_sensitivities.append(tprs[target_index])

    froc_score = sum(target_sensitivities) / len(target_fps)

    return froc_score, target_sensitivities

def _world_to_slide_coords(world_coords, spacing):
    return 1000*world_coords/spacing

def read_json_dict(path):
    with open(str(path), 'r') as f:
        data = json.load(f)
    return data

def _read_pred_points_probs(pred_json, spacing):
    """ returns [[x,y,prob]] """
    if os.stat(str(pred_json)).st_size == 0:
        return [] #empty result

    pred_dict = read_json_dict(pred_json)
    points_list = pred_dict['points']
    points = []
    for i, point_dict in enumerate(points_list):
        x, y, z = point_dict['point']
        x = _world_to_slide_coords(x, spacing)
        y = _world_to_slide_coords(y, spacing)
        prob = point_dict['probability']
        points.append([x, y, prob])
    return points