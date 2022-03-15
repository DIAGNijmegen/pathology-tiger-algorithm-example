import bisect
import numpy as np
from evalutils.evalutils import score_detection

#... gather gt and predictions, extract probs, count tissue area
target_fps=[5, 10, 20, 50, 100, 200, 500]
probs = np.sort(np.unique(probs))
thresholds = probs[::-1] #from big to small, important for bisect to get the selected tprs

tps, fns, fps = compute_scores(gts, preds, dist_thresh=8, thresholds=thresholds)
tprs = [tp/(tp+fn) for tp,fn in zip(tps, fns)]
av_fps = [fp/area_mm for fp in fps]
froc, sel_tprs = compute_froc_score(tprs, av_fps, target_fps)

def compute_scores(gt_coords, pred_coords, dist_thresh, thresholds):
    """ computes the overall tps, fns, fps per threshold given a list of gt and predictions for multiple slides
    gtcoords:[[x,y]], predcoords: [[x,y,prob]] """

    n_thresh = len(thresholds)

    tps = np.zeros((n_thresh)); fns = np.zeros((n_thresh)); fps = np.zeros((n_thresh));
    for i,thresh in enumerate(thresholds):
        for gt, pred in zip(gt_coords, pred_coords):
            if len(pred)==0:
                fns[i]+=len(gt)
            elif len(gt)==0:
                fps[i]+=len(pred)
            else:
                thresh_pred = pred[np.where(pred[:,2]>=thresh)[0],:2]
                det_score = score_detection(ground_truth=gt, predictions=thresh_pred, radius=dist_thresh)
                tps[i]+=det_score.true_positives
                fns[i]+=det_score.false_negatives
                fps[i]+=det_score.false_positives
    return tps, fns, fps

def compute_froc_score(tprs, fps, target_fps):
    """
    Compute the average sensitivity at predefined false positive rates.

    Args:
        tprs (list): List of true positive rates for different thresholds.
        fps (list): List of (average) false positives for different thresholds.
        target_fps (list): Target fps for score calculation.

    Returns:
        float: Computed FROC score.
    """
    n_thresh = len(tprs)
    print('computing froc score with %d thresholds and these average fps: %s' % (n_thresh, str(target_fps)))

    target_sensitivities = []
    for target_fp in target_fps:
        target_index = min(bisect.bisect_left(fps, target_fp), n_thresh - 1)
        target_sensitivities.append(tprs[target_index])

    froc_score = sum(target_sensitivities) / len(target_fps)

    return froc_score, target_sensitivities