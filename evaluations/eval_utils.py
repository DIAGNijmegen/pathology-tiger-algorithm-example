from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from pycm import ConfusionMatrix

from matplotlib import pyplot as plt


class CmScorer(object):
    def __init__(self, class_map, incremental=False, ignore_gt_zeros=True,
                 gt_remap={}, pred_remap={}, remap_inplace=False):
        """
        class_map: {label:name}
        incremental: accumulates metrics over repeated calls
        ignore_gt_zeros: ignores 0s in the ground truth
        gt_remap and pred_remap: remap values in target and prediction in-place
        """
        self._ignore_gt_zeros = ignore_gt_zeros
        self._incremental = incremental

        self.gt_remap = gt_remap
        self.pred_remap = pred_remap
        self.remap_inplace=remap_inplace

        self.class_map = class_map

        self.reset()

    def reset(self):
        self.cm = None

    def get_score(self):
        return self._get_score(self.cm)

    def _get_score(self, cm:ConfusionMatrix):
        results = dict(cm_map=cm.table, cm=cm.to_array())
        results['overall_metrics'] = cm.overall_stat
        results['class_metrics'] = cm.class_stat
        results['classes'] = cm.classes
        return results

    def _remap(self, arr, old_new_map):
        #if overlap - have to copy, otherwise could also do inplace
        if old_new_map.values() in old_new_map.keys() or not self.remap_inplace:
            arr_new = arr.copy()
            for old_val, new_val in old_new_map.items():
                arr_new[arr==old_val] = new_val
            arr = arr_new
        else:
            for old_val, new_val in old_new_map.items():
                arr[arr==old_val] = new_val
        return arr

    def __call__(self, gt, pred, show=False):
        """
        gt: ground truth numpy array
        pred: pred numpy array
        returns the confusion matrix and metrics for the given ground-truth and pred mask arrays as dict {label:dice}.
        call get_score in the incremental case to get the full score
        """
        if not self._incremental:
            self.reset()
        gt = self._remap(gt, self.gt_remap)
        pred = self._remap(pred, self.pred_remap)
        if self._ignore_gt_zeros:
            pred = pred[gt!=0]
            gt = gt[gt!=0]

        class_labels = list(sorted(self.class_map.keys()))
        class_names = [self.class_map[k] for k in class_labels]
        cm_arr = confusion_matrix(gt, pred, labels=class_labels)

        matrix = {}
        for i,cl_true in enumerate(class_names):
            cl_matrix = {}
            matrix[cl_true] = cl_matrix
            for j,cl_pred in enumerate(class_names):
                cl_matrix[cl_pred] = int(cm_arr[i,j])

        cm = ConfusionMatrix(matrix=matrix)

        if self.cm is None:
            self.cm = cm
        else:
            self.cm = self.cm.combine(cm)

        if show:
            self.cm.plot(normalized=True)
            plt.show()

        return self._get_score(cm)

class TigerSegmScorer(CmScorer):
    def __init__(self, incremental=True, **kwargs):
        #1: Invasive Tumor, 2: Tumor-assoc. Stroma, 3: DCIS, 4: Healthy, 5: Necrosis, 6: Inflamed Stroma, 7: Rest
        gt_remap={4:3, 5:3, 6:2, 7:3}

        #pred mask: map all other-classes to 3
        pred_remap = {k:3 for k in range(256)}
        pred_remap.update({1:1, 2:2, 6:2})
        super().__init__(class_map={1:'Tumor',2:'Stroma',3:'Rest'}, incremental=incremental,
                         gt_remap=gt_remap, pred_remap=pred_remap, ignore_gt_zeros=True, **kwargs)

    def _get_score(self, *args, **kwargs):
        """ returns just the cm and the dice metrics """
        metrics = super()._get_score(*args, **kwargs)
        dice_metrics = metrics['class_metrics']['F1']
        dice_metrics['cm'] = metrics['cm']
        dice_metrics['classes'] = metrics['classes']
        return dice_metrics

def _test_scorers():
    gt =   np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 6, 3, 3, 4, 5])#5 x ignore, 3 x tumor, 2 x stroma, 4 x rest
    pred = np.array([0, 0, 0, 7, 0, 0, 1, 1, 9, 2, 6, 5, 1, 1])#6 x rest, 2 x tumor, 1xrest, 2 x stroma, 3x rest

    #cm classes: rest, stroma, tumor
    expected_cm = [[1, 1, 2],
                   [1, 1, 0],
                   [1, 0, 2]]

    scorer = TigerSegmScorer(incremental=True)
    gt1 = gt[:5]
    gt2 = gt[5:]
    pred1 = pred[:5]
    pred2 = pred[5:]
    score1 = scorer(gt1, pred1) #partial results
    score2 = scorer(gt2, pred2)
    result = scorer.get_score()
    cm = result.pop('cm')
    assert (cm==np.array(expected_cm)).all()
    print('confusion matrix:')
    print(cm)
    print(result)

if __name__ == '__main__':
    _test_scorers()

