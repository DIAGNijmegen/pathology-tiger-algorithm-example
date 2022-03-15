""""
This script computes segmentation (classification) metrics (over rois) for whole slide images using pycm.
Since loading the complete slide segmentation might result in oom, the computation is done
per roi using target_masks!=0. To further reduce memory requirements the computation can be done
at lower resolutions (can lead to minor rounding errors).

"""

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import numpy as np
import os, time
from datetime import timedelta
import yaml
import glob
import json
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from pathlib import Path
from pycm import ConfusionMatrix

from evaluations.eval_utils import *

def _bar_autolabel(ax, rects,  formatter='%.2f', yoffset=0.05):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., (1+yoffset)*height,
                formatter % height,
                ha='center', va='bottom')


def _bar_autolabel_horizontal(ax, rects, formatter='%.2f', xoffset=0.05):
    rect_widths = sorted([rect.get_width() for rect in rects])
    xoffset = rect_widths[len(rect_widths)//2]*xoffset
    for rect in rects:
        width = rect.get_width()
        ax.text(rect.get_width()+xoffset, rect.get_y()+0.5*rect.get_height(),
                formatter % width,
                ha='center', va='center')


def _plot_barchart(values, names, err=None, normalize=False, ylabel=None, title=None, horizontal=False,
                   autolabel_formatter='%.2f', autolabel_offset=.05):
    values = np.nan_to_num(values)
    if normalize:
        values = values.astype('float') / values.sum()
        values = np.nan_to_num(values)
    else:
        autolabel_formatter = '%d'

    y_pos = np.arange(len(names))

    fig, ax = plt.subplots()

    if horizontal:
        rects = ax.barh(y_pos, values, xerr=err, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # labels read top-to-bottom
        _bar_autolabel_horizontal(ax, rects, formatter=autolabel_formatter, xoffset=autolabel_offset)
    else:
        if err is not None: raise ValueError('err not implemtend')
        rects = ax.bar(y_pos, values, align='center')
        plt.xticks(y_pos, names, rotation=45, formatter=autolabel_formatter, yoffset=autolabel_offset)
        _bar_autolabel(ax, ax.patches)

    if title is not None:
        plt.title(title)
    if ylabel is not None:
        plt.ylabel(ylabel)

    return fig, ax

def show(*arr): #for debugging
    """ shows the arrays in a single row """
    n_cols = len(arr)
    for i in range(n_cols):
        ax = plt.subplot(1,n_cols,i+1)
        ax.imshow(arr[i])
    plt.show()


def merge_if_overlapping(a, b):
    bottom = np.max([a[0],b[0]])
    top = np.min([a[0] + a[2], b[0] + b[2]])
    left = np.max([a[1],b[1]])
    right = np.min([a[1] + a[3], b[1] + b[3]])

    do_intersect = bottom < top and left < right

    if do_intersect:
        x_min = np.min([a[1],b[1]])
        y_min = np.min([a[0],b[0]])
        x_max = np.max([a[1]+a[3],b[1]+b[3]])
        y_max = np.max([a[0]+a[2],b[0]+b[2]])
        new_bbox = (y_min, x_min, y_max - y_min, x_max - x_min)
        return True, new_bbox

    return False, []


def get_gt_bounding_boxes(ground_truth_wsi, spacing, overview_spacing=8.0):
    """
    Bounding boxes will have form of (y,x,height,width)
    """
    if ground_truth_wsi.spacings[-1]+0.25 < overview_spacing:
        print('chaning overview_spacing from %f to %f' % (overview_spacing, ground_truth_wsi.spacings[-1]))
        overview_spacing = ground_truth_wsi.spacings[-1]
    overview_spacing = ground_truth_wsi.refine(overview_spacing)
    overview_level = ground_truth_wsi.level(spacing=overview_spacing)

    overview_shape = ground_truth_wsi.shapes[overview_level]
    gt_patch = ground_truth_wsi.read(overview_spacing, 0,0, overview_shape[0], overview_shape[1]).squeeze()

    spacing = ground_truth_wsi.refine(spacing)
    spacing_level = ground_truth_wsi.level(spacing)
    spacing_shape = ground_truth_wsi.shapes[spacing_level]
    spacing_ratio = overview_spacing / spacing

    #with smaller padding some pixels are missed
    padding = 5
    # noticed less grabbed pixels when overview_spacing << spacing
    if spacing_ratio <= 0.25:
        padding = 10
        #for the lowest resolution (highest level) even very large padding results in missing some pixels
        print('Warning: Very low resolution, results might be not precise')

    labels, n = label(gt_patch > 0, return_num=True, connectivity=1)
    regions = regionprops(labels)
    bboxes = [(max(0,int(np.floor((region.bbox[0] - padding) * spacing_ratio))-padding),
               max(0,int(np.floor((region.bbox[1] - padding) * spacing_ratio))-padding),
               int(np.ceil((region.bbox[2] - region.bbox[0] + padding*2) * spacing_ratio))+padding, #*2 to account for previously substracted padding
               int(np.ceil((region.bbox[3] - region.bbox[1] + padding*2) * spacing_ratio))+padding
               ) for region in regions]
    bboxes = [(box[0], box[1], min(box[2],spacing_shape[0]-box[0]), min(box[3],spacing_shape[1]-box[1])) for box in bboxes]
    merge_overlapping_bboxes(bboxes)

    return bboxes

def merge_overlapping_bboxes(bboxes):
    candidate_count = 0
    while candidate_count < len(bboxes):
        candidate_count += 1
        overlap = False
        candidate_box = bboxes.pop(0)
        for index, compare_box in enumerate(bboxes):
            overlapping, new_bbox = merge_if_overlapping(candidate_box, compare_box)
            if overlapping:
                overlap = True
                candidate_count = 0
                bboxes.pop(index)
                bboxes.append(new_bbox)
                break
        if not overlap:
            bboxes.append(candidate_box)

def _compute_cm_by_loading_everything(target_wsi, mask_wsi, spacing, target_class_mapping, pred_class_mapping,
                                      ignore_mask_zero=True):
    """ classes: <nr>:<name>; mapping: <nr>:<nr>"""
    #takes the final key->class name mapping is the pred_class_mapping adapted for when multiple class-keys are mapped
    #to the same class. In that case, the class gets the smallest key (thus most of the original asap colors can be shown)
    classname_key_mapping = {}
    for key, cname in pred_class_mapping.items():
        classname_key_mapping[cname] = min(classname_key_mapping.get(cname,1e12), key)
    class_mapping = _invert_dict(classname_key_mapping)
    classkeys = sorted(class_mapping.keys())
    classnames = [class_mapping[key] for key in classkeys]

    cm_boxes = []
    mask_patch = mask_wsi.content(spacing).squeeze()
    target_patch = target_wsi.content(spacing).squeeze()
    if pred_class_mapping is not None:
        # mask_patch_old = mask_patch
        mask_patch = remap_mask_arr(mask_patch, pred_class_mapping, class_mapping, check_all_mapped=False)

    if target_class_mapping is not None:
        # target_patch_old = target_patch
        target_patch = remap_mask_arr(target_patch, target_class_mapping, class_mapping)

    # remove segmentation outside ground truth area
    mask_patch *= (target_patch > 0)

    #ignore
    target_patch_non_zero = target_patch[target_patch != 0]
    mask_patch_non_zero = mask_patch[target_patch != 0]
    if ignore_mask_zero:
        target_patch_non_zero = target_patch_non_zero[mask_patch_non_zero!=0]
        mask_patch_non_zero = mask_patch_non_zero[mask_patch_non_zero!=0]

    cm = confusion_matrix(target_patch_non_zero, mask_patch_non_zero, labels=classkeys)
    cm_boxes.append((0, 0, target_patch.shape[1], target_patch.shape[0]))#todo verify that bbox col, row and not vice versa

    return cm, cm_boxes

def _compute_cm_with_bbox(target_wsi, mask_wsi, spacing, target_class_mapping, pred_class_mapping, ignore_mask_zero=True):
    """ mapping: <nr>:<nr>"""
    classname_key_mapping = {}
    for key, cname in pred_class_mapping.items():
        classname_key_mapping[cname] = min(classname_key_mapping.get(cname,1e12), key)
    class_mapping = _invert_dict(classname_key_mapping)
    classkeys = sorted(class_mapping.keys())
    classnames = [class_mapping[key] for key in classkeys]

    n_classes = len(classnames)
    cm = np.zeros((n_classes,n_classes))
    bboxes = get_gt_bounding_boxes(target_wsi, spacing)  # bboxes are x,y
    print("found {} ground truth regions".format(len(bboxes)))
    cm_boxes = []

    tmp_masks = []; tmp_targets = []; cmbs = []
    for b,bbox in enumerate(tqdm(bboxes)):
        mask_patch = mask_wsi.read(spacing, bbox[0], bbox[1], bbox[2], bbox[3]).squeeze()
        target_patch = target_wsi.read(spacing, bbox[0], bbox[1], bbox[2], bbox[3]).squeeze()
        if pred_class_mapping is not None:
            mask_patch = remap_mask_arr(mask_patch, pred_class_mapping, class_mapping, check_all_mapped=False)

        if target_class_mapping is not None:
            target_patch = remap_mask_arr(target_patch, target_class_mapping, class_mapping)

        if np.count_nonzero(target_patch) == 0:
            print('Ignoring box %d/%d' % (b+1, len(bboxes)))
            continue

        # remove segmentation outside ground truth area
        mask_patch *= (target_patch > 0)

        #ignore
        target_patch_non_zero = target_patch[target_patch != 0]
        mask_patch_non_zero = mask_patch[target_patch != 0]
        if ignore_mask_zero:
            target_patch_non_zero = target_patch_non_zero[mask_patch_non_zero!=0]
            mask_patch_non_zero = mask_patch_non_zero[mask_patch_non_zero!=0]
        if len(target_patch_non_zero)==0:
            #if ignore_mask_zero and some prediction classes are ignored (no entry in 'pred_class_mapping') the
            #grabbed box might contain the ignored class and thus be now empty
            continue

        cm_b = confusion_matrix(target_patch_non_zero, mask_patch_non_zero, labels=classkeys)
        cm_boxes.append(bbox)
        cm += cm_b
        cmbs.append(cm_b)

    return cm, cmbs, cm_boxes

def _compute_cm_metrics(confusion_matrix, class_names, output_dir=None,
                        class_metrics = {'F1':'F1', 'J':'Jaccard', 'PPV':'Precision', 'TPR':'Recall(TPR)', 'TNR':'TNR'},
                        overall_metrics={'Overall ACC':'Acc', 'Overall RACC':'Random Acc', 'Kappa':'Kappa', 'F1 Macro':'F1 Macro', 'F1 Micro':'F1 Micro',
                                         'PPV Macro': 'Precision Macro', 'PPV Micro':'Precision Micro', 'TPR Macro':'Recall(TPR) Macro', 'TPR Micro':'Recall(TPR) Micro'}):
    """ class_metrics and overall_metrics: pycm-name -> your-name, e.g. 'J'->'jaccard'
     output_dir for saving the confustion matrix and additional metrics"""
    #{class_name_true:{class_name_pred:value}}
    matrix = {}
    for i,cl_true in enumerate(class_names):
        cl_matrix = {}
        matrix[cl_true] = cl_matrix
        for j,cl_pred in enumerate(class_names):
            cl_matrix[cl_pred] = int(confusion_matrix[i,j])

    cm = ConfusionMatrix(matrix=matrix)

    results = {'overall_metrics':{}, 'class_metrics':{}}
    results['overall_metrics'].update({result_name:cm.overall_stat[cm_name] for cm_name,result_name in overall_metrics.items()})
    results['class_metrics'].update({result_name:cm.class_stat[cm_name] for cm_name,result_name in class_metrics.items()})

    if output_dir is not None and len(str(output_dir))>0:
        cm.save_csv(str(Path(output_dir)/'detailed'))
    return results

def _get_classnames_from_labelmap(class_map):
    class_names = []
    keys = sorted(class_map.keys())
    for key in keys:
        cname = class_map[key]
        if cname not in class_names:
            class_names.append(cname)
    return class_names

def _invert_dict(d):
    inv = {v: k for k, v in d.items()}
    return inv

def plot_data_dist(conf_mat, classnames, out_dir):
    out_dir = Path(out_dir)
    sample_dist = conf_mat.sum(axis=1)
    fig, ax = _plot_barchart(sample_dist, classnames, title='Data Distribution', normalize=True, horizontal=True,
                             autolabel_formatter='%.4f', autolabel_offset=0.25)
    out_path = out_dir/'data_dist.png'
    fig.savefig(str(out_path), bbox_inches="tight")
    # print('%s saved' % out_path)
    fig, ax = _plot_barchart(sample_dist, classnames, title='Data Distribution', normalize=False, horizontal=True,
                             autolabel_formatter='%.4f', autolabel_offset=0.25)
    out_path = out_dir/'data_dist_abs.png'
    fig.savefig(str(out_path), bbox_inches="tight")
    # print('%s saved' % out_path)

def eval_segm_masks(target_pathes, pred_pathes, pred_class_mapping, target_class_mapping, output_dir, spacing=0.5,
                  save_region_stats=False, show=False, slow=False, per_bbox=False):
    """ classes: <nr>:<name>
    """
    print('output_dir %s' % str(output_dir))
    print('pred_class_mapping: %s, target_class_mapping: %s' % (str(pred_class_mapping), str(target_class_mapping)))
    output_dir = Path(output_dir)

    if not is_list(target_pathes):
        target_pathes = [target_pathes]
    if not is_list(pred_pathes):
        pred_pathes = [pred_pathes]
    n_masks = len(target_pathes)
    pred_classnames = _get_classnames_from_labelmap(pred_class_mapping)
    target_classnames = _get_classnames_from_labelmap(target_class_mapping)
    if set(pred_classnames)!=set(target_classnames):
        raise ValueError('prediction and target class names do not match: %s and %s' % \
                         (str(pred_classnames), str(target_classnames)))
    classname_label_map = {i:cname for i,cname in enumerate(pred_classnames)}

    results = {}

    start = time.time()
    cm_list = []; filenames = []
    for i,ground_truth_filepath in enumerate(target_pathes):
        mask_path = pred_pathes[i]
        if not Path(mask_path).exists():
            raise ValueError('mask %s doesnt exist' % str(mask_path))
        if not Path(ground_truth_filepath).exists():
            raise ValueError('gt %s doesnt exist' % str(ground_truth_filepath))

        file_name = Path(ground_truth_filepath).stem
        filenames.append(file_name)

        print('Processing %d/%d: %s' % (i+1, len(target_pathes), file_name))

        mask_wsi = ImageReader(image_path=mask_path)
        target_wsi = ImageReader(image_path=ground_truth_filepath)

        if slow:#this is for checking whether the box impl. is correct
            cm_i, cm_i_boxes = _compute_cm_by_loading_everything(target_wsi, mask_wsi, spacing,
                                                                 pred_class_mapping=pred_class_mapping,
                                                                 target_class_mapping=target_class_mapping)
        else:
            cm_i, cm_is, cm_i_boxes = _compute_cm_with_bbox(target_wsi, mask_wsi, spacing, pred_class_mapping=pred_class_mapping,
                                                            target_class_mapping=target_class_mapping)
        cm_list.append(cm_i)
        results[file_name] = _compute_cm_metrics(cm_i, pred_classnames)

        worst_candidates = filenames
        if per_bbox:
            worst_candidates = []
            for ib,cm_ib in enumerate(cm_is):
                cm_ib_box = cm_i_boxes[ib]
                result_ib = _compute_cm_metrics(cm_ib, pred_classnames)
                fbox_name = file_name+'_%d_%d' % (cm_ib_box[0],cm_ib_box[1])
                results[fbox_name] = result_ib
                worst_candidates.append(fbox_name)
        print('Metrics for %s: %s' % (file_name, str(results[file_name])))

        if save_region_stats:
            for i,cm_box in enumerate(cm_i_boxes):
                results[file_name]['box_%d' % i] = _compute_cm_metrics(cm_box, pred_classnames)
        mask_wsi.close()
        target_wsi.close()

    output_dir.mkdir(exist_ok=True, parents=True)

    cm_all = np.sum(np.array(cm_list), axis=0).astype(np.int)
    results.update(_compute_cm_metrics(cm_all, pred_classnames, output_dir=output_dir))
    print('All results: %s' % str(results))

    plot_cm(cm_all, pred_classnames, normalize=False, save_path=output_dir/'cm_abs.png')
    plot_cm(cm_all, pred_classnames, normalize=True, save_path=output_dir/'cm.png', show=show)
    plot_data_dist(cm_all, pred_classnames, output_dir)
    output_path = output_dir/'metrics.yaml'
    with open(str(output_path), 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False, indent=4)
    print('output: %s' % str(output_dir))
    duration = time.time()-start
    print ('Done in %s' % (timedelta(seconds=duration)))
    return results


def _example_one(slow=False, spacing=0.5):
    target_path = "some_slide.tif"
    pred_path = "some_pred.tif"
    output_dir = "./out/tiger_eval_test"
    ##0: Exclude, 1: Invasive Tumor, 2: Stroma, 3: In-situ Tumor, 4: Normal glands, 5: Necrosis (6: Lymphocytes), 7: Rest
    class_mapping = {1: 'Tumor', 2: 'Stroma', 3: 'Rest', 4: 'Rest', 5: 'Rest', 6: 'Stroma', 7: 'Rest'}
    results = eval_segm_masks(target_path, pred_path, output_dir=output_dir, pred_class_mapping=class_mapping,
                            target_class_mapping=class_mapping, slow=slow, spacing=spacing,
                            per_bbox=True)
    json_object = json.dumps(results, indent = 4)
    print(json_object)

    tumor_dice = results["class_metrics"]['F1']['Tumor']
    stroma_dice = results["class_metrics"]['F1']['Stroma']
    print('Tumor F1: ', tumor_dice, 'Stroma F1:', stroma_dice)

if __name__ == '__main__':
    _example_one()