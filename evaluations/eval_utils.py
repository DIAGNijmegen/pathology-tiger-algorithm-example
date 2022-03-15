import argparse
from pathlib import Path

import cv2
import numpy as np

from skimage.transform import resize
from skimage import img_as_bool

from matplotlib import pyplot as plt
from tqdm import tqdm
from wholeslidedata.image.wholeslideimage import WholeSlideImage

class ImageReader(WholeSlideImage):
    def __init__(self, image_path, backend='asap'):
        super().__init__(image_path, backend=backend)

    def read(self, spacing, row, col, height, width):
        patch = self.get_patch(col, row, width, height, spacing, center=False, relative=True)
        # return patch.transpose([1,0,2])
        return patch

    def content(self, spacing):
        return self.get_slide(spacing)

    def refine(self, spacing):
        return self.get_real_spacing(spacing)

    def level(self, spacing):
        return self.get_level_from_spacing(spacing)



def is_iterable(obj):
    return hasattr(obj, '__iter__') and not isinstance(obj, str)

def is_callable(obj):
    return hasattr(obj, '__call__')

def is_string(obj):
    return isinstance(obj, str)

def is_int(obj):
    return isinstance(obj, (int, np.integer))

def is_float(obj):
    return isinstance(obj, (float, np.float))

def is_string_or_path(obj):
    return isinstance(obj, (str, Path))

def is_dict(obj):
    return isinstance(obj, dict)

def is_list(obj):
    return isinstance(obj, list)

def is_list_or_tuple(obj):
    return isinstance(obj, (list, tuple))

def is_ndarray(img):
    return isinstance(img, np.ndarray)

def invert_dict(d):
    inverted = dict([[v, k] for k, v in d.items()])
    return inverted

def list_not_in(lst1, lst2):
    """ returns values in lst1 not in lst2 """
    lst3 = [v for v in lst1 if v not in lst2]
    return lst3

def pathes_to_string(pathes, name_only=False, stem_only=False, sort=False):
    if name_only and stem_only:
        raise ValueError('either name_only or stem_only, not both')
    strings = []
    for path in pathes:
        if name_only:
            strings.append(path.name)
        elif stem_only:
            strings.append(path.stem)
        else:
            strings.append(str(path))
    if sort:
        strings = sorted(strings)
    return strings

def pathes_in(d, recursive=False, starting=None, ending=None, containing=[], not_containing=[],
              files_only=True, dirs_only=False, sort=False, sort_by_size=False, as_strings=False, print_progress=False,
              diry_dir_check=False):
    """ returns the contents of the given directory"""
    if files_only and dirs_only:
        raise ValueError('either files_only or dirs_only, not both')
    pathes = []
    if is_string(d):
        d = Path(d)
    if not d.exists():
        return []
    if not_containing is None:
        not_containing = []
    elif is_string(not_containing):
        not_containing = [not_containing]

    if containing is None:
        containing = []
    elif is_string(containing):
        containing = [containing]

    if sort_by_size:
        def _get_size(entry):
            return entry.stat().st_size

        all_files = sorted(d.iterdir(), key=_get_size)
    else:
        all_files = [f for f in d.iterdir()]

    if print_progress:
        from tqdm import tqdm
        pbar = tqdm(total=len(all_files))

    for f in all_files:
        if recursive and f.is_dir():
            files_in_dir = pathes_in(f, recursive=True, starting=starting,
                                     containing=containing, not_containing=not_containing, ending=ending)
            pathes.extend(files_in_dir)
        selected=True
        if files_only and Path(f).is_dir():
            selected = False
        if dirs_only and f.is_file():
            selected = False

        if starting is not None and not f.name.startswith(starting):
            selected = False
        for co in containing:
            if not co.lower() in str(f).lower():
                selected = False
                break
        for nc in not_containing:
            if nc.lower() in str(f).lower():
                selected = False
                break
        if ending is not None and not f.name.endswith(ending):
            selected = False

        if selected:
            pathes.append(f)
        if print_progress:
            pbar.update(1)
    if print_progress:
        pbar.close()
    if sort:
        pathes = sorted(pathes)
    if as_strings:
        pathes = pathes_to_string(pathes)
    return pathes

def get_path_named_like(name, all_pathes, same_name=False, take_shortest=False, as_string=False,
                        replacements={}):
    """ assumes there are no two names where one name contains the other one
     replacements: dict {repl_str:with_str} replaces repl_stri with the with_str for matching the pathes """
    if isinstance(name, Path):
        name = name.stem

    found = []
    for p in all_pathes:
        p = Path(p)
        other = p.stem
        for k,v in replacements.items():
            other = other.replace(k,v)
        if same_name:
            if other == name:
                found.append(p)
        else:
            if other.startswith(name):# or name.startswith(p.stem):
                found.append(p)
    if found is None or len(found)==0:
        return None
    if len(found)>1:
        found.sort(key=lambda s: len(str(s)))
        if take_shortest:
            print('selecting %s to match %s from %d found pathes %s' % (found[0], name, len(found), str([fo.stem for fo in found[1:]])))
        else:
            raise ValueError('too many files found for %s, :%s' % (name, str(found)))
    result = found[0]
    if as_string:
        result = str(result)
    return result

def get_corresponding_pathes(pathes1, pathes2, must_all_match=False, ignore_missing=False,
                             ignore_missing2=True, as_string=False, **kwargs):
    """ finds pathes in pathes2 named like in pathes1 and returns both matching pathes"""
    sel1 = []; sel2 = []
    for p1 in tqdm(pathes1):
        p2 = get_path_named_like(Path(p1).stem, pathes2, as_string=as_string, **kwargs)
        if p2 is None:
            if must_all_match:
                raise ValueError('no match for %s in %s' % (Path(p1).stem, str(pathes2)))
            elif ignore_missing:
                continue
            else:
                pass
                #p2 will be added as None
        if as_string:
            p1 = str(p1)
            p2 = str(p2)
        sel1.append(p1)
        sel2.append(p2)
    if not ignore_missing2:
        s2 = list(set([str(p) for p in sel2]))
        a2 = list(set([str(p) for p in pathes2]))
        m2 = list_not_in(a2, s2)
        if len(s2)!=len(pathes2):
            print('no matches for %d %s' % (len(m2), m2))
    return sel1, sel2

def get_corresponding_pathes_all(pathes1, pathes2, **kwargs):
    _, found = get_corresponding_pathes(pathes1, pathes2, must_all_match=True, **kwargs)
    return found

def get_corresponding_pathes_dirs(dir1, dir2, containing1=None, ending1=None, not_containing1=None,
                                  containing2=None, ending2=None, not_containing2=None,
                                  take_shortest=False, must_all_match=False, allow_dirs=False,
                                  ignore_missing=False, ignore_missing2=True, **kwargs):
    """ dir: directory or list of pathes"""
    type = 'file'
    if allow_dirs:
        type = 'all'
    if is_iterable(dir1):
        pathes1 = dir1
        if ending1 is not None: raise ValueError('no ending if list of pathes1 is given')
    else:
        pathes1 = pathes_in(dir1, containing_or=containing1, ending=ending1,
                                        not_containing=not_containing1, type=type)
    if is_iterable(dir2):
        pathes2 = dir2
        if ending2 is not None: raise ValueError('no ending if list of pathes2 is given')
    else:
        pathes2 = pathes_in(dir2, containing_or=containing2, ending=ending2,
                                        not_containing=not_containing2, type=type)

    return get_corresponding_pathes(pathes1, pathes2, take_shortest=take_shortest,
                                    must_all_match=must_all_match,
                                    ignore_missing=ignore_missing, ignore_missing2=ignore_missing2, **kwargs)

def remap_mask_arr(mask, label_classname_map, new_label_classname_map, check_all_mapped=False):
    """ label_classname_map: nr->name"""
    if check_all_mapped:
        #check that all values in the mask are mapped:
        mask_values = sorted(list(np.unique(mask)))
        if 0 in mask_values: mask_values.remove(0)
        class_labels = sorted(label_classname_map.keys())
        if 0 in class_labels: class_labels.remove(0)
        if not all(elem in class_labels for elem in mask_values):
            raise ValueError('Mask labels and mapping label->class name dont match: %s, %s' % (str(mask_values), str(class_labels)))

    new_classname_label_map = invert_dict(new_label_classname_map)
    label_map = {}
    for label, classname in label_classname_map.items():
        new_label = new_classname_label_map[classname]
        label_map[label] = new_label
    remapped = remap_mask_arr_labels(mask, label_map)


    return remapped

def remap_mask_arr_labels(mask, label_map):
    """ label_mal: nr->nr. Every value in the mask not in the label_map is set to zero"""
    remapped = np.zeros_like(mask)
    for key, value in label_map.items():
        remapped[mask == key] = value
    return remapped


def remap_mask_arr_labels_inplace(mask, label_map):
    """ label_mal: nr->nr. remaps the values inplace - collisions are possible"""
    for key, value in label_map.items():
        mask[mask == key] = value


def plot_cm(cm, classes, normalize=False, title=None, cmap=plt.cm.Blues, save_path=None, show=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True',
           xlabel='Predicted')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if show:
        plt.show(block=True)
    if save_path is not None:
        plt.savefig(str(save_path), bbox_inches='tight')
    return fig, ax