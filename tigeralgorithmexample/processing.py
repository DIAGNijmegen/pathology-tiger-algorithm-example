import time
from functools import wraps
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from .gcio import (TMP_DETECTION_OUTPUT_PATH, TMP_SEGMENTATION_OUTPUT_PATH,
                   TMP_TILS_SCORE_PATH, copy_data_to_output_folders,
                   get_image_path_from_input_folder,
                   get_tissue_mask_path_from_input_folder,
                   initialize_output_folders)
from .rw import (READING_LEVEL, WRITING_TILE_SIZE, DetectionWriter,
                 SegmentationWriter, TilsScoreWriter,
                 open_multiresolutionimage_image)


# https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap


@timing
def process_image_tile_to_segmentation(
    image_tile: np.ndarray, tissue_mask_tile: np.ndarray
) -> np.ndarray:
    """Example function that shows processing a tile from a multiresolution image for segmentation purposes.

    NOTE
        This code is only made for illustration and is not meant to be taken as valid processing step.

    Args:
        image_tile (np.ndarray): [description]
        tissue_mask_tile (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """

    prediction = np.copy(image_tile[:, :, 0])
    prediction[image_tile[:, :, 0] > 90] = 1
    prediction[image_tile[:, :, 0] <= 90] = 2
    return prediction * tissue_mask_tile


@timing
def process_image_tile_to_detections(
    image_tile: np.ndarray,
    segmentation_mask: np.ndarray,
) -> List[tuple]:
    """Example function that shows processing a tile from a multiresolution image for detection purposes.

    NOTE
        This code is only made for illustration and is not meant to be taken as valid processing step. Please update this function

    Args:
        image_tile (np.ndarray): [description]
        tissue_mask_tile (np.ndarray): [description]

    Returns:
        List[tuple]: list of tuples (x,y) coordinates of detections
    """
    if not np.any(segmentation_mask == 2):
        return []

    prediction = np.copy(image_tile[:, :, 2])
    prediction[(image_tile[:, :, 2] <= 40) & (segmentation_mask == 2)] = 1
    xs, ys = np.where(prediction.transpose() == 1)
    probabilities = [1.0] * len(xs)
    return list(zip(xs, ys, probabilities))


@timing
def process_segmentation_detection_to_tils_score(
    segmentation_path: Path, detections: List[tuple]
) -> int:
    """Example function that shows processing a segmentation mask and corresponding detection for the computation of a tls score.

    NOTE
        This code is only made for illustration and is not meant to be taken as valid processing step.

    Args:
        segmentation_mask (np.ndarray): [description]
        detections (List[tuple]): [description]

    Returns:
        int: til score (between 0, 100)
    """

    level = 4
    cell_area_level_1 = 16 * 16

    image = open_multiresolutionimage_image(path=segmentation_path)
    width, height = image.getDimensions()
    slide_at_level_4 = image.getUCharPatch(
        0, 0, int(width / 2 ** level), int(height / 2 ** level), level
    )
    area = len(np.where(slide_at_level_4 == 2)[0])
    cell_area = cell_area_level_1 // 2 ** 4
    n_detections = len(detections)
    if cell_area == 0 or n_detections == 0:
        return 0
    value = min(100, int(area / (n_detections / cell_area)))
    return value


def process():
    """Proceses a test slide"""

    level = READING_LEVEL
    tile_size = WRITING_TILE_SIZE  # should be a power of 2

    initialize_output_folders()

    # get input paths
    image_path = get_image_path_from_input_folder()
    tissue_mask_path = get_tissue_mask_path_from_input_folder()

    print(f"Processing image: {image_path}")
    print(f"Processing with mask: {tissue_mask_path}")

    # open images
    image = open_multiresolutionimage_image(path=image_path)
    tissue_mask = open_multiresolutionimage_image(path=tissue_mask_path)

    # get image info
    dimensions = image.getDimensions()
    spacing = image.getSpacing()

    # create writers
    print(f"Setting up writers")
    segmentation_writer = SegmentationWriter(
        TMP_SEGMENTATION_OUTPUT_PATH,
        tile_size=tile_size,
        dimensions=dimensions,
        spacing=spacing,
    )
    detection_writer = DetectionWriter(TMP_DETECTION_OUTPUT_PATH)
    tils_score_writer = TilsScoreWriter(TMP_TILS_SCORE_PATH)

    print("Processing image...")
    # loop over image and get tiles
    for y in tqdm(range(0, dimensions[1], tile_size)):
        for x in range(0, dimensions[0], tile_size):
            tissue_mask_tile = tissue_mask.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
            ).squeeze()

            if not np.any(tissue_mask_tile):
                continue

            image_tile = image.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
            )

            # segmentation
            segmentation_mask = process_image_tile_to_segmentation(
                image_tile=image_tile, tissue_mask_tile=tissue_mask_tile
            )
            segmentation_writer.write_segmentation(tile=segmentation_mask, x=x, y=y)

            # detection
            detections = process_image_tile_to_detections(
                image_tile=image_tile, segmentation_mask=segmentation_mask
            )
            detection_writer.write_detections(
                detections=detections, spacing=spacing, x_offset=x, y_offset=y
            )

    print("Saving...")
    # save segmentation and detection
    segmentation_writer.save()
    detection_writer.save()

    print("Number of detections", len(detection_writer.detections))

    print("Compute tils score...")
    # compute tils score
    tils_score = process_segmentation_detection_to_tils_score(
        TMP_SEGMENTATION_OUTPUT_PATH, detection_writer.detections
    )
    tils_score_writer.set_tils_score(tils_score=tils_score)

    print("Saving...")
    # save tils score
    tils_score_writer.save()

    print("Copy data...")
    # save data to output folder
    copy_data_to_output_folders()

    print("Completed!")
