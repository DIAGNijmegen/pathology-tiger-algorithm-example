import numpy as np
from hooknet.configuration.config import create_hooknet
from hooknet.model import HookNet
from tigeralgorithmexample.gcio import (
    TMP_SEGMENTATION_OUTPUT_PATH,
    copy_data_to_output_folders,
    get_image_path_from_input_folder,
    get_tissue_mask_path_from_input_folder,
    initialize_output_folders,
)


from tigeralgorithmexample.rw import (
    WRITING_TILE_SIZE,  # set to 1024
    SegmentationWriter,
    open_multiresolutionimage_image,
)
from tqdm import tqdm

HOOKNET_CONFIG_FILE = "/home/user/pathology-tiger-baseline/configs/seg-inference-config/hooknet_params.yml"


def crop_to_tile_size(data: np.ndarray):
    if (data.shape[0] == WRITING_TILE_SIZE) and (data.shape[1] == WRITING_TILE_SIZE):
        return data

    cropx = (data.shape[1] - WRITING_TILE_SIZE) // 2
    cropy = (data.shape[0] - WRITING_TILE_SIZE) // 2

    if len(data.shape) == 2:
        return data[cropy:-cropy, cropx:-cropx]
    if len(data.shape) == 3:
        return data[cropy:-cropy, cropx:-cropx, :]


def process_image_tile_to_segmentation(
    hooknet: HookNet, target_image_tile: np.ndarray, context_image_tile: np.ndarray,  tissue_mask_tile: np.ndarray
) -> np.ndarray:
    
    # create batch 
    hooknet_batch = [np.array([target_image_tile]), np.array([context_image_tile])]
    
    # predict on batch
    prediction = hooknet.predict_on_batch(hooknet_batch)[0]
    
    # crop prediction to writing tile size
    prediction = crop_to_tile_size(prediction)
    
    # crop mask to writing tile size
    tissue_mask_tile = crop_to_tile_size(tissue_mask_tile)
    
    return prediction * tissue_mask_tile


def process():
    """Proceses a test slide"""

    target_level = 0
    context_level = 4

    # init hooknet model
    hooknet = create_hooknet(
        user_config=HOOKNET_CONFIG_FILE,
        mode="default",
    )

    # init output folders
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
        tile_size=WRITING_TILE_SIZE,
        dimensions=dimensions,
        spacing=spacing,
    )

    # get downsampling context
    downsample = image.getLevelDownsample(context_level)

    # get input shape for hooknet
    input_width = int(hooknet.input_shape[1])
    input_height = int(hooknet.input_shape[0])

    # compute offset  due to the difference between the shape of the model's input and the writing tile size
    x_offset = int((input_width - WRITING_TILE_SIZE) // 2)
    y_offset = int((input_height - WRITING_TILE_SIZE) // 2)

    print("Processing image...")
    # loop over image and get tiles
    for y in tqdm(range(0, dimensions[1], WRITING_TILE_SIZE)):
        for x in range(0, dimensions[0], WRITING_TILE_SIZE):

            # get coordinates for target tile
            x_target = x - x_offset
            y_target = y - y_offset

            # get coordinates for context tile
            x_context = int(
                (x_target + input_width // 2) - downsample * (input_width // 2)
            )
            y_context = int(
                (y_target + input_height // 2) - downsample * (input_height // 2)
            )

            tissue_mask_tile = tissue_mask.getUCharPatch(
                startX=x_target,
                startY=y_target,
                width=input_width,
                height=input_height,
                level=target_level,
            ).squeeze()

            if not np.any(tissue_mask_tile):
                continue

            # get target tile
            target_image_tile = image.getUCharPatch(
                startX=x_target,
                startY=y_target,
                width=input_width,
                height=input_height,
                level=target_level,
            )

            # get context tile
            context_image_tile = image.getUCharPatch(
                startX=x_context,
                startY=y_context,
                width=input_width,
                height=input_height,
                level=context_level,
            )

            # segmentation
            segmentation_mask = process_image_tile_to_segmentation(
                hooknet=hooknet,
                target_image_tile=target_image_tile,
                context_image_tile=context_image_tile,
                tissue_mask_tile=tissue_mask_tile,
            )
            segmentation_writer.write_segmentation(tile=segmentation_mask, x=x, y=y)

    print("Saving...")
    # save segmentation and detection
    segmentation_writer.save()

    print("Copy data...")
    # save data to output folder
    copy_data_to_output_folders()

    print("Completed!")
