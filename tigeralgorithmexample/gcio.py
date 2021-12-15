""" GrandChallenge Input/Output (gcio)

In this file settings concerning folders and paths for reading and writing on GrandChallenge are defined.
Note that these settings are moslty specific to the GrandChallenge Tiger Challenge.

"""


from pathlib import Path
from shutil import copy


# Grand Challenge paths
GRAND_CHALLENGE_SEGMENTATION_OUTPUT_PATH = Path(
    "/output/images/breast-cancer-segmentation-for-tils/segmentation.tif"
)
GRAND_CHALLENGE_DETECTION_OUTPUT_PATH = Path("/output/detected-lymphocytes.json")
GRAND_CHALLENGE_TILS_SCORE_PATH = Path("/output/til-score.json")

### Temporary paths
TMP_FOLDER = Path("/home/user/tmp")
TMP_SEGMENTATION_OUTPUT_PATH = (
    TMP_FOLDER / GRAND_CHALLENGE_SEGMENTATION_OUTPUT_PATH.name
)
TMP_DETECTION_OUTPUT_PATH = TMP_FOLDER / GRAND_CHALLENGE_DETECTION_OUTPUT_PATH.name
TMP_TILS_SCORE_PATH = TMP_FOLDER / GRAND_CHALLENGE_TILS_SCORE_PATH.name


def initialize_output_folders():
    """This function initialize all folders for (mandatory) grandchallgenge output as well ass tempory folder"""

    GRAND_CHALLENGE_SEGMENTATION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAND_CHALLENGE_DETECTION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAND_CHALLENGE_TILS_SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TMP_FOLDER.mkdir(parents=True, exist_ok=True)


def get_image_path_from_input_folder() -> Path:
    """Gets a test image which needs to be processed for the Tiger Challenge

    Returns:
        Path: path to multiresolution image from the test set
    """

    return list(Path("/input/").glob("*.tif"))[0]


def get_tissue_mask_path_from_input_folder() -> Path:
    """Gets the tissue mask for the corresponding image that needs to be processed

    Returns:
        Path: path to tissue tissue mask
    """

    return list(Path("/input/images/").glob("*.tif"))[0]


def copy_data_to_output_folders():
    """Copies all temporary files to the (mandatory) output files/folders"""

    # copy segmentation tif to grand challenge
    copy(TMP_SEGMENTATION_OUTPUT_PATH, GRAND_CHALLENGE_SEGMENTATION_OUTPUT_PATH)
    # copy detections json to grand challenge
    copy(TMP_DETECTION_OUTPUT_PATH, GRAND_CHALLENGE_DETECTION_OUTPUT_PATH)
    # copy tils score json to grand challenge
    copy(TMP_TILS_SCORE_PATH, GRAND_CHALLENGE_TILS_SCORE_PATH)
