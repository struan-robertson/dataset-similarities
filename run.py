"""Calculate how similar each image in the dataset is and then graph."""

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from shoeprint_image_retrieval.config import load_config
from shoeprint_image_retrieval.network import Model
from tqdm import tqdm

from comparison import compare

config = load_config("sims.toml")

# Calculated to be the best _on the current dataset_
# TODO change if dataset changes
scale = 0.78125
block = 6

dataset_dir = Path(config["dataset"]["dir"])

image_paths: list[Path] = [
    image_path for image_path in dataset_dir.rglob("*") if image_path.is_file()
]

pairs = list(combinations(image_paths, 2))

feature_maps: list[NDArray[Any]] = []

model = Model(config, block)

print("Loading image feature maps")
for image_path in tqdm(image_paths):
    image = Image.open(image_path)

    scaled_height = int(image.height * scale)
    scaled_width = int(image.width * scale)
xg
    image = image.resize(
        (scaled_width, scaled_height),
        Image.Resampling.LANCZOS,
    )

    image_arr = np.array(image)

    feature_map = model.get_feature_maps(image_arr)

    feature_maps.append(feature_map)

similarity_array = compare(config, feature_maps)

# Load all images into shared array
# I need a 2D array for storing similarities
# The index of feature_maps, image_paths and 2D similarity array will correspond
