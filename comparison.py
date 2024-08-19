"""Perform similarity comparisons between images in a dataset."""

from __future__ import annotations

import time
from multiprocessing import Array, Process, Value
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized, SynchronizedArray

    from shoeprint_image_retrieval.config import Config


import numpy as np
from more_itertools import chunked
from numpy import float32
from numpy.typing import NDArray
from shoeprint_image_retrieval.similarity import get_similarity
from tqdm import tqdm


def compare(config: Config, feature_maps: list[NDArray[Any]]) -> NDArray[Any]:
    """Given a list of feature maps, compare them against each other and return the values."""
    n_maps = len(feature_maps)
    n_processes = config["comparison"]["n_processes"]

    all_indexes = list(range(n_maps))

    chunked_indexes: list[list[int]] = list(chunked(all_indexes, n_processes))

    shared_images: list[tuple[SynchronizedArray[Any], tuple[int, int, int]]] = []
    for i in range(n_maps):
        shape = cast(tuple[int, int, int], feature_maps[i].shape)

        shared = Array("f", feature_maps[i].size)
        np_shared: NDArray[Any] = cast(
            NDArray[Any],
            np.frombuffer(shared.get_obj(), dtype=np.float32).reshape(shape),  # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportArgumentType]
        )
        np.copyto(np_shared, feature_maps[i])

        shared_images.append((shared, shape))

    shared_sims_array: tuple[SynchronizedArray[Any], tuple[int, int]] = (
        Array("f", n_maps * n_maps),
        (n_maps, n_maps),
    )
    np_sims_array: NDArray[np.float32] = cast(
        NDArray[np.float32],
        np.frombuffer(shared_sims_array[0].get_obj(), dtype=np.float32).reshape((n_maps, n_maps)),  # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportArgumentType]
    )

    np.copyto(np_sims_array, np.zeros((n_maps, n_maps), np.float32))

    counter: Synchronized[int] = Value("i", 0)

    processes: list[Process] = []
    for i in range(n_processes):
        p = Process(
            target=comparison_worker,
            args=(
                shared_images,
                chunked_indexes[i],
                shared_sims_array,
                counter,
            ),
        )
        processes.append(p)
        p.start()

    work = n_maps**2
    with tqdm(total=work) as pbar:
        while counter.value < work:
            _ = pbar.update(counter.value - cast(int, pbar.n))
            pbar.refresh()

            time.sleep(1)

        _ = pbar.update(counter.value - cast(int, pbar.n))
        pbar.refresh()

    for p in processes:
        p.join()

    return np_sims_array


def comparison_worker(
    shared_images: list[tuple[SynchronizedArray[Any], tuple[int, int, int]]],
    image_indexes: list[int],
    shared_similarities: tuple[SynchronizedArray[float32], tuple[int, int]],
    counter: Synchronized[int],
):
    """Worker for multithreaded feature map comparison."""
    images: list[NDArray[Any]] = [np.empty(0)] * len(shared_images)

    # Convert shared image arrays into numpy arrays
    for i in range(len(shared_images)):
        images[i] = np.frombuffer(  # pyright: ignore[reportCallIssue, reportUnknownMemberType]
            shared_images[i][0].get_obj(),  # pyright: ignore[reportArgumentType]
            dtype=np.float32,
        ).reshape(shared_images[i][1])

    # Convert shared similarity array into numpy array
    similarities: NDArray[np.float32] = np.frombuffer(  # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportUnknownVariableType]
        shared_similarities[0].get_obj(),  # pyright: ignore[reportArgumentType]
        dtype=np.float32,
    ).reshape(shared_similarities[1])

    for query_index in image_indexes:
        query_image = images[query_index]

        for gallery_index, gallery_image in enumerate(images):
            if query_index == gallery_index:
                similarities[query_index, gallery_index] = 1.0
                with counter.get_lock():
                    counter.value += 1

                continue

            similarity = get_similarity(query_image, gallery_image)

            similarities[query_index, gallery_index] = similarity

            with counter.get_lock():
                counter.value += 1
