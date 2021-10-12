import json
from abc import abstractmethod, ABC
from enum import Enum, auto
from typing import Tuple
import numpy as np
from skimage.transform import resize
from matplotlib import pyplot as plt

from BaseDataset import BaseDataset, JSONDatasetStructure


class BatchGeneratorStrategy(Enum):
    RESIZED_IMAGES = auto()
    TILED_IMAGES = auto()
    FULL_IMAGES = auto()


class BatchGeneratorConfig:
    """Describes the possible configurations of the batch generator.

    Can be loaded from a json file."""

    def __init__(self, json_file: str):
        with open(json_file, "r") as fp:
            d = json.load(fp)
            self.loaded = True
            try:
                self.batch_size = d['batch_size']
                self.image_size = tuple(d['image_size']) if 'image_size' in d else None
                self.image_channels = d['image_channels'] if 'image_channels' in d else 3
                self.anno_channels = d['anno_channels'] if 'anno_channels' in d else 2
                self.batch_strategy = BatchGeneratorStrategy[d['batch_strategy']]
                self.epochs = d['epochs']
            except KeyError as e:
                self.loaded = False
                print(f"Couldn't load dataset structure: {e}")


class TilingHandler:
    @staticmethod
    def random_tiles(im: np.array, anno: np.array, batch_size: int, tile_size: Tuple[int, int]) \
            -> Tuple[np.array, np.array]:
        coords = np.random.random((batch_size, 2))
        coords[:, 0] *= im.shape[0]-tile_size[0]
        coords[:, 1] *= im.shape[1]-tile_size[1]
        coords = coords.astype('int')

        batch_x = np.array([im[coords[i, 0]:coords[i, 0]+tile_size[0], coords[i, 1]:coords[i, 1]+tile_size[1]]
                            for i in range(batch_size)])
        batch_y = np.array([anno[coords[i, 0]:coords[i, 0]+tile_size[0], coords[i, 1]:coords[i, 1]+tile_size[1]]
                            for i in range(batch_size)])
        if len(batch_y.shape) == 3:
            batch_y = batch_y[..., np.newaxis]

        return batch_x, batch_y


class BatchGenerator(ABC):
    """Generates batches of image/annotations using one of the generation strategies:

    * Resized images (select n next images & resize)
    * Tiled images (select 1 image and sample n tiles)
    * ?other"""
    def __init__(self, dataset: BaseDataset, config: BatchGeneratorConfig, is_train=True):
        self.dataset = dataset
        self.config = config

        self.n = self.dataset.getsize(is_train)
        self.batches_in_epoch = self.get_batches_in_epoch()

    @abstractmethod
    def get_batches_in_epoch(self) -> int:
        pass

    @abstractmethod
    def next_batch(self):
        pass


class TiledImageBatchGenerator(BatchGenerator):
    def get_batches_in_epoch(self) -> int:
        return self.n

    def next_batch(self):
        """Next batch generator"""
        for e in range(self.config.epochs):
            gen = self.dataset.randomGenerator()
            for im, anno in gen:
                batch_x, batch_y = TilingHandler.random_tiles(
                    im, anno, self.config.batch_size, self.config.image_size
                )
                yield batch_x, batch_y


class ResizedImageBatchGenerator(BatchGenerator):
    def get_batches_in_epoch(self) -> int:
        return self.n // self.config.batch_size

    def next_batch(self):
        """Next batch generator"""
        batch_x = np.zeros((self.config.batch_size,) + self.config.image_size + (self.config.image_channels,))
        batch_y = np.zeros((self.config.batch_size,) + self.config.image_size + (self.config.anno_channels,))
        for e in range(self.config.epochs):
            gen = self.dataset.randomGenerator()
            id_in_batch = 0
            for im, anno in gen:
                batch_x[id_in_batch] = resize(
                    im, self.config.image_size, order=1, preserve_range=False
                )
                batch_y[id_in_batch] = resize(
                    anno, self.config.image_size, order=0, preserve_range=True, anti_aliasing=False
                )
                id_in_batch += 1
                if id_in_batch >= self.config.batch_size:
                    id_in_batch = 0
                    yield batch_x, batch_y


class FullImageBatchGenerator(BatchGenerator):
    def get_batches_in_epoch(self) -> int:
        return self.n // self.config.batch_size

    def next_batch(self):
        """Next batch generator"""
        for e in range(self.config.epochs):
            gen = self.dataset.randomGenerator()
            batch_x = []
            batch_y = []
            for im, anno in gen:
                if len(anno.shape) == 2:
                    anno = anno[..., np.newaxis]
                batch_x.append(im)
                batch_y.append(anno)
                if len(batch_x) >= self.config.batch_size:
                    yield np.array(batch_x), np.array(batch_y)
                    batch_x = []
                    batch_y = []


class BatchGeneratorFactory:
    batchGenerators = {
        BatchGeneratorStrategy.TILED_IMAGES: TiledImageBatchGenerator,
        BatchGeneratorStrategy.RESIZED_IMAGES: ResizedImageBatchGenerator,
        BatchGeneratorStrategy.FULL_IMAGES: FullImageBatchGenerator
    }

    @classmethod
    def get_batch_generator(cls, dataset: BaseDataset, config: BatchGeneratorConfig, is_train=True) -> BatchGenerator:
        if config.batch_strategy not in cls.batchGenerators:
            raise NotImplementedError(f"Strategy not implemented: {config.batch_strategy}")

        return cls.batchGenerators[config.batch_strategy](dataset, config, is_train)


def test_batch_generator():
    dstruct = JSONDatasetStructure("./dataset_structure_definitions/glas_original.json")
    print(dstruct)
    dataset = BaseDataset(dstruct)
    dataset.preload()
    config = BatchGeneratorConfig("./batch_generator_config/basic_tile_config.json")
    batch_generator = BatchGeneratorFactory.get_batch_generator(dataset, config)
    batch_x = batch_y = None

    for idb, (batch_x, batch_y) in enumerate(batch_generator.next_batch()):
        print(f"Epoch {(idb//batch_generator.batches_in_epoch)+1} - Batch {(idb%batch_generator.batches_in_epoch)+1}")

    plt.figure()
    for i in range(config.batch_size):
        plt.subplot(1, config.batch_size, i+1)
        plt.imshow(batch_x[i])
        if len(np.unique(batch_y[i, ..., 0])) > 1:
            plt.contour(batch_y[i, ..., 0])

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(batch_y[0, ..., 0], interpolation='none')
    plt.subplot(1, 2, 2)
    plt.imshow(batch_y[0, ..., 1], interpolation='none')
    plt.show()


def test_batch_generator_from_pkl():
    dataset = BaseDataset.loadFrom("e:/data/GlaS/glas_original.pkl")
    print(dataset.dstruct)

    config = BatchGeneratorConfig("./batch_generator_config/basic_resize_config.json")
    batch_generator = BatchGeneratorFactory.get_batch_generator(dataset, config)
    batch_x = batch_y = None

    for idb, (batch_x, batch_y) in enumerate(batch_generator.next_batch()):
        print(f"Epoch {(idb//batch_generator.batches_in_epoch)+1} - Batch {(idb%batch_generator.batches_in_epoch)+1}")

    plt.figure()
    for i in range(config.batch_size):
        plt.subplot(1, config.batch_size, i + 1)
        plt.imshow(batch_x[i])
        if len(np.unique(batch_y[i, ..., 0])) > 1:
            plt.contour(batch_y[i, ..., 0])

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(batch_y[0, ..., 0], interpolation='none')
    plt.subplot(1, 2, 2)
    plt.imshow(batch_y[0, ..., 1], interpolation='none')
    plt.show()


if __name__ == "__main__":
    test_batch_generator_from_pkl()
