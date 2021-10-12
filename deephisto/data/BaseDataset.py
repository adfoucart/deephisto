import json
import os
import random
import pickle
from abc import ABC
from enum import Enum, auto
from skimage.io import imread
from skimage.transform import resize
from typing import Generator
import numpy as np


class InvalidDatasetError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class AnnotationStrategy(Enum):
    INSTANCE = auto(),
    CLASS = auto(),
    KEEP = auto()


class DatasetStructure(ABC):
    """Describes the structure of the files of a dataset.

    Can be loaded from json:
    * dataset_dir -> the base directory
    * train_images_dir -> directory where training images can be found
    * train_images_list -> list of all images in the training set
    * train_annotations_dir -> directory where training annotations can be found
    * train_annotations_list -> list of all annotations in the training set
    * test_images_dir -> directory where test images can be found
    * test_images_list -> list of all images in the test set
    * test_annotations_dir -> directory where testing annotations can be found
    * test_annotations_list -> list of all annotations in the training set
    """

    accepted_keys = [
        'dataset_dir',
        'train_images_dir',
        'train_images_list',
        'train_annotations_dir',
        'train_annotations_list',
        'test_images_dir',
        'test_images_list',
        'test_annotations_dir',
        'test_annotations_list',
        'pre_resize',
        'annotation_strategy',
        'n_classes'
    ]

    def __init__(self):
        self.loaded = False
        self.d = None
        self.dataset_dir = None
        self.train_images_dir = None
        self.train_images_list = None
        self.train_annotations_dir = None
        self.train_annotations_list = None
        self.test_images_dir = None
        self.test_images_list = None
        self.test_annotations_dir = None
        self.test_annotations_list = None
        self.pre_resize = None
        self.annotation_strategy = None
        self.n_classes = None

    def __str__(self):
        output = f"Dataset in: {self.dataset_dir}\n" + \
                 f"Training images: {self.train_images_dir} ({len(self.train_images_list)} images)\n" + \
                 f"Test images: {self.test_images_dir} ({len(self.test_images_list)} images)\n"
        if (len(self.train_images_list) != len(self.train_annotations_list)) or \
                (len(self.test_images_list) != len(self.test_annotations_list)):
            output += f"Mismatch in image/annotations list -> \n" + \
                      f"Train: {len(self.train_images_list)} vs {len(self.train_annotations_list)}\n" + \
                      f"Test: {len(self.test_images_list)} vs {len(self.test_annotations_list)}\n"
        if self.pre_resize is not None:
            output += f"Images will be resized to {self.pre_resize}\n"
        output += f"Annotation strategy: {self.annotation_strategy.name}"
        return output


class JSONDatasetStructure(DatasetStructure):
    def __init__(self, json_file: str):
        super().__init__()

        with open(json_file, "r") as fp:
            self.d = json.load(fp)
            self.loaded = True
            try:
                self.dataset_dir = self.d['dataset_dir']
                self.train_images_dir = self.d['train_images_dir']
                self.train_images_list = self.d['train_images_list']
                self.train_annotations_dir = self.d['train_annotations_dir']
                self.train_annotations_list = self.d['train_annotations_list']
                self.test_images_dir = self.d['test_images_dir']
                self.test_images_list = self.d['test_images_list']
                self.test_annotations_dir = self.d['test_annotations_dir']
                self.test_annotations_list = self.d['test_annotations_list']
                if 'pre_resize' not in self.d or self.d['pre_resize'] is None:
                    self.pre_resize = None
                else:
                    self.pre_resize = tuple(self.d['pre_resize'])
                self.annotation_strategy = AnnotationStrategy[self.d['annotation_strategy']] \
                    if 'annotation_strategy' in self.d \
                    else AnnotationStrategy.KEEP
                self.n_classes = self.d['n_classes'] if 'n_classes' in self.d else None
            except KeyError as e:
                self.loaded = False
                print(f"Couldn't load dataset structure: {e}")

        if (len(self.train_images_list) != len(self.train_annotations_list)) or \
                (len(self.test_images_list) != len(self.test_annotations_list)):
            raise InvalidDatasetError("Images and annotations lists should always have the same size. Sizes are:"
                                      f"Training: {len(self.train_images_list)} vs {len(self.train_annotations_list)}"
                                      f"Test: {len(self.test_images_list)} vs {len(self.test_annotations_list)}")


class DictDatasetStructure(DatasetStructure):
    def __init__(self, d: dict):
        super().__init__()

        self.d = d
        self.loaded = True

        try:
            self.dataset_dir = self.d['dataset_dir']
            self.train_images_dir = self.d['train_images_dir']
            self.train_images_list = self.d['train_images_list']
            self.train_annotations_dir = self.d['train_annotations_dir']
            self.train_annotations_list = self.d['train_annotations_list']
            self.test_images_dir = self.d['test_images_dir']
            self.test_images_list = self.d['test_images_list']
            self.test_annotations_dir = self.d['test_annotations_dir']
            self.test_annotations_list = self.d['test_annotations_list']
            if 'pre_resize' not in self.d or self.d['pre_resize'] is None:
                self.pre_resize = None
            else:
                self.pre_resize = tuple(self.d['pre_resize'])
            self.annotation_strategy = AnnotationStrategy[self.d['annotation_strategy']] \
                if 'annotation_strategy' in self.d \
                else AnnotationStrategy.KEEP
            self.n_classes = self.d['n_classes'] if 'n_classes' in self.d else None
        except KeyError as e:
            self.loaded = False
            print(f"Couldn't load dataset structure: {e}")

        if (len(self.train_images_list) != len(self.train_annotations_list)) or \
                (len(self.test_images_list) != len(self.test_annotations_list)):
            raise InvalidDatasetError("Images and annotations lists should always have the same size. Sizes are:"
                                      f"Training: {len(self.train_images_list)} vs {len(self.train_annotations_list)}"
                                      f"Test: {len(self.test_images_list)} vs {len(self.test_annotations_list)}")


class BaseDataset:
    """Base dataset handler.

    Datasets should be accompanied with a json file describing their structure.
    """

    def __init__(self, dstruct: DatasetStructure):
        self.dstruct = dstruct
        self.train_images = []
        self.train_annos = []
        self.n_train = len(dstruct.train_images_list) if dstruct is not None else 0
        self.test_images = []
        self.test_annos = []
        self.n_test = len(dstruct.test_images_list) if dstruct is not None else 0
        self.loaded = False

    def preload(self):
        """Pre-loads the images and annotations into a list."""
        print(f"Pre-loading dataset. [....]", end="\r")
        train_x_dir = os.path.join(self.dstruct.dataset_dir, self.dstruct.train_images_dir)
        self.train_images = [self._imload(os.path.join(train_x_dir, fname), False)
                             for fname in self.dstruct.train_images_list]
        print(f"Pre-loading dataset. [-...]", end="\r")
        train_y_dir = os.path.join(self.dstruct.dataset_dir, self.dstruct.train_annotations_dir)
        self.train_annos = [self._imload(os.path.join(train_y_dir, fname), True)
                            for fname in self.dstruct.train_annotations_list]

        print(f"Pre-loading dataset. [--..]", end="\r")
        test_x_dir = os.path.join(self.dstruct.dataset_dir, self.dstruct.test_images_dir)
        self.test_images = [self._imload(os.path.join(test_x_dir, fname), False)
                            for fname in self.dstruct.test_images_list]
        print(f"Pre-loading dataset. [---.]", end="\r")
        test_y_dir = os.path.join(self.dstruct.dataset_dir, self.dstruct.test_annotations_dir)
        self.test_annos = [self._imload(os.path.join(test_y_dir, fname), True)
                           for fname in self.dstruct.test_annotations_list]
        self.loaded = True
        print(f"Pre-loading dataset. [Done]")

    def getsize(self, is_train=True) -> int:
        if is_train:
            return self.n_train
        return self.n_test

    def _imload(self, path: str, is_anno: bool) -> np.array:
        """Loads an image with optional pre-resizing"""
        return self._pre_resize(imread(path), is_anno)

    def _annotation_preset(self, im: np.array) -> np.array:
        """Modifies the annotation depending on the type.
        If annotations are encoded as instances -> binarize (2 channels).
        If annotations are encoded as classes -> separate into channels
        If annotation strategy is "keep" -> don't change anything."""
        if self.dstruct.annotation_strategy is AnnotationStrategy.KEEP:
            return im
        if self.dstruct.annotation_strategy is AnnotationStrategy.INSTANCE:
            im_ = np.zeros(im.shape[:2] + (2,)).astype('int')
            im_[..., 0] = im > 0
            im_[..., 1] = im == 0
            return im_
        if self.dstruct.annotation_strategy is AnnotationStrategy.CLASS:
            im_ = np.zeros(im.shape[:2] + (self.dstruct.n_classes,))
            for i in range(1, self.dstruct.n_classes + 1):
                im_[..., i] = im == i
            return im_

    def _pre_resize(self, im: np.array, is_anno: bool) -> np.array:
        """Resizes an image if it's required in the config. If annotation, make sure to preserve the values."""
        if is_anno:
            im = self._annotation_preset(im)

        if self.dstruct.pre_resize is None:
            return im

        if is_anno:
            return resize(im, self.dstruct.pre_resize, order=0, preserve_range=True, anti_aliasing=False)

        return resize(im, self.dstruct.pre_resize, order=3, preserve_range=False, anti_aliasing=True)

    def _preloaded_generator(self, is_train=True, order=None) -> Generator:
        if is_train:
            list_images = [im for im in self.train_images] if order is None \
                else [self.train_images[idx] for idx in order]
            list_annos = [im for im in self.train_annos] if order is None \
                else [self.train_annos[idx] for idx in order]
        else:
            list_images = [im for im in self.test_images] if order is None \
                else [self.test_images[idx] for idx in order]
            list_annos = [im for im in self.test_annos] if order is None \
                else [self.test_annos[idx] for idx in order]

        for im, anno in zip(list_images, list_annos):
            yield im, anno

    def _hotloaded_generator(self, is_train=True, order=None) -> Generator:
        if is_train:
            list_images = [im for im in self.dstruct.train_images_list] if order is None \
                else [self.dstruct.train_images_list[idx] for idx in order]
            list_annos = [im for im in self.dstruct.train_annotations_list] if order is None \
                else [self.dstruct.train_annotations_list[idx] for idx in order]

            x_dir = os.path.join(self.dstruct.dataset_dir, self.dstruct.train_images_dir)
            y_dir = os.path.join(self.dstruct.dataset_dir, self.dstruct.train_annotations_dir)
        else:
            list_images = [im for im in self.dstruct.test_images_list] if order is None \
                else [self.dstruct.test_images_list[idx] for idx in order]
            list_annos = [im for im in self.dstruct.test_annotations_list] if order is None \
                else [self.dstruct.test_annotations_list[idx] for idx in order]

            x_dir = os.path.join(self.dstruct.dataset_dir, self.dstruct.test_images_dir)
            y_dir = os.path.join(self.dstruct.dataset_dir, self.dstruct.test_annotations_dir)

        for im, anno in zip(list_images, list_annos):
            yield self._imload(os.path.join(x_dir, im), False), self._imload(os.path.join(y_dir, anno), True)

    def simpleGenerator(self, is_train=True) -> Generator:
        """Generates tuples of im, annotations sequentially in the training or test set."""

        if self.loaded:
            return self._preloaded_generator(is_train)
        return self._hotloaded_generator(is_train)

    def randomGenerator(self, seed=None, is_train=True) -> Generator:
        """Generates tuples of im, annotations in random order in the training or test set."""
        idxs = list(range(self.getsize(is_train)))

        if seed is not None:
            random.seed(seed)

        random.shuffle(idxs)

        if self.loaded:
            return self._preloaded_generator(is_train, order=idxs)
        return self._hotloaded_generator(is_train, order=idxs)

    def sequenceGenerator(self, sequence: list, is_train=True) -> Generator:
        """Generates tuples of im, annotations in the prespecified order."""
        # check that sequence is valid:
        if len(sequence) != self.getsize(is_train):
            raise InvalidDatasetError("Invalid sequence provided.")

        if self.loaded:
            return self._preloaded_generator(is_train, order=sequence)
        return self._hotloaded_generator(is_train, order=sequence)

    def saveTo(self, f: str):
        """Save all the pre-loaded images and annotations to a pickle file, alongside the datastruct info"""
        if not self.loaded:
            raise ValueError("Cannot save dataset that hasn't been pre-loaded.")

        toSave = {}
        for key, value in self.dstruct.d.items():
            toSave[key] = value

        toSave['train_images'] = self.train_images
        toSave['train_annos'] = self.train_annos
        toSave['test_images'] = self.test_images
        toSave['test_annos'] = self.test_annos

        with open(f, "wb") as fp:
            pickle.dump(toSave, fp)

    @staticmethod
    def loadFrom(f: str):
        """Loads all pre-loaded images and annotations from a pickle file"""
        with open(f, "rb") as fp:
            toLoad = pickle.load(fp)
            d = {}
            for key in DatasetStructure.accepted_keys:
                d[key] = toLoad[key] if key in toLoad else None
            dstruct = DictDatasetStructure(d)
            dataset = BaseDataset(dstruct)

            dataset.train_images = toLoad['train_images']
            dataset.train_annos = toLoad['train_annos']
            dataset.test_images = toLoad['test_images']
            dataset.test_annos = toLoad['test_annos']

            dataset.loaded = True

        return dataset


def test_base_dataset(preload=False, save=False):
    from matplotlib import pyplot as plt

    dstruct = JSONDatasetStructure("./dataset_structure_definitions/glas_original.json")
    dataset = BaseDataset(dstruct)

    sequence = list(range(len(dataset.dstruct.test_images_list)))
    sequence.reverse()
    if preload:
        dataset.preload()
    if save:
        dataset.saveTo("e:/data/GlaS/glas_original.pkl")
    for im, anno in dataset.randomGenerator(1, is_train=False):
        plt.figure()
        plt.imshow(im)
        plt.contour(anno[..., 0])
        plt.show()
        break


def test_loaded_from_pkl():
    from matplotlib import pyplot as plt

    dataset = BaseDataset.loadFrom("e:/data/GlaS/glas_resized.pkl")

    sequence = list(range(len(dataset.dstruct.test_images_list)))
    sequence.reverse()
    for im, anno in dataset.sequenceGenerator(sequence, is_train=False):
        plt.figure()
        plt.imshow(im)
        plt.contour(anno[..., 0])
        plt.show()
        break


if __name__ == "__main__":
    test_base_dataset()
