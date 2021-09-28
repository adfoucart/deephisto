# Deep Learning applied to Digital Pathology

PhD Thesis code base

## Research topics:

* Artefact detection in whole-slide imaging
* Effects of annotation imperfections on segmentation methods (SNOW)
* Analysis of multi-expert annotations using the Gleason2019 dataset
* Analysis of digital pathology challenge evaluation metrics

Research blog: https://adfoucart.be/research 

## Publications

Foucart, A., Debeir, O., & Decaestecker, C.<br />
Processing multi-expert annotations in digital pathology: a study of the Gleason2019 challenge.<br />
17th International Symposium on Medical Information Processing and Analysis (SIPAIM 2021)<br />
Accepted to the conference, awaiting publication.

Foucart, A., Debeir, O., & Decaestecker, C.<br />
Snow Supervision in Digital Pathology: Managing Imperfect Annotations for Segmentation in Deep Learning<br />
Preprint (2020) - https://www.researchsquare.com/article/rs-116512/v1

Foucart, A., Debeir, O., & Decaestecker, C. <br />
SNOW: Semi-Supervised, NOisy and/or Weak Data for Deep Learning in Digital Pathology. <br />
In 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019) (pp. 1869-1872) <br />
https://doi.org/10.1109/ISBI.2019.8759545

Van Eycke, Y.-R., Foucart, A., & Decaestecker, C. <br />
Strategies to Reduce the Expert Supervision Required for Deep Learning-Based Segmentation of Histopathological Images. <br />
Frontiers in Medicine, 6, 222. (2019)<br />
https://doi.org/doi:10.3389/fmed.2019.00222

Foucart, A., Debeir, O., & Decaestecker, C. <br />
Artifact Identification in Digital Pathology from Weak and Noisy Supervision with Deep Residual Networks. <br />
The 4th International Conference on Cloud Computing Technologies and Application (CloudTech'18) (2018) <br />
https://doi.org/doi:10.1109/CloudTech.2018.8713350

## Dependencies

* "Legacy code" require Python 3.6 / TensorFlow 1.14 (mostly in data/ & network/)
* More up-to-date code require Python 3.7+ / TensorFlow 2 (in generator/ & model/ + run/network_train_v2.py)
* Openslide (for running the whole-slide artefact detector)
* Scikit-Image
* Numpy
* SimpleITK (for running the Gleason code)
* Scikit-learn (for running the Gleason code)

## Datasets

The datasets used in this project are:

### GlaS challenge

https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/

K. Sirinukunwattana, J. P. W. Pluim, H. Chen, et al,  “Gland segmentation in colon histology images: The glas challenge contest,” 
Med. Image Anal., vol. 35, pp. 489–502, 2017.

### Epithelium + Nuclei

http://www.andrewjanowczyk.com/deep-learning/

A. Janowczyk and A. Madabhushi, “Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases,” 
J. Pathol. Inform., vol. 7, no. 29, 2016.

### Artefact

Foucart, Adrien (2020). "Artefact segmentation in digital pathology whole-slide images [Data set]." https://doi.org/10.5281/zenodo.3773096

## Gleason2019 challenge

https://gleason2019.grand-challenge.org/Register/

Nir G, Hor S, Karimi D, Fazli L, Skinnider BF, Tavassoli P, Turbin D, Villamil CF, Wang G, Wilson RS, Iczkowski KA. Automatic grading of prostate cancer in digitized histopathology images: Learning from multiple experts. Medical image analysis. 2018 Dec 1;50:167-80.

Karimi D, Nir G, Fazli L, Black PC, Goldenberg L, Salcudean SE. Deep Learning-Based Gleason Grading of Prostate Cancer From Histopathology Images—Role of Multiscale Decision Aggregation and Data Augmentation. IEEE journal of biomedical and health informatics. 2019 Sep 30;24(5):1413-26.