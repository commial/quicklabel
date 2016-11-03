# QuickLabel
Quick and dirty helper for interactive image labelisation

Description
===========

QuickLabel is a quick & dirty CLI implementation of an interactive image labelisation helper. This tool is inspired by the MLDB tool [DeepTeach](http://blog.mldb.ai/blog/posts/2016/10/deepteach/).

The idea is to start with a few labelised images (let's say 5 in each label). Using a pre-trained CNN model, without the classifier part, these images are converted into vectors representation.
A fast classifier is then trained on them. Then, this classifier is used to predict new elements from the user provided dataset (unlabelised images).
Depending on the prediction confidence, computed images are copied into their estimated label, or into a `needhelp` directory.

The user, by manually labelling a few images from these results, will then refined what it is looking for.

Iteratively, the classifier *obtains a clearer idea* on how images should be labelled, with a fast convergence (empirically, less than a dozen of iteration are needed). Informally, by sorting corner cases at each step, the user is refinning the classifier on its decision frontiers, providing useful information on its goal.

Finally, the user can apply the learned classifier on the full dataset.
Another option would be to use the labelled elements to fine-tune a pre-trained CNN (often called "transfer learning").

By default, QuickLabel uses:
* CNN: ResNet50
* Classifier: RandomForest

But any classifier from `sklearn` and any feature extractor from Keras can be use.

Installation
============

QuickLabel depends on [scikit-learn](http://scikit-learn.org) and [Keras](http://keras.io). Please refer to their official sites for installation. 

Usage
=====

Using the default settings, the workflow is as follow:

1. In an empty directory, create a `dataset` directory containing the unlabelled images, with unique names (you may want to use a symbolic link for this)
1. In this same directory, create a directory for each label you want to consider (for instance, `mylabel` and `not_mylabel`)
1. Launch `labelizer.py` with this directory as argument
1. Aside, launch your preferred image visualisation tool, with an easy-to-use directory to directory copy interface. For instance, `gthumb` or several `thunar` window
1. Put a few images in each label directory
1. Launch the feature extraction, learning and search for new images by typing `step` in the CLI
1. For each label, a `{label}_potential` directory may have been created. If so, it contains images the classifier has considered to belong to `{label}`. Feel free to move correctly labelled images to their respecting directory.
1. A `needhelp` directory may have been created. If so, it contains classifier corner cases. Moving them in their expecting label will likely increases the classifier performance.
1. Iterate until `_potential` directory contains only exepected elements (using `step`)
1. You can use `generalize` to apply this classification to the full dataset.

Do not hesitate to take a look to `options` and modify the different hyperparameters, as thresholds, model, classifier, ...
