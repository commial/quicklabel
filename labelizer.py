import os
import shutil
import random
from collections import namedtuple

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from sklearn.ensemble import RandomForestClassifier
import numpy as np

MODEL2CLS = {"ResNet50": ResNet50}
CLASSIFIER2CLS = {"RandomForest": RandomForestClassifier}

DEFAULT_OPTIONS = {"suffix": "_potential",
                   "dataset": "dataset",
                   "needhelp": "needhelp",
                   "needhelp_prob": 0.6,
                   "new_sample": 50,
                   "model": "ResNet50",
                   "classifier": "RandomForest",
}
Options = namedtuple("Options", DEFAULT_OPTIONS.keys())


class Labelizer(object):

    def __init__(self, directory):
        # basename -> feature vector
        self._cache = {}
        self.options = Options(**DEFAULT_OPTIONS)
        self.directory = directory

    def load_model(self):
        cls = MODEL2CLS.get(self.options.model, None)
        if cls is None:
            raise ValueError("Unknown model %s" % self.options.model)
        self._model = cls(weights='imagenet', include_top=False)

    def process(self, img_path):
        # TODO: preprocess_input according to model
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = self._model.predict(x)
        # features.shape : 1, 1, 1, NB_FEATURE
        return features[0][0][0]

    def init_dir(self):
        """Clean target directory and get labels"""
        labels = set()
        for element in os.listdir(self.directory):
            dir_path = os.path.join(self.directory, element)
            if not os.path.isdir(dir_path):
                continue
            if (element.endswith(self.options.suffix) or
                element == self.options.needhelp):
                shutil.rmtree(dir_path)
            elif element == self.options.dataset:
                continue
            else:
                labels.add(element)
        self.labels = list(labels)

    def labels_count(self):
        return {label:len(os.listdir(os.path.join(self.directory, label)))
                for label in self.labels}

    def get_feature(self, elem_path):
        name = os.path.basename(elem_path)
        feature = self._cache.get(name, None)
        if feature is None:
            feature = self.process(elem_path)
            self._cache[name] = feature
        return feature

    def learn(self):
        """Learn from current labels"""
        # PRE: init_dir
        assert hasattr(self, "labels")

        # Process labelised elements
        X_train = []
        Y_train = []
        already_done = set()
        for i, label in enumerate(self.labels):
            base_dir = os.path.join(self.directory, label)
            for element in os.listdir(base_dir):
                elem_path = os.path.join(base_dir, element)
                if not os.path.isfile(elem_path):
                    continue
                name = element

                # Avoid further processing of an already labelised element
                already_done.add(name)
                feature = self.get_feature(elem_path)

                X_train.append(feature)
                Y_train.append(i)

        # Learn
        ## Format elements
        assert len(X_train) == len(Y_train)
        if (len(Y_train) == 0 or
            len(set(Y_train)) == 1):
            raise ValueError("At least 2 labels with at least one element are required")
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        ## Fit classifier
        cls = CLASSIFIER2CLS.get(self.options.classifier, None)
        if cls is None:
            raise ValueError("Unknown classifier %s" % self.options.classifier)
        self.classifier = cls()
        self.classifier.fit(X_train, Y_train)
        self._already_done = already_done

    def get_new_samples(self, limit=True, force_diversity=False):
        """Apply the current classifier on new sample
        @force_diversity: avoid only one potential class
        @limit: consider only a limited number of element
        """
        # PRE: learn
        assert hasattr(self, "classifier")

        # Get some new samples
        dataset_path = os.path.join(self.directory, self.options.dataset)
        needhelp_dir = os.path.join(self.directory,
                                    self.options.needhelp)

        elements = os.listdir(dataset_path)
        random.shuffle(elements)

        i = 0
        has_diversity = False
        founds_label = set()
        for element in elements:
            if element in self._already_done:
                continue
            if ((limit and i >= self.options.new_sample) and
                (not force_diversity or has_diversity)):
                # wait for all label or a needhelp
                break
            i += 1
            name = element
            element_path = os.path.join(dataset_path, name)
            if not os.path.isfile(element_path):
                continue
            feature = self.get_feature(element_path)

            y = self.classifier.predict_proba(np.array([feature]))[0]

            label_id = y.argmax()
            label_prob = y[label_id]
            if label_prob < self.options.needhelp_prob:
                # Need help
                if not os.path.exists(needhelp_dir):
                    os.mkdir(needhelp_dir)
                shutil.copy(element_path, needhelp_dir)
                has_diversity = True
            else:
                label_dir = os.path.join(self.directory,
                                         "%s%s" % (self.labels[label_id],
                                                   self.options.suffix))
                if not os.path.exists(label_dir):
                    os.mkdir(label_dir)
                shutil.copy(element_path, label_dir)
                founds_label.add(label_id)
            if len(founds_label) == len(self.labels):
                has_diversity = True

