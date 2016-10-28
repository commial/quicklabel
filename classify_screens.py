from argparse import ArgumentParser
import os
import shutil
import random
from pdb import pm

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import numpy as np
from sklearn.ensemble import RandomForestClassifier

parser = ArgumentParser()
parser.add_argument("directory", help="Base directory to watch")
args = parser.parse_args()

print "Load model"
model = ResNet50(weights='imagenet', include_top=False)

suffix = "_potential"
dataset = "dataset"
needhelp = "needhelp"

# basename -> feature vector
cache = {}

NEEDHELP_PROB = 0.7
NEW_SAMPLE = 50

def process(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    return features[0][0][0]


while True:
    labels = set()
    for element in os.listdir(args.directory):
        dir_path = os.path.join(args.directory, element)
        if not os.path.isdir(dir_path):
            continue
        if element.endswith(suffix) or element == needhelp:
            shutil.rmtree(dir_path)
        elif element == dataset:
            continue
        else:
            labels.add(element)

    labels = list(labels)
    print "Labels:", ", ".join(labels)
    X_train = []
    Y_train = []
    already_done = set()
    for i, label in enumerate(labels):
        print "Process %s elements" % label
        base_dir = os.path.join(args.directory, label)
        for element in os.listdir(base_dir):
            elem_path = os.path.join(base_dir, element)
            if not os.path.isfile(elem_path):
                continue
            name = element
            already_done.add(name)
            feature = cache.get(name, None)
            if feature is None:
                feature = process(elem_path)
                cache[name] = feature

            X_train.append(feature)
            Y_train.append(i)

    # Learn
    assert len(X_train) == len(Y_train)
    rfc = RandomForestClassifier()
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    rfc.fit(X_train, Y_train)

    # Get some new samples
    print "Get new samples"
    dataset_path = os.path.join(args.directory, dataset)
    elements = os.listdir(dataset_path)
    random.shuffle(elements)

    i = 0
    found_all_needhelp = False
    founds_label = set()
    for element in elements:
        if element in already_done:
            continue
        if i >= NEW_SAMPLE and found_all_needhelp:
            print "%d samples done, wait for all label or a needhelp" % i
            break
        i += 1
        name = element
        feature = cache.get(name, None)
        if feature is None:
            feature = process(os.path.join(dataset_path, element))
            cache[name] = feature

        y = rfc.predict_proba(np.array([feature]))[0]

        label_id = y.argmax()
        label_prob = y[label_id]
        element_path = os.path.join(dataset_path, name)
        if label_prob < NEEDHELP_PROB:
            # Need help
            needhelp_dir = os.path.join(args.directory, needhelp)
            if not os.path.exists(needhelp_dir):
                os.mkdir(needhelp_dir)
            shutil.copy(element_path, needhelp_dir)
            found_all_needhelp = True
        else:
            label_dir = os.path.join(args.directory, labels[label_id] + suffix)
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)
            shutil.copy(element_path, label_dir)
            founds_label.add(label_id)
        if not found_all_needhelp and len(founds_label) == len(labels):
            found_all_needhelp = True

    raw_input("[Press enter to proceed]")
