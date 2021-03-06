from argparse import ArgumentParser
from pdb import pm
import cmd
import pickle

from labelizer import Labelizer, INFO_OPTIONS, DEFAULT_OPTIONS


# UNCOMMENT TO REMOVE RANDOMIZATION (for reproductability)
#
# import numpy as np
# import random
# np.random.seed(0)
# random.seed(0)

class LabelizerCli(cmd.Cmd):
    """Interaction for quick labelization through Labelizer"""

    def __init__(self, options, *args, **kwargs):
        self.labelizer = Labelizer(options.directory)
        self.verbose = not options.quiet
        self.log("Load model's weights")
        self.labelizer.load_model()

        cmd.Cmd.__init__(self, *args, **kwargs)

    def log(self, string):
        if self.verbose is True:
            print(string)

    def preloop(self):
        self.log("Welcome. Type 'step' to move a step forward")

    def do_labels(self, line):
        self.log("Labels:")
        for label, count in self.labelizer.labels_count().iteritems():
            self.log("\t%s:\t%d" % (label, count))

    def do_classifier(self, line):
        if hasattr(self.labelizer, "classifier"):
            self.log(self.labelizer.classifier)
        else:
            self.log("Error: at least one step is needed")

    def do_model(self, line):
        self.log(self.labelizer.options.model)

    def do_step(self, line):
        self.log("Clean the directory")
        self.labelizer.init_dir()
        self.do_labels("")
        self.log("Learn on labelled elements")
        self.labelizer.learn()
        self.log("Look for new elements")
        self.labelizer.get_new_samples()

    def do_generalize(self, line):
        self.log("Apply on the full dataset")
        self.labelizer.get_new_samples(limit=False)
        self.do_labels("")

    def do_preload(self, line):
        self.log("Preload the full dataset")
        self.labelizer.preload()

    def do_dump(self, line):
        with open(line, "w") as fdesc:
            pickle.dump(self.labelizer._cache, fdesc)

    def do_load(self, line):
        with open(line) as fdesc:
            self.labelizer._cache = pickle.load(fdesc)

    def do_options(self, line):
        if line:
            # Set a value
            try:
                info = line.split(" ", 1)
                name, value = info
                self.labelizer.options.set_option(name, value)
            except Exception as e:
                self.log(e)
                return

            # Special cases
            if name == "model":
                self.labelizer.load_model()
            elif name == "classifier":
                if hasattr(self.labelizer, name):
                    delattr(self.labelizer, name)
            self.log("%s = %s" % (name, getattr(self.labelizer.options, name)))
        else:
            self.log("Current options:\n%s" % self.labelizer.options)
            self.log("Use options <name> <value> to set an option")

    def do_EOF(self, line):
        return True

if __name__ == '__main__':
    parser = ArgumentParser("Quick interactive labelizer")
    parser.add_argument("directory", help="Base directory to watch")
    parser.add_argument("-q", "--quiet", help="Quiet mode")
    args = parser.parse_args()

    LabelizerCli(args).cmdloop()
