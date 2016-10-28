from argparse import ArgumentParser
from pdb import pm
import cmd

from labelizer import Labelizer, DEFAULT_OPTIONS


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

    def do_options(self, line):
        self.log("Current options:")
        for option, default_value in DEFAULT_OPTIONS.iteritems():
            cur_value = getattr(self.labelizer.options, option)
            self.log("\t{name:20s}\t{cur:20s} (default is '{default:s}')".format(
                name=str(option),
                cur=str(cur_value),
                default=str(default_value)))

    def do_EOF(self, line):
        return True

if __name__ == '__main__':
    parser = ArgumentParser("Quick interactive labelizer")
    parser.add_argument("directory", help="Base directory to watch")
    parser.add_argument("-q", "--quiet", help="Quiet mode")
    args = parser.parse_args()

    LabelizerCli(args).cmdloop()
