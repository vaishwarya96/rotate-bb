import argparse

class TrainOptions():

    def __init__(self):
        self.args = None

    def initialize(self, parser):
        parser.add_argument('--train_image_paths', required=True, type=str, help='Path to the train images directory')
        parser.add_argument('--train_label_paths', required=True, type=str, help='Path to the train labels directory')
        parser.add_argument('--test_image_paths', required=True, type=str, help='Path to the test images directory')
        parser.add_argument('--test_label_paths', required=True, type=str, help='Path to the test labels directory')

        parser.add_argument('--image_size', default=64, type=int, help='Size of the image to be resized to')
        parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Path to save the model')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum for Adam optimizer')
        parser.add_argument('--beta2', type=float, default=0.9, help='momentum for Adam optimizer')
        parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
        parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
        parser.add_argument('--result_dir', type=str, default='results', help='Path to save the results')


        return parser

    def get_args(self):

        return self.args

    def print_parse(self):

        parser = argparse.ArgumentParser()
        parser = self.initialize(parser)
        self.args = parser.parse_args()
        self.parser = parser

        message=''
        message += "--------------Input arguments---------------\n"

        for key, value in vars(self.args).items():
            comment=''
            default = self.parser.get_default(key)

            if value != default:
                comment = '\t[default: %s]' %str(default)

            message += '{:>25}: {:<30}{}\n'.format(str(key), str(value), comment)
        message += "------------------End-----------------------"
        #print(message)
        return parser

