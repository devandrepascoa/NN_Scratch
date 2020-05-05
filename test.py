import argparse
import sys

from PIL import Image, ImageOps

from neural import losses
from neural.Network import Network
from neural.datasets import loadMnist
from neural.layers import *
import numpy as np

(x_train, y_train), (x_test, y_test) = loadMnist()


def import_to_gray(img_path):
    im = Image.open(img_path)
    im = im.resize((28, 28)).convert('L')
    im = ImageOps.invert(im)
    im = np.array(im)
    im = im.reshape(784, 1) / 255.0
    return im


def run_nn(args):
    assert args.fit is not None or args.evaluate is not None, "You have to choose either training and evaluating or " \
                                                              "both. "

    print("""
          _---~~(~~-_.
        _{        )   )
      ,   ) -~~- ( ,-' )_
     (  `-,_..`., )-- '_,)
    ( ` _)  (  -~( -_ `,  }     BIG BRAIN MACHINE BY ANDRE PASCOA

    (_-  _  ~_-~~~~`,  ,' )
      `~ -^(    __;-,((()))
            ~~~~ {_ -_(())
                   `\  }
                     { } 
    """)

    model = Network()
    model.add(DenseLayer(784, 128))
    model.add(Sigmoid())
    model.add(Dropout(keep_prob=0.8))
    model.add(DenseLayer(128, 128))
    model.add(Sigmoid())
    model.add(Dropout(keep_prob=0.8))
    model.add(DenseLayer(128, 10))
    model.add(Softmax())

    if args.fit:
        optimizer = SGD(learning_rate=args.lr)
        model.compile(optimizer=optimizer, loss=losses.cross_entropy)
        try:
            model.fit((x_train, y_train), epochs=args.epochs, val_dataset=(x_test, y_test),
                      batch_size=args.mini_batch_size)
        except KeyboardInterrupt:
            print("Saving neural network..")
            model.save()
            sys.exit()

        if args.save:
            model.save(args.save)
        else:
            model.save()

    if args.evaluate:
        model.load(args.evaluate)
        print("Model accuracy:" + str(model.evaluate((x_test, y_test))["accuracy"]))


def argparser():
    argparser = argparse.ArgumentParser(description="Mnist neural network trainer")
    argparser.add_argument("--save", type=str, help="Save network to a certain path")
    argparser.add_argument("--evaluate", default="neural_network",
                           type=str, help="Use if evaluation on validation dataset is required")
    argparser.add_argument("--fit", action="store_const", const=True, help="Use in case you want to train the network")
    argparser.add_argument("--epochs", type=int, default=500, help="Number of epochs for training")
    argparser.add_argument("--lr", type=float, default=0.01, help="Selects learning rate value ")
    argparser.add_argument("--mini-batch-size", type=int, default=64, help="Selects mini batch size")
    return argparser.parse_args()


if __name__ == "__main__":
    args = argparser()
    run_nn(args)
