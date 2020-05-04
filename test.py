import argparse

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
    shape = [784]
    if args.shape is None:
        for i in range(1, args.size + 1):
            size = input("Enter the size of layer " + str(i) + ": ")
            shape.append(int(size))
    else:
        for layer in args.shape:
            shape.append(int(layer))

    shape.append(10)
    model = Network()
    model.add(DenseLayer(784, 128))
    model.add(Relu())
    model.add(DenseLayer(128, 10))
    model.add(Softmax())

    if args.fit:
        optimizer = SGD()
        model.compile(optimizer=optimizer, loss=losses.cross_entropy)
        model.fit((x_train, y_train), val_dataset=(x_test, y_test))

        print(model.evaluate((x_test, y_test))["accuracy"])

    if args.save:
        model.save(args.save)
    else:
        model.save()

    if args.evaluate:
        model.load()
        print(model.evaluate((x_test, y_test))["accuracy"])


def argparser():
    argparser = argparse.ArgumentParser(description="Mnist neural network trainer")
    argparser.add_argument("--shape", nargs='+', default=["128", "128"],
                           help="Example: --shape 128 128 will create nn of shape 784 128 128 10")
    argparser.add_argument("--save", type=str, help="Save network to a certain path")
    argparser.add_argument("--evaluate",
                           type=str, help="Use if evaluation on validation dataset is required")
    argparser.add_argument("--fit", action="store_const", const=True, help="Use in case you want to train the network")
    argparser.add_argument("--epochs", type=int, default=5000, help="Number of epochs for training")
    argparser.add_argument("--optimizer", default="gd", type=str,
                           help="Optimizer choice: momentum-> Gradient Descent with momentum\n"
                                "rms->Root Mean Squared Prop\n"
                                "adam->Adam optimizer \n"
                                "gd->Gradient descent")
    argparser.add_argument("--dropout", type=float, default=0.8, help="Selects dropout value")
    argparser.add_argument("--learning-rate", type=float, default=0.01, help="Selects learning rate value ")
    argparser.add_argument("--mini-batch-size", type=int, default=64, help="Selects mini batch size")
    return argparser.parse_args()


if __name__ == "__main__":
    args = argparser()
    run_nn(args)
