import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import neural
from neural.neuralnet import *

(x_train, y_train), (x_test, y_test) = loadMnist()


def import_to_gray(img_path):
    im = Image.open(img_path)
    im = im.resize((28, 28)).convert('L')
    im = ImageOps.invert(im)
    im = np.array(im)
    im = MathUtils.normalize(im.reshape(784, 1))
    return im


print("""
      _---~~(~~-_.
    _{        )   )
  ,   ) -~~- ( ,-' )_
 (  `-,_..`., )-- '_,)
( ` _)  (  -~( -_ `,  }     BIG BRAIN MACHINE BY ANDRÉ PÁSCOA

(_-  _  ~_-~~~~`,  ,' )
  `~ -^(    __;-,((()))
        ~~~~ {_ -_(())
               `\  }
                 { } 
""")

nn = NN((x_train, y_train), val_dataset=(x_test, y_test), epochs=10000, shape=[784, 30, 10],
        learning_rate=0.5,
        enable_dropout=True)
nn.save()

new_nn = NN.load()
print(new_nn.evaluate((x_test, y_test))["accuracy"])

image = import_to_gray("icon.png")
plt.imshow(image.reshape(28, 28))
for i in range(0, len(image)):
    if image[i] == 1:
        image[i] = 0.99

result = new_nn.predict(image)  # input (784,1) [0,1]
print("Predicted {}".format(np.argmax(result)))

plt.show()
