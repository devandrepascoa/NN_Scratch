import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from neural.neuralnet import NN, MathUtils, loadMnist
import numpy as np

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

model = NN(shape=[784, 128, 128, 10])

model.fit((x_train, y_train), val_dataset=(x_test, y_test),
          optimizer="adam", epochs=5000,mini_batch_active=True)
model.save()
model.load()
print(model.evaluate((x_test, y_test))["accuracy"])

image = import_to_gray("icon.jpg")
plt.imshow(image.reshape(28, 28))
for i in range(0, len(image)):
    if image[i] == 1:
        image[i] = 0.99

result = model.predict(image)  # input (784,1) [0,1]
print("Predicted {}".format(result))

plt.show()
