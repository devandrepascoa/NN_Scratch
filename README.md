# NN_Scratch

Neural network from scratch, implemented with numpy and a lot of patience with calculus and linear algebra, (vectorization and calculating proper gradients), recently implemented Gradient checking and Dropout regularization, now focusing on implementing a proper library, which I will be able to use to further understand the fundamental concepts of machine learning,

Performance of a [784,128,128,10] network for MNIST classification with 0.5 learning rate is at 
97% for validation accuracy and 96% for training data accuracy(which means regularization has been properly implemented as it has generalized)

The start of the program execution
```
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

Starting training, NN dimensions: [784, 128, 128, 10]
Epoch:0,Cost:2.851550957041901, Accuracy:10.383333333333333
Validation Cost:2.2301635069375916, Validation Accuracy:19.29
Epoch:100,Cost:0.4705938394307374, Accuracy:86.16
Validation Cost:0.263984076214809, Validation Accuracy:92.39
Epoch:200,Cost:0.3546978889384424, Accuracy:89.82833333333333
Validation Cost:0.1997824425174201, Validation Accuracy:94.02000000000001
Epoch:300,Cost:0.2995063853680104, Accuracy:91.39333333333333
Validation Cost:0.16883666002554917, Validation Accuracy:95.05
	Epoch:400,Cost:0.2639805195222709, Accuracy:92.41333333333334
Validation Cost:0.14901901587825286, Validation Accuracy:95.69

```
