# TouchDynamics
### Touch Dynamics Biometrics test on 4 State of Art datasets<br>
<p align="center">
  <img src="doc/images/logo.png">
</p>

### Index

- [**What is TouchDynamics?**](#what-is-touch-dynamics)
- [**System Dependencies**](#requirements-and-dependencies)
- [**Quick Start**](#quick-start)
- [**Classifiers**](#classifiers)
- [**Credits**](#credits)

<ul>


## What is Touch Dynamics?

**TouchDynamics** is project developed for a "Biometric systems" exam 
at **University Aldo Moro** of Taranto Italy. <br>
its goal is to replicate and test best and most popular classifiers on State of Art datasets. 
It is tested on 4 different datasets, 2 are datasets of Keystrokes, 2 are datasets od swipes.
1 - TouchAnalytics [58] (Keystrokes)
2 - BioIdent [70]   (Keystrokes)
3 - The Mobikey [31]    (Swipes)
4 - Weka Arff [68]  (Swipes)



## Requirements and Dependencies
```
(Tested on Python on 3.7)
pandas~=1.0.5
scikit-learn~=0.24.2
utils~=1.0.1
Keras~=2.3.1
matplotlib~=3.3.3
numpy~=1.19.4
scipy~=1.7.0
joblib~=0.14.1
sklearn~=0.0
imblearn~=0.0
seaborn~=0.10.1
joblib~=0.14.1
keras~=2.3.1
tensorflow~=1.15.0
 ```


## Quick Start

**Automatic Setup installer with Python :**<br>
1 - Download and Install "Logs Manager - Setup Online.exe" <br>
2 - Good work with **Logs Manager** <br>
NB: Internet Connection required


## Classifiers

<li>
    Random Forest <br>
    Collection classifier obtained from the bagging of decision trees.
    Random forests are a solution that minimizes the overfitting of 
    the training set with respect to decision trees.
    <p align="center">
        <img src="doc/images/RF.png" width="500" height="600">
    </p>
</li>
<li>
    kNN <br>
    Object classifier based on the features of objects close to the one considered.
    The INPUT consists of the k closest training examples in the functionality space.
    The OUTPUT is a membership in a class.
    An object is classified by a plurality vote of its neighbors, with the object 
    assigned to the most common class among its k closest neighbors.
    If k = 1, the object is simply assigned to the class of that single closest neighbor.
    <p align="center">
        <img src="doc/images/kNN.png" width="500" height="600">
    </p>
</li>
<li>
    SVM <br>
    Represent the examples as points in space so that they are separated by as 
    large a space as possible.
    The new examples will be predicted based on the space / category in which they fall.
    If not linear, the SVM carries out the classification using the kernel method,
    implicitly mapping their inputs in a space of multi-dimensional features.
    <p align="center">
        <img src="doc/images/SVM.gif" width="500" height="600">
    </p>
</li>
<li>
    Neural Network <br>
    A feed-forward neural network is an artificial neural network where connections 
    between units do not form loops, differentiating themselves from recurrent neural networks.
    In this neural network, information only moves in one direction, forward,
    with respect to entry nodes, through hidden nodes (if any) to exit nodes.
    Feed-forward networks do not have input memory that occurred in previous times, 
    so the output is determined only by the current input.
    <p align="center">
        <img src="doc/images/NN.gif" width="500" height="600">
    </p>
</li>

## Instructions

- XX
- XXX
- XXXX


<li>

### Credits

**Developed by:**

[**Chrism1c**](https://github.com/Chrism1c)

</li>
</ul>
