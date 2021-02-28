### **Model for Generating Sequential Gaze Points Using Salient Images**

#### Introduction
This project is a proof of concept for a model to generate sequential gaze points using salient images. As input, this algorithm takes in images of any size and uses a premade [salient
image calculator](https://github.com/yhenon/pyimgsaliency) created by Jianming Zhang et Al. It is trained on a series of eye tracking points from the [POET dataset](http://calvin-vision.net/datasets/poet-dataset/)
created by Dim P. Papadopoulos et Al. This algorithm takes the best points from the dataset and uses them to train and test the model.

------------------------------------------------------------------

#### Using the Dataset
POET Points Download: [train.zip](http://calvin-vision.net/datasets/poet-dataset/)

To use this dataset, download it and put it in the cwd directory where your train.py and test.py code is. You must run all create_*.py files to create local data to use in
train and test.

------------------------------------------------------------------

#### Model
![Failed to Find Model](https://github.com/nvinden/SequentialSalientGazePath/blob/main/SequentialModel.png)

------------------------------------------------------------------

#### Results

This problem has proven to be tough to solve. This model is underfit. When loss is minimized, I found that the output point data settled at the middle of the image, as this
is the place where the loss between target and model output is lowest given random point data. The model did not provide enough information to reliably accuratly point
gaze points that reflect human behaviour.

As of now, I am only able to train these models on a cpu, and I believe higher computing power is important in an algorithm this deep.

I also believe that the model could have preformed better, given more advanced deep structures, that I have not yet been able to implement. I want to learn more about
creating a model like this.
