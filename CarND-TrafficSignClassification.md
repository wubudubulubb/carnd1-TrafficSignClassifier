#**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/DataBarChart.png "Visualization"
[image2]: ./examples/30.jpg "Traffic Sign 1"
[image3]: ./examples/aheadonly.jpeg "Traffic Sign 2"
[image4]: ./examples/RoadWork.jpg "Traffic Sign 3"
[image5]: ./examples/STOP.jpg "Traffic Sign 4"
[image6]: ./examples/keepright.jpg "Traffic Sign 5"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup 

###Data Set Summary & Exploration

####1. 
I used numpy methods to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is spreaded in training, validation and test sets.

![alt text][image1]

###Design and Test a Model Architecture

####1.

As pre-processing, I converted the images to gray-scale and normalized the pixel-color values. (i.e from [0,255] to [-1,1])
The reason for gray scale conversion is that from a quick research in web I found out that having the color layers didn't add too much into the performance of the network for classification of the data. (https://navoshta.com/traffic-signs-classification/)
 

I didn't use augmented data for training. What I did was, as suggested by my mentor, first use 20% of the training data for training the model as overfitting to this portion, and then use the whole set of training data with dropout to reach the required accuracy.


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Dropout               | used 0.5 keep_prob while training             |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32	|
| RELU		            |              									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32   				|
| Flatten				| output = 800									|
| Dropout               | used 0.5 keep_prob while training             |
| Fully Connected       | input = 800 output = 400                      |
| RELU		            |              									|
| Fully Connected       | input = 400 output = 200                      |
| RELU		            |              									|
| Dropout               | used 0.5 keep_prob while training             |
| Fully Connected       | input = 200 output = 43 (n_classes)           |
 


To train the model, I used an AdamOptimizer. Adam was the mostly suggested one in Karpathy's online lessons in youtube, as it's faster compared to other optimizers. 

First I used the 20% of the training data to find a complex enough model to overfit. The model is described in above table. 
Hyper parameters were selected as:
EPOCHS = 50
BATCH_SIZE =128
learning_rate = 0.001
sigma = 0.1

While the number of EPOCHS is 50, I break the training if the validation accuracy reaches >0.95.

My final model results were:
* training set accuracy of 0.99
* validation set accuracy of 0.951 
* test set accuracy of 0.990

I started with using the LeNet architecture. It was recommended by the course, and it is known to be good at image classifiation. 
I used the original LeNet architecture on the 20% of the images in the training set. First, I changed the dimensions of the fully connected layers to match the number of classes in German Traffic sign dataset (43).

The results were not satisfactory. Then I increased the layers of convolutions in the first two concolution layers to 32 and 16, respectively. It was not enough for overfitting with training accuracy of >0.91. I increased the layers by twice their size. I was able to have a training accuracy grater than 0.92 in this represantative training set.

Then I added dropouts and used the whole training dataset to train my network. The training and validation accuracies are 0.99 and 0.951 respectively.
 
###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

The first image might be difficult to classify because the original height of the image is longer than the width, and it will be skewed considerably when preprocessed to size (32,32)
The same is valid for the second, third and fourth image. Moreover, the fourth image has the watermark from getty images on it. 


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1,Speed limit(30km/h) | 1,Speed limit(30km/h)   						| 
| 35,Ahead only     	| 11,Right-of-way at the next intersection 		|
| 25,Road work			| 31,Wild animals crossing					    |
| 14,Stop	      		| 12,Priority road					 			|
| 38,Keep right			| 38,Keep right      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This is lower compared to the testing accuracy. The most probable reason being the aspect ratio of the photos I chose were not compatible/ the zoom level of the photos I chose were not comparable to the ones in the dataset. The keep right image was classified correctly in each of my iterations. Which has no skew/rotation.


The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit(30km/h) sign (probability of 0.58), and the image does contain a Speed limit(30km/h). We can see from the top five softmax probabilities that model is almost 97% sure that this is a speed limitation sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .584         			| Speed limit(30km/h)   						| 
| .256     				| Speed limit (70km/h) 							|
| .103					| Speed limit (50km/h)							|
| .023	      			| Speed limit (100km/h)					 		|
| .008				    | Speed limit (80km/h)    						|


For the second image, the model is considerably sure that it is a "Right-of-way at the next intersection" sign, which is the wrong prediction. The correct sign (Ahead only) does not exist in top 5 most probable predictions 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .888         			| Right-of-way at the next intersection   		| 
| .065     				| Roundabout mandatory 							|
| .045					| Turn right ahead								|
| .0008	      			| General caution					 			|
| .0007				    | Traffic signals      							|

For the third image, the system is not really sure about the prediction. The prediction with the highest probability is "Wild animals crossing", with prob = 0.222. The system gives a probability of 0.158 to "Road work" sign, which is the correct sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .222         			| Wild animals crossing                  		| 
| .158     				| Road work           							|
| .131					| Bicycles crossing								|
| .068	      			| Beware of ice/snow 				 			|
| .046				    | Children crossing     						|

For the fourth image, the system again is not really sure about the prediction. One other observation is that the correct prediction does not exist in the top5 softmax probabilities. This is most probably due to the watermark on the image. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .311         			| Priority road                         		| 
| .116     				| Speed limit (80km/h)           				|
| .084					| Roundabout mandatory							|
| .077	      			| Speed limit (100km/h) 			 			|
| .065				    | Speed limit (30km/h)     						|

For the fifth image, siystem is 100% sure it's Keep right sign, which is the correct prediction. This was expected, because it is the only image in the 5-image-downloaded-dataset with almost no noise. 
