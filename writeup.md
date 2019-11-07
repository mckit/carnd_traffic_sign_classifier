# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./german_traffic_signs/image_1.jpg "Traffic Sign 1"
[image4]: ./german_traffic_signs/image_2.jpg "Traffic Sign 2"
[image5]: ./german_traffic_signs/image_3.jpg "Traffic Sign 3"
[image6]: ./german_traffic_signs/image_4.jpg "Traffic Sign 4"
[image7]: ./german_traffic_signs/image_5.jpg "Traffic Sign 5"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mckit/carnd_traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I calculated a summary of the traffic sign data set. 

* The size of training set is 34799. 
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43. 

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. To begin with I simply visualized a random image from the X_train set and along with it's label. 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I performed two steps to preprocess my data. First I shuffled the training set so that image order would not effect the outcome of my model. Next I normalized the data to try to recude errors and have more uniform values using a mean zero and equal variance equation. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following five layers:

| Layer 1        		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Layer 2          | 
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16   	|
| RELU            |                 |
| Max Pooling		| 2x2 stride, outputs 5x5x16  				|
| Flattening				| outputs 400   					|
|	Layer 3					|												|
|	Fully Connected					|	outputs 120											|
| RELU             |                   |
| Layer 4       |             |
| Fully Connected       | outputs 84     |
| RELU     |           |
| Dropout            |            |
| Layer 5     |         | 
| Fully Connected    | outputs 43       |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used a batch size of 130. After some testing I settled on 25 epochs and a learning rate of 0.005. It is possible that a better combination of these three variables could yield a more accurate model. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.988
* validation set accuracy of 0.934
* test set accuracy of 0.921

I followed the example we learned during the neural network lessons and used the LeNet architecture. This has been my first exposure to neural networks, so I wanted to stick with what was presented in the lesson. After some reading, I did add a dropout to the fourth layer to get a better result. As mentioned above, I experimented with the hyperparameters to try to get a more accurate and efficient model. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

[image3]
[image4]
[image5]
[image6]
[image7]

I included five images of German traffic signs from the web in a folder called german_traffic_signs. I think all of the images should be reasonably easy to classify. Image one is at a slight angle, but is otherwise clear. Perhaps a perspective transform could help with more extreme examples like this. Image two appears to be almost directly head on. Image three is also clear and relatively straight on. Image four is a sign on the back of a stop sign, so perhaps the outline of the other sign could interfere with the prediction. Image five is a head on view of a speed limit sign that should not present many issues. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 14 Stop     		| 14    									| 
| 39 Keep Left    			| 39 Keep Left 										|
| 13	Yield				| 13 Yield											|
| 17	No Entry      		| 34 Turn Left Ahead 					 				|
| 4	Speed Limit (70)		| 4 Speed Limit (70)							|


My model predicted four of the five signs I added correctly. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code and full results of the softmax probabilities for each prediction are included in my Ipython notebook. The four correct predictions all have a high softmax probability, while the incorrect prediction (No Entry) has a relatively low probability of .50. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99        			| Stop   									| 
| 1.0     				| Keep Left										|
| 1.0					| Yield											|
| .50      			| No Entry					 				|
| 1.0				    | Speed Limit (70)      							|
