# **Behavioral Cloning Project**

## Final Video:
[![SDC on youtube](https://img.youtube.com/vi/mX612XJhpyc/0.jpg)](https://youtu.be/mX612XJhpyc)

[//]: # (Image References)

[image1]: ./ReportMedia/tumblerToy.jpg "Tumbler Toy"
[image2]: ./ReportMedia/nn-architecture.png "CNN Architecture"
[image3]: ./ReportMedia/LEFT_ANGLE.gif "Left angle formula"
[image4]: ./ReportMedia/RIGHT_ANGLE.gif "Right angle formula"
[image5]: ./ReportMedia/center_lane.jpg "Center Lane"
[image6]: ./ReportMedia/normal.jpg "Normal Image"
[image7]: ./ReportMedia/flipped_img.jpg "Flipped Image"
[image8]: ./ReportMedia/1.jpg "1"
[image9]: ./ReportMedia/2.jpg "2"
[image10]: ./ReportMedia/3.jpg "3"
[image11]: ./ReportMedia/4.jpg "4"
[video1]: ./ReportMedia/robust.mp4 "Video"
[video2]: ./ReportMedia/video.mp4 "Video"
[video3]: ./ReportMedia/grayscale_issue.mp4 "Grayscale issue"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 - A video recording of my vehicle driving autonomously one lap around the track. Can be found [here][video2]

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 100-125) 

The model includes RELU layers to introduce nonlinearity (code lines 104-108), and the data is normalized in the model using a Keras lambda layer (code line 102). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 103, 110, 112). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 86). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 117).

And at the same line, you can see that I used **Mean Squared Error** as the loss function because I am doing regression, that is, calculating the error between the predicted steering value and the actual steering value.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I drove the car down the track in the initial direction of the car(when the car is spawn) and in the opposite direction for 2 or 3 laps each.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to watch the car closely to make assumptions about what was missing or wrong.

My first step was to use a convolution neural network model similar to the Lenet-5 architecture. I wanted to give this model a try because I was really impressed by the performance in the Traffic Sign Classification. And I wanted to see it on a regression problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the number of epochs as the first step. If the validation loss was to start increasing at Epoch number X, I would set the number of epochs to (X-1).

Then I also added **Dropout Layers**. I added one before the convolution layers with drop rate 0.2. I also added 2 dropout layers before the fully connected (Dense) layers with drop rate 0.5 to prevent overfitting. I decided to keep the rate higher for Dense layers because the number of activated nodes is much higher there.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... To improve the driving behavior in these cases, I recorded some recovery behaviors. That solved all but one weird behavior; the bridge...

Then, I realized that was quite normal. The images were still in `RGB`. So, for the model, the bridge and a regular asphalt road was totally different since they were in different colors. **At that moment, I made my luckiest mistake of this project.** Why not preprocess the images to convert them to grayscale? Now is a  good time to note that I have never recorded any training data that is off the track.

Although working on grayscale images was a little more tidious than I expected (because the scripts were expecting an image array of shape (x,y,3), not (x,y,1)), after changing `drive.py` and `video.py` a little bit, [I finally got this video here][video3].

Now, how cool is that? I have never shown the model anything about how to drive on the dirt road, but it actually went onto and out of that dirt road with only a bit of a scratch. Then I realized, it was really difficult to distinguish the dirt and the asphalt on a grayscale image and that's how the vehicle ended up in the dirt road. So, I unintentionally made a rebellious driver... :sweat_smile:

Then, I tried the `HSV` color space using 1 channel at a time. S-channel was actually good enough to keep the car on the track, but it was oscillating most of the time around the center.

So, I finally went back to `RGB` color space and to fix the recovery issue on the bridge, I just added more training data targeting specifically recovering on the bridge.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road. And it was staying on the track even when I interfered by driving the car towards the edges. The moment I let it drive autonomously, it would drive towards the center like a tumbler toy. [Here is a 3rd-person-view (compressed) video][video1] of it.

![Tumbler toy picture][image1]


#### 2. Final Model Architecture

I used [Nvidia's Self Driving Car deep learning network architecture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

1. Normalization layer
2. Convolution layer (24, 5, 5)
3. Convolution layer (36, 5, 5)
4. Convolution layer (48, 5, 5)
5. Convolution layer (64, 3, 3)
6. Convolution layer (64, 3, 3)
7. Convolution layer (64, 3, 3)
9. Dense layer(100)
10. Dense layer(50)
11. Dense layer(10)
12. Dense layer(1)


Here is a visualization of the architecture:

![CNN Architecture][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][image5]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it ever ended up on the side of the road. These images show what a recovery looks like starting from the right side of the road:

![1][image8]
![2][image9]
![3][image10]
![4][image11]

To augment the data set, I also flipped images and angles thinking that this would give the model more 'insight' about the pattern between the image and the steering angle. For example, here is an image that has then been flipped:

![Normal Image][image6]
![Flipped Image][image7]

I also used the left and the right camera images mounted on the vehicle. To pair those images with some steering angle, I came up with below formula consisting of a constant and a trigonometric part:

![Left formula][image3]

![Right formula][image4]

So, according to the formula,
1. If CENTER = -1, trigonometric part is -1.
2. If CENTER = 0, trigonometric part is 0.
3. If CENTER = 1, trigonometric part is 1.


I collected 12871 images taken from the center of the vehicle. By augmenting the data and using the left&right camera photos as well:

Left, Right Images = 12871 * 2

Flipped version of all images = 12871 * 3

I ended up with 77226 data points. I then preprocessed this data by cropping 70 pixels from top and 25 pixels from the bottom to remove irrelevant parts of the image. Also, I normalized the data points by getting the mean closer to 0 as 

`x = x/255.0 - 0.5`.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 in my case. Greater numbers have resulted in overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
