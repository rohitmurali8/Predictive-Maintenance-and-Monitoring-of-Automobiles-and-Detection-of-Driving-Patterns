# Predictive-Maintenance-and-Monitoring-of-Automobiles-and-Detection-of-Driving-Patterns
With the evolution of data being used for predictive modelling, this project aims to leverage
this and develop a system to improve road safety by monitoring driver behaviour and to
improve vehicular health by analysing the effects of its usage.Transportation industry has
its branches in almost all sectors and supply chain management and optimizing it is a
major concern faced by all industries.The proposed system facilitates the monitoring of a
fleet,by analysing both driver behaviour and its effects on carrier,which creates a transparent
environment for business owners.The driver fatigue and emotional state along with driving
patterns act as input parameters to the system.These driver fatigue is detected by first
detecting the facial landmarks of a driver in real time using a subset of shape prediction
model and then performing mathematical analysis like calculating eye aperture ratio for the
eyes along with monitoring of the spatial co-ordinates of the mouth region for detection of
any occurrence of a yawn and calculating its frequency. The second parameter utilised for
performing the said predictive analysis is the emotional state of a driver and its effects on
vehicular health. The real time input image of a driver is taken and given to a prediction
model which classifies the emotion into a particular class. This provides us with a tool to
evaluate driving patterns when a certain emotion is experienced by the driver. These driving
patterns are then compared to the ones practiced when a neutral emotion is experienced and
the deviation is calculated. The parameters of driver fatigue and emotional state provide
insights into the driver health and helps evaluate his/her ability to perform the action of
driving by comparing it to optimum levels of these parameters and thresholding.


This Project demonstrated the methodologies to monitor the driver and vehicular health.
A combination of data analytics, computer vision and neural networks is used to develop
this system.This model can be used by fleet service providers to better monitor their drivers
and their driving patterns to provide better quality of service along with improving road
safety. This system is proactive and not reactive and provides corrective feedback in real
time. The image of the driver was captured and the facial parameters were extracted.
These facial parameters were used to perform facial landmark detection.Once the landmark
detection was achieved, Fisher face algorithm was used to train a model to detect and
classify emotional state of the driver. The driver activity monitoring was also performed
and classified into various classes using a convolutional neural network. The predictive
maintenance was performed using parameters from the engine control unit.
The proposed system facilitates fleet management by analysing driver behaviour through
emotion and fatigue detection along with driver activity monitoring with an accuracy of 74%
and 95.69% for respective neural networks. Predictive maintenance using logistic regression
was able to perform convergence up to 97.23%. These factors play a major role in fleet
management and are paramount towards optimizing the business outcomes.
