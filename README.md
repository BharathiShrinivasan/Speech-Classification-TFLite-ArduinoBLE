# Speech Classification TFLite ArduinoBLE
 This is a simple ML project targeted for edge inferencing. A custom voice command set is acquired. Trained to multiple models. TF lite model of best is deployed in Arduino. Live demo.

# Project Detail

My ML project - Speech classification | Class={Forward, Reverse, Unknown/noise}

Objective :
	1. Learn complete workflow of solving an engineering problem using ML technique. And implement a run-time inferencing in Embedded system.
	
Activities carried :
>1. Own Dataset preparation
>2. Voice feature - Spectrogram. Preprocessing of >dataset.
>3. Complete training of multiple ML models (ANN, 2D >CNN, 1D CNN with no. of kernals)
>4. Finding the best model with constrain - Metrics : >accuracy && Memory < 100KB
>5. Preparing the model to quantize/prune to get tflite >model, which is deployable.
>6. Deploying in Arduino nano 33 BLE. Running >inferencing in Embedded system i.e. Ardunio nano33


### FILES RELATED

#### Google colab files - 
>1. Training_SpeechClassification_1.ipynb  : Has the main code that does the Training of different models and tflite conversion.
>2. Inferencing_SpeechClassification_VoiceInput.ipynb  : This gets a live audio -> preprocess -> get input_tensor which will be used by Arduino nano code to run inferencing in Arduino board.

#### Dataset -
MyDataset.zip has the complete self recorded speech - 900 files in total. Classes - {Forward, Reverse, Unknown/noise}

#### HardwareDeployment directory -
This has the arduino sketch used to deploy tflite into nano33 ble.

>/Sketch_trail0/Sketch_trail0.ino - is the arduino porject.
/Sketch_trail0/AudioInputTensor.h - is the input tensor / float values (64x64)
/Sketch_trail0/SC_LiteModel_toDeploy.h - is our trained model's weight,baise info in bytes.

#### Edge Impulse platform -
>HardwareDeployment/SpeechClassification_EdgeImpulse/ speechclassifier_ml-nano-33-ble-sense-v2.zip - has the edge impulse build - library for Arduino nano 33 ble
HardwareDeployment/SpeechClassification_EdgeImpulse/ nano_ble33_sense_microphone/nano_ble33_sense_microphone.ino - has the arduino project that uses the edge impulse built and does live inferencing