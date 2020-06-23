# TFNNScore
Keras implementation of NNScore described in "Soft Computing Tools for Virtual Drug Discovery" by Daniel M Hagan and Martin T Hagan. This Python code will not give the exact same results as our original Matlab code (although they should be very similar), this is because things such as the optimizer we used in Matlab are not currently implemented in Keras/Python. I will add the original Matlab code as a branch for anyone interested. We started with our Matlab code and did our best to recreate it in Python using Keras since open-sourced Python will almost certainly outlast Matlab. Once I had this code set up, it allows you to easily increase the size of the network. Even though our initial results indicated that larger networks were not giving better results, I decided to try much, much larger networks and found that they dramatically increase the accuracy from ~80% to basically 100% accurate. This only held true for the training data though, once we tried a benchmark of diverse receptors, we did no better than this model does. This means that with the new techniques for avoidinig overfitting (batch normalization) are working quite well even with these extremely large networks. For this problem, what we need is possibly a better set of bad binders. Another solution might be to redock all of the PDB poses with vina, although this would be too time consuming for one person to do: Bounding boxes would need to be acquired for each complex, etc... Alternate solutions anyone might think of can be sent to dhagan@okstate.edu! I highly recommend you check out the TFNNScoreMod branch, where I posted the code for one of the best deep networks (6 layers with 300 neurons each). Please feel free to tinker with the code and submit any improvements or interesting things you find!

Required Packages:
Numpy,
Scipy,
Keras,
Tensorflow,
ContextLib (Context Manager)

tfnnscore2layer.py trains a model to perform virtual screening of docked molecules as described in "Soft Computing Tools for Virtual Drug Discovery". tfdata.mat has the preprocessed training and testing data. In order to run the program, put tfnnscore2layer.py and tfdata.mat in the same folder and then in the command line "python3 tfnnscore2layer.py". 
The out file showing the main results will be saved in the current directory as tfnnscoredata.txt, showing performance on both training and held out test data. 

Training takes 1-2 hours on a VM with 8 vCPU's, 30gb memory and 1 NVIDIA Tesla P100 GPU.
