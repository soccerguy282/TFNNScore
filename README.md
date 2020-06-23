# TFNNScore
Keras implementation of NNScore described in "Soft Computing Tools for Virtual Drug Discovery" by Daniel M Hagan and Martin T Hagan

**Note that this Python code will not give the exact same results as our original Matlab code (although they should be very similar), this is** **because things such as the optimizer we used in Matlab are not currently implemented in Keras/Python. I will add the original Matlab** **code as a branch for anyone interested. We started with our Matlab code and did our best to recreate it in Python with Keras.**

Required Packages:


tfnnscore2layer.py contains the program and tfdata.mat has the preprocessed data. All you need to do to run the program is put tfnnscore2layer.py and tfdata.mat in the same folder and then in the command line "python3 tfnnscore2layer.py". 
The out file showing the main results will be saved in the current directory as tfnnscoredata.txt.

Training takes 1-2 hours on a VM with 8 vCPU's, 30gb memory and 1 NVIDIA Tesla P100 GPU.
