'''


			Command line in order to compile from shell:
 			python test.py -i 'ewk_input.npy' -o 'ewk_truth.npy' -e 100


Christian Uccheddu
'''



#	Import of useful libraries


import numpy as np  
import argparse
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import scipy.odr as odr 
from keras.activations import relu
from keras.activations import tanh
import matplotlib.pyplot as plt 
import matplotlib as mp
import itertools
import random
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dropout
from array import array
from sklearn import metrics
from keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn import svm, datasets
from scipy import interp
import models as M
from keras.callbacks import History 
history = History()
from keras.callbacks import *



#>>>>>>>>>>>> FUNCTIONS USED IN THE PROGRAMME <<<<<<<<<<<<<<<<<<

#	Initialization of the parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help="Inserire il file di input")
parser.add_argument('-o', '--output', type=str, required=True, help="Inserire il file di input")
parser.add_argument('-e', '--initial_epoch', type=int, required=False, default=50)


args = parser.parse_args()

#	Creation and opening of the directory
def mkdir_p(mypath):
    

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
  



#	Initialization of some useful plots
#	Crates a function to fit the predicted data compared to the test data y = m*x + b
def f(B, x):

    return B[0]*x + B[1]

def scatter_plotter(y, pred):
    
    #Function used to fit the points thanks to scipy ODR.
    sxx = np.std(pred)
    syy = np.std(y)
    linear = odr.Model(f)
    mydata = odr.RealData(np.concatenate(pred), y, sx=sxx, sy=syy)
    myodr = odr.ODR(mydata, linear, beta0=[1, 0.5])
    myoutput = myodr.run()
    
    mark = dict(marker= 'o',
            color = "black", 
            fillstyle="none",
            markersize=15, 
            linewidth = 0)
    
    #Plotting
    fig = plt.figure(figsize=(15,8))
    plt.plot(pred,y, **mark, label = "Predicted")
    top = max(pred)
    bottom = min(pred)
    
    #Plot of the bisector, the line in which the poinst must be in the neighborhood
    plt.plot([bottom,top],[bottom,top], "r--", color = "deepskyblue", label = "Perfect NN")
    
    #Plot of the fit
    plt.plot(pred,f(myoutput.beta, pred), "r--", color = "m", label = "Linear inrterpolation, m=%.2f, bias=%.2f" %( myoutput.beta[0], myoutput.beta[1]))
    
    plt.xlabel('denseneural_prediction') 
    plt.ylabel('True y values') 
    plt.title("NN regression visualization, MSE: %.4f" % mean_squared_error(truth_train, pred), fontsize=20)
    plt.legend(loc="best", prop={'size': 15})
    plt.show()
    plt.savefig('scatter_plot.png')

def acc_plotter(history):
    plt.figure(figsize=(10,5))
    plt.plot(history.history['acc'], label = "train_accuracy")
    plt.plot(history.history['val_acc'], label="val_accuracy")
    plt.legend(['acc_train', 'acc_test'], loc='lower right')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.show()
    plt.savefig('accumulation_plot.png')

def loss_plotter(history):
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label = "train_loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.legend(['loss_train', 'loss_test'], loc='upper right')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    plt.savefig('loss_plot.png')

def histo_plotter(prediction):
    plt.hist(predictions, bins=100)
    plt.xlabel('W_mass')
    plt.ylabel('counts')
    plt.show()	
    plt.savefig('histogram_plot.png')

#	Makes some cut on the dataset to test it on different conditions 



#>>>>>>>>>>>> CREATIONS OF THE DATASET <<<<<<<<<<<<<<<<<<


#	Let's remember that the dataset is divided in the following way:
#	lepton flavour (1 electron, 0 muon) | E_l | px_l | py_l | pz_l | MET | MET_phi saved in 'ewk_input.py'
#	massW | E_nu | px_nu| py_nu | pz_nu saved in 'ewk_truth.py'
#	Opening of the dataset

ewk_input = np.load(args.input)
ewk_truth = np.load(args.output)


#	Import of 10000 entries of the nparray 

#definiamo i primi 4000 per train e altri 4000 validation

ewk_input_train = ewk_input[0:4000,:]
ewk_truth_train = ewk_truth[0:4000,0]
ewk_input_validation = ewk_input[4000:8000,:]
ewk_truth_validation = ewk_truth[4000:8000,0]

#Stampo la prima riga e le dimensioni dei numpyarray in modo da vedere se sono stati importati nella maniera corretta
print(ewk_input[0,:])
print(ewk_input.shape)
print(ewk_truth.shape)


#definiamo gli ultimi 2000 come holdout per testare la rete
ewk_input_test = ewk_input[8000:10000,:]
ewk_truth_test = ewk_truth[8000:10000,0]


#	Reshaping the vector for z-scale

ewk_truth_train = ewk_truth_train.reshape(-1,1)
ewk_truth_validation = ewk_truth_validation.reshape(-1,1)
ewk_truth_test = ewk_truth_test.reshape(-1,1)

#	z-scale of the dataset:



scaler = StandardScaler()
scaler.fit(ewk_input_train)
scaler.fit(ewk_truth_train)
scaler.fit(ewk_input_validation)
scaler.fit(ewk_truth_validation)
scaler.fit(ewk_input_test)
scaler.fit(ewk_truth_test)
input_train = scaler.transform(ewk_input_train)
truth_train = scaler.transform(ewk_truth_train)
input_validation = scaler.transform(ewk_input_validation)
truth_validation = scaler.transform(ewk_truth_validation)
input_test = scaler.transform(ewk_input_test)
truth_test = scaler.transform(ewk_truth_test)
print(input_test[0,:])

#	Open the directory

output_dir = "/home/christian/Scrivania/tesi/davide/neutrinoreconstruction/DeepLearning/DenseNeural/DenseNeural_results"
mkdir_p(output_dir)

#	Hyperparameters Definition
n_first_layer_neuron = 50;
n_first_layer_dropout = 0.3;

n_second_layer_neuron = 10;
n_second_layer_dropout = 0.3;

n_third_layer_neuron = 8;
n_third_layer_dropout = 0.3;

model = Sequential()
model.add(Dense(n_first_layer_neuron, input_dim=(input_train.shape[1]), init='uniform', activation='relu'))
#model.add(Dropout(n_first_layer_dropout = 0.3))
model.add(Dense(n_second_layer_neuron, init='uniform', activation='relu'))
#model.add(Dropout(n_second_layer_dropout))
model.add(Dense(n_third_layer_neuron, init='uniform', activation='sigmoid'))
#model.add(Dropout(n_third_layer_dropout))

#	The output is a Real Number so it dosen't have an activation function.

model.add(Dense(1))


#	Compile
model.compile(  loss='mean_squared_error', 
		optimizer='adam', 
		metrics=['mse'])

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=0.1, verbose=1, mode='auto')


#	Fit the model
history = model.fit(input_test, truth_test, shuffle=True, epochs=(args.initial_epoch), batch_size=10,  verbose=1, callbacks = [early_stop])


#	Calculate predictions
predictions = model.predict(input_train)
predictions = np.concatenate(predictions)



#	Inverse scaling of the data
predictions = scaler.inverse_transform(predictions)
print(predictions)

#	Some useful plots

#acc_plotter(history)
#loss_plotter(history)
histo_plotter(predictions)
scatter_plotter(predictions, ewk_truth_train)


print(">>>>>>>>> SAVING HYPERPARAMETERS >>>>>>>>")

#	Open the file to save the results

f = open(output_dir + "/DenseNeural_hyperparameters.txt", "w")
f.write("{}\n".format(args.initial_epoch))
f.write("{}\n".format(n_first_layer_neuron ))
f.write("{}\n".format(n_second_layer_neuron))
f.write("{}\n".format(n_third_layer_neuron ))
f.write("{}\n".format(first_layer_dropout ))
f.write("{}\n".format(second_layer_dropout))
f.write("{}\n".format(third_layer_dropout ))
#f.write("{}\n".format(third_layer_dropout ))

#	Closing of the file

f.close()





