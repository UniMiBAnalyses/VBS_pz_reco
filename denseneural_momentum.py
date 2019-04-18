'''
Programme to evaluate pz_nu by a regression

			Command line in order to compile from shell:
 			python denseneural_momentum.py -i 'ewk_input.npy' -t 'ewk_truth.npy' -ms "mse_three_hidden_layer_400" -o "/home/christian/Scrivania/tesi/Risultati per tesi/rete_scelta_mse_three_hidden_layer_400"


Christian Uccheddu
'''



#	Import of useful libraries

import pylab as pl
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
from models import *
from keras.callbacks import *
from sklearn.externals import joblib
from scipy.stats import norm
import scipy.stats


#>>>>>>>>>>>> FUNCTIONS USED IN THE PROGRAMME <<<<<<<<<<<<<<<<<<

#	Initialization of the parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help="Inserire il file di input")
parser.add_argument('-t', '--truth', type=str, required=True, help="Inserire il file di input")
parser.add_argument('-e', '--epoch', type=int, required=False, default=1000)
parser.add_argument('-ie', '--initial_epoch', type=int, required=False, default=0)
parser.add_argument('-bs', '--batch-size', type=int, required=False, default=500)
parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=1e-4)
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-p', '--patience', type=str, required=False, default="0.001:50",
                    help="Patience format:  delta_val:epochs")
parser.add_argument('-m', '--model', type=str, required=False)
parser.add_argument('-ev', '--evaluate', action="store_true")
parser.add_argument('-sst', '--save-steps', action="store_true")
parser.add_argument('-dr', '--decay-rate', type=float, required=False, default=0)
parser.add_argument('-ms', '--model-schema', type=str, required=True, help="Model structure")


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


def scatter_plotter(pred, y):
    #Plotting
    fig = plt.figure(figsize=(15,8))
    plt.plot(pred,y, 'o', label = "Predicted")
    top = max(pred)
    bottom = min(pred)
    
    #Plot of the bisector, the line in which the poinst must be in the neighborhood
    plt.plot([bottom,top],[bottom,top], "r--", color = "deepskyblue", label = "Perfect NN")
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$p_{z}^{predicted}$', size = 35) 
    plt.ylabel(r'$p_{z}^{true}$', size = 35) 
    plt.legend(loc="best", prop={'size': 15})
    plt.show()
    plt.ioff()
    fig.savefig(args.output+ '/scatter_plot.png', bbox_inches='tight')

def acc_plotter(history):
    fig = plt.figure(figsize=(10,5))
    plt.plot(history.history['acc'], label = "train_accuracy")
    plt.plot(history.history['val_acc'], label="val_accuracy")
    plt.legend(['acc_train', 'acc_test'], loc='lower right')
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.show()
    fig.savefig(args.output+ '/accumulation_plot.pdf', bbox_inches='tight')

def loss_plotter(history):
    fig = plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label = "train_loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.legend(['loss_train', 'loss_test'], loc='upper right', fontsize = 20)
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel('epochs', size = 35)
    plt.ylabel('loss', size = 35)
    plt.show()
    fig.savefig(args.output+ '/loss_plot.pdf', bbox_inches='tight')

def histo_plotter(prediction):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(predictions, bins=300, range = [-250,250])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$p_{z,predicted}^{nu}$', size = 35)
    plt.ylabel('counts', size = 35)
    plt.show()	
    fig.savefig(args.output+ '/predictions_histogram_plot.pdf', bbox_inches='tight')

def plot_2d_hist_1(y, pred):  
    

    fig = plt.figure()
    ax = fig.add_subplot(111)

    H = ax.hist2d(y, pred, bins=200, range = [[-250,250],[-250,250]] , cmap = "Blues")

    fig.colorbar(H[3], ax=ax, shrink=0.8, pad=0.01, orientation="vertical")
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$p_{z}^{predicted}$', size = 35) 
    plt.ylabel(r'$p_{z}^{true}$', size = 35) 
    plt.show()
    fig.savefig(args.output+ '/predictions_histogram2d_plot.pdf', bbox_inches='tight')






#>>>>>>>>>>>> CREATIONS OF THE DATASET <<<<<<<<<<<<<<<<<<

#	Open of the directory in which results will be written

mkdir_p(args.output)

#	Let's remember that the dataset is divided in the following way:
#	lepton flavour (1 electron, 0 muon) | E_l | px_l | py_l | pz_l | MET | MET_phi saved in 'ewk_input.py'
#	massW | E_nu | px_nu| py_nu | pz_nu saved in 'ewk_truth.py'
#	Opening of the dataset

ewk_input = np.load(args.input)
ewk_truth = np.load(args.truth)


#	Import of 10000 entries of the nparray 

# 100000 entries for train

ewk_input_train = ewk_input[0:100000,:]
ewk_truth_train = ewk_truth[0:100000,4].reshape(-1, 1)


# 100000 entries for validation

ewk_input_validation = ewk_input[100000:200000,:]
ewk_truth_validation = ewk_truth[100000:200000,4].reshape(-1, 1)

#Stampo la prima riga e le dimensioni dei numpyarray in modo da vedere se sono stati importati nella maniera corretta
print(ewk_input[0,:])
print(ewk_input.shape)
print(ewk_truth.shape)


# 100000 entries for test
ewk_input_test = ewk_input[200000:300000,:]
ewk_truth_test = ewk_truth[200000:300000,4].reshape(-1, 1)

#	z-scale of the dataset:


#	Definition of the scaler
scaler_input = StandardScaler()
scaler_truth = StandardScaler()


scaler_input.fit(ewk_input_train)
scaler_truth.fit(ewk_truth_train)

input_train = scaler_input.transform(ewk_input_train)
truth_train = scaler_truth.transform(ewk_truth_train)
input_validation = scaler_input.transform(ewk_input_validation)
truth_validation = scaler_truth.transform(ewk_truth_validation)
input_test = scaler_input.transform(ewk_input_test)
truth_test = ewk_truth_test

print(input_test[0,:])

#	Saving the scaler models
joblib.dump(scaler_input, args.output+ "/scaler_input.pkl")
joblib.dump(scaler_truth, args.output+ "/scaler_truth.pkl")

#	Import the model

if args.model == None:
    model = getModel(args.model_schema, input_validation.shape[1])
else:
    print(">>> Loading model ({0})...".format(args.model))
    model = load_model(args.model)

if not args.evaluate:
    # Training procedure
    if args.save_steps:
        auto_save = ModelCheckpoint(args.output+"/current_model_epoch{epoch:02d}", monitor='val_loss',
                    verbose=1, save_best_only=False, save_weights_only=False,
                    mode='auto', period=1)
    else:
        auto_save = ModelCheckpoint(args.output +"/current_model", monitor='val_loss',
                    verbose=1, save_best_only=True, save_weights_only=False,
                    mode='auto', period=2)

    min_delta = float(args.patience.split(":")[0])
    p_epochs = int(args.patience.split(":")[1])
    early_stop = EarlyStopping(monitor='val_loss', min_delta=min_delta,
                               patience=p_epochs, verbose=1)

    def reduceLR (epoch):
        return args.learning_rate * (1 / (1 + epoch*args.decay_rate))

    lr_sched = LearningRateScheduler(reduceLR, verbose=1)

    csv_logger = CSVLogger(args.output +'/training.log')


    print(">>> Training...")

    W_val = np.ones(input_validation.shape[0])

    history = model.fit(input_train, truth_train,
                        validation_data = (input_validation, truth_validation),
                        epochs=args.epoch, initial_epoch=args.initial_epoch,
                        batch_size=args.batch_size, shuffle=True,
                        callbacks=[auto_save, early_stop, lr_sched, csv_logger])

#	Calculate predictions
predictions = model.predict(input_test,batch_size=2048)
predictions = np.concatenate(predictions)



#	Inverse scaling of the data
predictions = scaler_truth.inverse_transform(predictions)


#	Some useful plots

loss_plotter(history)
histo_plotter(predictions)
scatter_plotter(predictions, np.concatenate(truth_test))
plot_2d_hist_1(predictions, np.concatenate(truth_test))

#   Evaluating pz_nu

pz_nu = predictions.mean()
pz_nu_std = predictions.std()


#   Saving predictions numpyarray in order to evaluate estimators

np.savetxt(args.output + "/predictions.txt" , predictions)


#	Saving Hyperparameters

if not args.evaluate:
    print(">>>>>>>>> SAVING HYPERPARAMETERS >>>>>>>>")
    f = open(args.output + "/configs.txt", "w")
    f.write("epochs: {0}\n".format(args.epoch))
    f.write("model_schema: {0}\n".format(args.model_schema))
    f.write("batch_size: {0}\n".format(args.batch_size))
    f.write("learning_rate: {0}\n".format(args.learning_rate))
    f.write("decay_rate: {0}\n".format(args.decay_rate))
    f.write("patience: {0}\n".format(args.patience))
    f.write("pz_nu: {0}\n".format(pz_nu))
    f.write("pz_nu_std: {0}\n".format(pz_nu_std))
    f.close()

