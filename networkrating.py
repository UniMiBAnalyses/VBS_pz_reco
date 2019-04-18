'''
Programme to rate the network in order to find the best network

			Command line in order to compile from shell:
 			python networkrating.py -t 'ewk_truth.npy' -e "/home/christian/Scrivania/tesi/Risultati per tesi/rete_scelta/mse_three_hidden_layer_400"
Christian Uccheddu
'''



#	Import of useful libraries


import pylab as pl
import numpy as np  
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt 
import matplotlib as mp
import itertools
from array import array
from scipy import interp



#>>>>>>>>>>>> FUNCTIONS USED IN THE PROGRAMME <<<<<<<<<<<<<<<<<<

#	Initialization of the parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--truth', type=str, required=True, help="Inserire il file dei valori veri")
parser.add_argument('-e', '--estimator', type=str, required=True, help="Inserire la directory in cui trovare gli estimatori")
parser.add_argument('-ev', '--evaluate', action="store_true")




args = parser.parse_args()

def gaussian_histo(x, mu, sig):
    y =[]
    y = 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power(x-mu,2.)/2*np.power(sig,2.))
    y = np.concatenate(y)
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    plt.plot(x,y,'o')
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ estimator = {\frac{p_{z,i}^{true}-p_{z,i}^{predict}}{p_{z,i}^{true}}}$', size = 35)
    plt.ylabel(r'$P(estimator)$', size = 35)
    plt.show()	
    fig.savefig(args.estimator+ '/gauss_interpolation.pdf', bbox_inches='tight', dpi = 50)

def histo_plotter_first_estimator(estimator):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(estimator, bins=300, range = [-30,30])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ {\frac{p_{z,i}^{true}-p_{z,i}^{predict}}{p_{z,i}^{true}}}$', size = 35)
    plt.ylabel(r'Counts', size = 35)
    plt.show()	
    fig.savefig(args.estimator+ '/histogram_plot_estimator1.pdf', bbox_inches='tight', dpi = 50)

def histo_plotter_second_estimator(estimator):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(estimator, bins=300, range = [-30,30])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ {p_{z,i}^{true}-p_{z,i}^{predict}}$', size = 35)
    plt.ylabel(r'Counts', size = 35)
    plt.show()	
    fig.savefig(args.estimator+ '/histogram_plot_estimator2.pdf', bbox_inches='tight', dpi = 50)

def plotter_1(estimator,quantity):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    plt.plot(estimator,quantity,'o')
    plt.axis([-100,100,-1500,1500])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ estimator$', size = 35)
    plt.ylabel(r'$p_{z}^{t}$', size = 35)
    plt.show()	
    fig.savefig(args.estimator+ '/plotter_1.pdf', bbox_inches='tight', dpi = 50)

def plotter_2(estimator,quantity):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    plt.plot(estimator,quantity,'o')
    plt.axis([-100,100,0,1500])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ estimator$', size = 35)
    plt.ylabel(r'$ \left|p_{z}^{t}\right|$', size = 35)
    plt.show()	
    fig.savefig(args.estimator+ '/plotter_2.pdf', bbox_inches='tight', dpi = 50)

def plotter_3(estimator,quantity):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    plt.plot(estimator,quantity,'o') 
    plt.axis([-100,100,0,1500])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ estimator$', size = 35)
    plt.ylabel(r'$p_{z}^{t}$', size = 35)
    plt.show()	
    fig.savefig(args.estimator+ '/plotter_3.pdf', bbox_inches='tight', dpi = 50)

def asimmetria(x,mu):
    n = len(x)
    asimmetria = np.sum(1/n*np.power(x-mu,3.))/np.power(np.sum(1/n*(np.power(x-mu,2.))),1.5)
    return asimmetria

    
def curtosi(x,mu):
    n = len(x)
    curtosi = np.sum(1/n*np.power(x-mu,4.))/(np.sum(1/n*(np.power(x-mu,2.))))
    return curtosi




#>>>>>>>>>>>> CORE OF THE PROGRAMME <<<<<<<<<<<<<<<<<<

#	Opening of the dataset


ewk_truth = np.load(args.truth)
ewk_truth_train = ewk_truth[0:100000,4].reshape(-1, 1)
ewk_truth_test = ewk_truth[200000:300000,4].reshape(-1, 1)

#   Loading of all the numpyarray

predictions = np.loadtxt(args.estimator+'/predictions.txt', delimiter=',')


#   Evaluating useful parameters

predictions = predictions.reshape(-1,1)
pos_predictions =  abs(predictions)

#   Evaluating first estimator
estimator1 = (ewk_truth_test - predictions)/ewk_truth_test
estimator1_mean = estimator1.mean()
estimator1_std =  estimator1.std() 
estimator1_asimmetria = asimmetria(estimator1,estimator1_mean)
estimator1_curtosi = curtosi(estimator1,estimator1_mean)

#   Evaluating second estimator

estimator2 = ewk_truth_test - predictions
estimator2_mean = estimator2.mean()
estimator2_std =  estimator2.std() 
estimator2_asimmetria = asimmetria(estimator2,estimator2_mean)
estimator2_curtosi = curtosi(estimator2,estimator2_mean)

#   Some useful plots
histo_plotter_first_estimator(estimator1)
#gaussian_histo(estimator1, estimator1_mean, estimator1_std)
histo_plotter_second_estimator(estimator2)
#gaussian_histo(estimator2, estimator2_mean, estimator2_std)
plotter_1(estimator1,predictions)
plotter_2(estimator1,pos_predictions)


#   Saving of useful values

if not args.evaluate:    
    print(">>>>>>>>> SAVING HYPERPARAMETERS >>>>>>>>")
    f = open(args.estimator + "/configs_estimator.txt", "w")
    f.write("estimator_mean: {0}\n".format(estimator1_mean))
    f.write("estimator_std: {0}\n".format(estimator1_std))
    f.write("estimator_asimmetria: {0}\n".format(estimator1_asimmetria))
    f.write("estimator_curtosi: {0}\n".format(estimator1_curtosi))
    f.write("estimator_mean: {0}\n".format(estimator2_mean))
    f.write("estimator_std: {0}\n".format(estimator2_std))
    f.write("estimator_asimmetria: {0}\n".format(estimator2_asimmetria))
    f.write("estimator_curtosi: {0}\n".format(estimator2_curtosi))
    f.close()
