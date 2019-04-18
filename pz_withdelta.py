'''
Programme to evaluate the mass of the W boson 

			Command line in order to compile from shell:
 			python pz_withdelta.py -t "ewk_signal_withreco_withdelta_truth.npy"  -p "/home/christian/Scrivania/tesi/Risultati per tesi/rete_scelta/mse_three_hidden_layer_400"

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
from array import array



#>>>>>>>>>>>> FUNCTIONS USED IN THE PROGRAMME <<<<<<<<<<<<<<<<<<

#	Initialization of the parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--truth', type=str, required=True, help="Inserire il file dei valori veri")
parser.add_argument('-p', '--predictions', type=str, required=True, help="Inserire la directory in cui trovare le predizioni")
parser.add_argument('-ev', '--evaluate', action="store_true")

args = parser.parse_args()


def histo_plotter_estimator_pz_analytical_pos(estimator):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(estimator, bins=300, range = [-1.5,1.5])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ \frac{p_{z,\nu}^{analytical}-p_{z,\nu}^{true}}{p_{z,\nu}^{true}} $', size = 35)
    plt.ylabel(r'Counts', size = 35)
    plt.show()	
    fig.savefig(args.predictions + '/histogram_plot_analytical_pz_estimator_positive_delta.pdf', bbox_inches='tight')

def histo_plotter_estimator_pz_analytical_neg(estimator):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(estimator, bins=300, range = [-1.5,1.5])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ \frac{p_{z,\nu}^{analytical}-p_{z,\nu}^{true}}{p_{z,\nu}^{true}} $', size = 35)
    plt.ylabel(r'Counts', size = 35)
    plt.show()	
    fig.savefig(args.predictions + '/histogram_plot_analytical_pz_estimator_negative_delta.pdf', bbox_inches='tight')


#>>>>>>>>>>>> CORE OF THE PROGRAMME <<<<<<<<<<<<<<<<<<

#	Opening of the dataset, I'll take only the test dataset to compare it with neural network predictions
#   Importing first the values of neutrino

ewk_truth = np.load(args.truth)

p_nu_z_analytical = ewk_truth[200000:300000,8].reshape(-1,1)
p_z_true = ewk_truth[200000:300000,4].reshape(-1,1)
delta = ewk_truth[200000:300000,9].reshape(-1,1)

p_nu_z_predicted = np.loadtxt(args.predictions+ '/predictions.txt', delimiter=',')

#   Importing the data for lepton



#   Computation of p_z in analytical way and with prediction

p_z_analytical_pos_delta = []
p_z_analytical_neg_delta = []
p_z_true_pos_delta = []
p_z_true_neg_delta = []


for i in range(200000,300000,1):
    if ewk_truth[i,9] > 0:
        p_z_analytical_pos_delta.append(ewk_truth[i,8])
        p_z_true_pos_delta.append(ewk_truth[i,4])

    else:
        p_z_analytical_neg_delta.append(ewk_truth[i,8])
        p_z_true_neg_delta.append(ewk_truth[i,4])


p_z_analytical_pos_delta = np.array(p_z_analytical_pos_delta)
p_z_analytical_neg_delta = np.array(p_z_analytical_neg_delta)
p_z_analytical_pos_delta = np.array(p_z_analytical_pos_delta).reshape(-1,1)
p_z_analytical_neg_delta = np.array(p_z_analytical_neg_delta).reshape(-1,1)
p_z_true_pos_delta = np.array(p_z_true_pos_delta)
p_z_true_neg_delta = np.array(p_z_true_neg_delta)
p_z_true_pos_delta = np.array(p_z_true_pos_delta).reshape(-1,1)
p_z_true_neg_delta = np.array(p_z_true_neg_delta).reshape(-1,1)

#   Computation of the bias for cases with positive and negative delta


estimator1 = (p_z_analytical_pos_delta-p_z_true_pos_delta)/(p_z_true_pos_delta)
estimator2 = (p_z_analytical_neg_delta-p_z_true_neg_delta)/(p_z_true_neg_delta)


histo_plotter_estimator_pz_analytical_pos(estimator1)
histo_plotter_estimator_pz_analytical_neg(estimator2)
