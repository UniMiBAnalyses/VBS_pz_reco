'''
Programme to compare the network with the analytical way. 

			Command line in order to compile from shell:
 			python analytical_comparison.py -t 'ewk_signal_withreco_truth.npy' -i 'ewk_input.npy' -p "/home/christian/Scrivania/tesi/Risultati per tesi/rete_scelta/mse_three_hidden_layer_400"

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
parser.add_argument('-i', '--input', type=str, required=True, help="Inserire il file di input")
parser.add_argument('-t', '--truth', type=str, required=True, help="Inserire il file dei valori veri")
parser.add_argument('-p', '--predictions', type=str, required=True, help="Inserire la directory in cui trovare le predizioni")
parser.add_argument('-ev', '--evaluate', action="store_true")

args = parser.parse_args()

#   Definitioni of useful plotting function

def histo_plotter_analytical(analytical):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(analytical, bins=300, range = [-250,250])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ p_{z, analytical}^{\nu} $', size = 35)
    plt.ylabel(r'Counts', size = 35)
    plt.show()	
    fig.savefig(args.predictions + '/histogram_plot_analytical.pdf', bbox_inches='tight')


def histo_plotter_true(true):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(true, bins=300, range = [-250,250])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ p_{z, true}^{\nu} $', size = 35)
    plt.ylabel(r'Counts', size = 35)
    plt.show()	
    fig.savefig(args.predictions + '/histogram_plot_true.pdf', bbox_inches='tight')

def histo_plotter_prediction(true):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(true, bins=300, range = [-250,250])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ p_{z, true}^{\nu} $', size = 35)
    plt.ylabel(r'Counts', size = 35)
    plt.show()	
    fig.savefig(args.predictions + '/histogram_plot_prediction_pz.pdf', bbox_inches='tight')

def scatter_plotter(analytical, true):
    fig = plt.figure(figsize=(15,8))
    plt.plot(analytical,true, 'o', label = "analytical")
    top = max(analytical)
    bottom = min(analytical)
    
    #Plot of the bisector, the line in which the poinst must be in the neighborhood
    plt.plot([bottom,top],[bottom,top], "r--", color = "deepskyblue", label = "Perfect NN")
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$p_{z}^{analytical}$', size = 35) 
    plt.ylabel(r'$p_{z}^{true}$', size = 35) 
    plt.legend(loc="best", prop={'size': 15})
    plt.show()
    plt.ioff()
    fig.savefig(args.predictions+ '/scatter_plot_analytical_true.pdf', bbox_inches='tight')

def plot_2d_histo_1(analytical, predictions):  
    

    fig = plt.figure()
    ax = fig.add_subplot(111)

    H = ax.hist2d(analytical, predictions, bins=200, range = [[-250,250],[-250,250]] , cmap = "Blues")

    fig.colorbar(H[3], ax=ax, shrink=0.8, pad=0.01, orientation="vertical")
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$p_{z}^{predicted}$', size = 35) 
    plt.ylabel(r'$p_{z}^{true}$', size = 35) 
    plt.show()
    fig.savefig(args.predictions+ '/analytical_vs_prediction_histogram2d_plot_pz.pdf', bbox_inches='tight')
    
def plot_2d_histo_2(analytical, true):  
    

    fig = plt.figure()
    ax = fig.add_subplot(111)

    H = ax.hist2d(analytical, true, bins=200, range = [[-250,250],[-250,250]] , cmap = "Blues")

    fig.colorbar(H[3], ax=ax, shrink=0.8, pad=0.01, orientation="vertical")
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$p_{z}^{analytical}$', size = 35) 
    plt.ylabel(r'$p_{z}^{true}$', size = 35) 
    plt.show()
    fig.savefig(args.predictions+ '/analytical_vs_true_histogram2d_plot_pz.pdf', bbox_inches='tight')

def histo_plotter_estimator_pz_predicted(estimator):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(estimator, bins=300, range = [-1.5,1.5])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ \frac{p_{z,\nu}^{predicted}-p_{z,\nu}^{true}}{p_{z,\nu}^{true}} $', size = 35)
    plt.ylabel(r'Counts', size = 35)
    plt.show()	
    fig.savefig(args.predictions + '/histogram_plot_predicted_pz_estimator.pdf', bbox_inches='tight')

def histo_plotter_estimator_pz_analytical(estimator):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(estimator, bins=300, range = [-1.5,1.5])
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$ \frac{p_{z,\nu}^{analytic}-p_{z,\nu}^{true}}{p_{z,\nu}^{true}} $', size = 35)
    plt.ylabel(r'Counts', size = 35)
    plt.show()	
    fig.savefig(args.predictions + '/histogram_plot_analytical_pz_estimator.pdf', bbox_inches='tight')


#>>>>>>>>>>>> CORE OF THE PROGRAMME <<<<<<<<<<<<<<<<<<

#	Opening of the dataset, I'll take only the test dataset to compare it with neural network predictions

ewk_truth = np.load(args.truth)
p_nu_z_true = ewk_truth[200000:300000,4]
p_nu_z_analytic = ewk_truth[200000:300000,8]



#   Importing predictions
p_nu_z_predicted = np.loadtxt(args.predictions+ '/predictions.txt', delimiter=',')

#   Some useful plots
histo_plotter_analytical(p_nu_z_analytic)
histo_plotter_true(p_nu_z_true)
histo_plotter_prediction(p_nu_z_predicted)
scatter_plotter(p_nu_z_analytic,p_nu_z_true)
plot_2d_histo_1(p_nu_z_analytic,p_nu_z_predicted)
plot_2d_histo_2(p_nu_z_analytic,p_nu_z_true)


#   Computation of the efficiecy of the algorithm

efficiency_p_z_nu_analytic_reco = (abs(p_nu_z_analytic-p_nu_z_true))/(p_nu_z_true)
efficiency_p_z_nu_predicted_reco = (abs(p_nu_z_predicted-p_nu_z_true))/(p_nu_z_true)

efficiency_p_z_nu_analytic_reco = efficiency_p_z_nu_analytic_reco.mean()
efficiency_p_z_nu_predicted_reco = efficiency_p_z_nu_predicted_reco.mean()

estimator1 = (p_nu_z_predicted-p_nu_z_true)/(p_nu_z_true)
estimator2 = (p_nu_z_analytic-p_nu_z_true)/(p_nu_z_true)


histo_plotter_estimator_pz_predicted(estimator1)
histo_plotter_estimator_pz_analytical(estimator2)


#   Saving useful values

if not args.evaluate:
    print(">>>>>>>>> SAVING HYPERPARAMETERS >>>>>>>>")
    f = open(args.predictions + "/p_z_reco.txt", "w")
    f.write("efficiency_p_z_nu_analytic_reco: {0}\n".format(efficiency_p_z_nu_analytic_reco))
    f.write("efficiency_p_z_nu_predicted_reco: {0}\n".format(efficiency_p_z_nu_predicted_reco))
    f.close()