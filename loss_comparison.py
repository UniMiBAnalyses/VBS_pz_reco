'''
Programm to compare the loss function for the same network for different learning rate. 

			Command line in order to compile from shell:
 			python loss_comparison.py -lr "/home/christian/Scrivania/tesi/Risultati per tesi/loss_comparison" -n "/home/christian/Scrivania/tesi/Risultati per tesi/neurons_comparison" -l "/home/christian/Scrivania/tesi/Risultati per tesi"

Christian Uccheddu
'''



#	Import of useful libraries

import ROOT as r
import numpy as np
import pandas as pd
import matplotlib as mp
import argparse
import matplotlib.pyplot as plt


#>>>>>>>>>>>> FUNCTIONS USED IN THE PROGRAMME <<<<<<<<<<<<<<<<<<

#	Initialization of the parser arguments
parser = argparse.ArgumentParser()

parser.add_argument('-lr', '--learningrate', type=str, required=True, help="Inserire la directory in cui salvare i plot di learning rate")
parser.add_argument('-n', '--neurons', type=str, required=True, help="Inserire la directory in cui salvare i plot di neurons")
parser.add_argument('-l', '--layer', type=str, required=True, help="Inserire la directory in cui salvare i plot di layer")


args = parser.parse_args()


def loss_comparison_learning_rate(loss_1,loss_2, loss_3):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    plt.plot(d1["loss"], label= 'learning rate' r'$ = 10^{-3}$')
    plt.plot(d2["loss"], label='learning rate' r'$ = 10^{-4}$')
    plt.plot(d3["loss"], label='learning rate' r'$ = 10^{-5}$')
    plt.legend(fontsize = 20)
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$epoch$', size = 35)
    plt.ylabel(r'$loss$', size = 35)
    plt.show()	
    fig.savefig(args.learningrate+ '/loss_function_learning_rate_comparison.pdf', bbox_inches='tight', dpi = 50)

def loss_comparison_neurons(loss_4,loss_5, loss_6,loss_7,loss_8,loss_9):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    plt.plot(d4["loss"], label= '400 neurons per layer')
    plt.plot(d5["loss"], label='300 neurons per layer')
    plt.plot(d6["loss"], label='200 neurons per layer')
    plt.plot(d7["loss"], label='100 neurons per layer')
    plt.plot(d8["loss"], label='50 neurons per layer')
    plt.plot(d9["loss"], label='10 neurons per layer')
    plt.legend(fontsize = 20)
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$epoch$', size = 35)
    plt.ylabel(r'$loss$', size = 35)
    plt.show()	
    fig.savefig(args.neurons+ '/loss_function_neurons_comparison.pdf', bbox_inches='tight', dpi = 50)


def loss_comparison_layer(loss_1,loss_2):
    fig = plt.figure(figsize=(10,5)) 
    ax = fig.add_subplot(111)
    plt.plot(d1["loss"], label=  r'$ 3 $' ' Strati')
    plt.plot(d2["loss"], label= r'$ 4 $' ' Strati' )
    plt.legend(fontsize = 20)
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel(r'$epoch$', size = 35)
    plt.ylabel(r'$loss$', size = 35)
    plt.show()	
    fig.savefig(args.learningrate+ '/loss_function_layer_comparison.pdf', bbox_inches='tight', dpi = 50)




#>>>>>>>>>>>> CORE OF THE PROGRAMME <<<<<<<<<<<<<<<<<<

#   Opening of loss function used with different learning rate

d1 = pd.read_csv(args.learningrate+ "/mse_three_hidden_layer_3/training.log")
d2 = pd.read_csv(args.learningrate+ "/mse_three_hidden_layer_4/training.log")
d3 = pd.read_csv(args.learningrate+ "/mse_three_hidden_layer_5/training.log")


d4 = pd.read_csv(args.neurons+"/mse_three_hidden_layer_400/training.log")
d5 = pd.read_csv(args.neurons+"/mse_three_hidden_layer_300/training.log")
d6 = pd.read_csv(args.neurons+"/mse_three_hidden_layer_200/training.log")
d7 = pd.read_csv(args.neurons+"/mse_three_hidden_layer_100/training.log")
d8 = pd.read_csv(args.neurons+"/mse_three_hidden_layer_50/training.log")
d9 = pd.read_csv(args.neurons+"/mse_three_hidden_layer_10/training.log")

d10 = pd.read_csv(args.layer+"/mse_three_hidden_layer_15/training.log")
d11 = pd.read_csv(args.layer+"/mse_four_hidden_layer_15/training.log")


#   Some useful plots
loss_comparison_learning_rate(d1,d2,d2)
loss_comparison_neurons(d4,d5,d6,d7,d8,d9)
loss_comparison_layer(d10,d11)
