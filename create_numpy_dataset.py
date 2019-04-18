import ROOT as R 
from VBSAnalysis.EventIterator import EventIterator

import numpy as np  
import argparse
import sys

if len(sys.argv) < 3 :
    print("args:  input_root | output name")

input_file = sys.argv[1]
output = sys.argv[2]

f = R.TFile(input_file, "read")

criteria = [
    ("log", 10000, "no cuts events"),
    ("pt_min_lepton", 20.),
    ("eta_max_lepton", 3.),
    ("min_val", "MET", 20),
    ("pt_min_jets", 30),
    ("min_njets", 4),
    ("log", 10000, "passing cut events")
]

eiter = EventIterator(f, criteria, treename="tree")

dataset_input = []
dataset_truth = []
# lepton flavour (1 electron, 0 muon) | E_l | px_l | py_l | pz_l | MET | MET_phi
line_input = []   
# massW | E_nu | px_nu| py_nu | pz_nu
line_truth = []

for event in eiter:
    if event.flavour == 1:
        #electron
        line_input  = [
            1,  event.electron.E(), event.electron.Px(), event.electron.Py(),
            event.electron.Pz(), event.MET, event.MET_phi
        ]
        line_truth = [
            event.W_el.M(), event.el_neutr.E(),event.el_neutr.Px(),
            event.el_neutr.Py(),event.el_neutr.Pz()
        ]
    elif event.flavour == 0:
        #muon
        line_input  = [
            0,  event.muon.E(), event.muon.Px(), event.muon.Py(),
            event.muon.Pz(), event.MET, event.MET_phi
        ]
        line_truth = [
            event.W_mu.M(), event.mu_neutr.E(),event.mu_neutr.Px(),
            event.mu_neutr.Py(),event.mu_neutr.Pz()
        ]

    dataset_input.append(line_input)
    dataset_truth.append(line_truth)

# Save datasets on disk
np.save(output +"_input.npy", dataset_input)
np.save(output +"_truth.npy", dataset_truth)