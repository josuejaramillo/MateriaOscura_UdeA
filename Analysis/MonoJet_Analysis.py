import awkward as ak
import pandas as pd
import numpy as np
from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents import NanoAODSchema, DelphesSchema
import mplhep as hep
import matplotlib.pyplot as plt
import seaborn as sns
import DM_HEP_AN as dm
from math import pi
hep.style.use("CMS")


BM = "toy"
my = 750
mx = 10
gdm = 1
#Set the number of jets you want to take into account in your analysis
n_jets = 4
n_lep = 4


fname = "/home/tomas/Documents/MG5_aMC_v2_9_15/MonoJet_DM_UdeA/Events/run_25/tag_1_delphes_events.root"
output_name = f'/home/tomas/Documents/UdeA_Results/DM/pt_miss_histogram_monojet_BM_{BM}_my_{my}_GeV_mDM_{mx}_GeV_gDM_{gdm}.csv'
x_sec = 63.29 * 1e-12


tree_test = dm.Converter(fname)
tree_test.generate(jet_elements = n_jets, e_mu_elements = n_lep)
df = tree_test.df


#Cut p_T^{miss} > 200 GeV
df_cut = df[df['missinget_met'] > 200]

#Cut p_T(j0) > 100 GeV
df_cut = df_cut[df_cut['jet_pt0'] > 100]

#Cut eta(j0) > 2.5 
df_cut = df_cut[np.abs(df_cut['jet_eta0']) < 2.5]

#Cut HT > 110 GeV
#HT := sum(jet_pt) over all the jets per event
#Sumar solo por los que tengan mayor pt que 20
df_cut = df_cut[np.sum(df_cut[[f"jet_pt{i}" for i in range(n_jets)]], axis = 1) > 110]


#Select number of b, taus, and leptons
df_cut['n_taus'] = df_cut.apply(dm.veto_tag, axis = 1, args = ["tau", n_jets])
df_cut['n_bs'] = df_cut.apply(dm.veto_tag, axis = 1, args = ["b", n_jets])
df_cut['n_ele'] = df_cut.apply(dm.veto_tag, axis = 1, args = ["electron", n_lep])
df_cut['n_mu'] = df_cut.apply(dm.veto_tag, axis = 1, args = ["muon", n_lep])

df_cut = df_cut[df_cut['n_taus'] < 1]
df_cut = df_cut[df_cut['n_bs'] < 1]
df_cut = df_cut[df_cut['n_ele'] < 1]
df_cut = df_cut[df_cut['n_mu'] < 1]




df_cut["deltaphi_jet0_met"] = df_cut.apply(dm.DeltaPhi, args = ("jet_phi0", "missinget_phi"), axis = 1)
df_cut["deltaphi_jet1_met"] = df_cut.apply(dm.DeltaPhi, args = ("jet_phi1", "missinget_phi"), axis = 1)
df_cut["deltaphi_jet2_met"] = df_cut.apply(dm.DeltaPhi, args = ("jet_phi2", "missinget_phi"), axis = 1)
df_cut["deltaphi_jet3_met"] = df_cut.apply(dm.DeltaPhi, args = ("jet_phi3", "missinget_phi"), axis = 1)

df_copy = df_cut.copy()


df_cut = df_cut[np.abs(df_cut["deltaphi_jet0_met"]) > 0.5]
df_cut = df_cut[(np.abs(df_cut["deltaphi_jet1_met"]) > 0.5) | (df_cut["deltaphi_jet1_met"].isna())]
df_cut = df_cut[(np.abs(df_cut["deltaphi_jet2_met"]) > 0.5) | (df_cut["deltaphi_jet2_met"].isna())]
df_cut = df_cut[(np.abs(df_cut["deltaphi_jet3_met"]) > 0.5) | (df_cut["deltaphi_jet3_met"].isna())]

#Calculation of the number of events
Luminosity = 12.9
n_mc_ev = df.shape[0]
n_ex_ev = x_sec * Luminosity / (1e-15)
w = n_ex_ev / n_mc_ev

#Creation of the Histogram
bins = np.concatenate([np.arange(200, 330, 30),
                       np.arange(350, 560, 40),
                       np.arange(590, 800, 50),
                       np.arange(840, 1030, 60),
                       np.arange(1090, 1170, 70),
                       np.array([2000])])

counts, ptmiss = np.histogram(df_cut.missinget_met, bins)
counts = counts * w
error = w * counts ** 0.5
di_missing_et = {f"{ptmiss[i]} - {ptmiss[i+1]}" :
                 counts[i] for i in range(len(bins)-1)}



df_jessica = pd.DataFrame(data = [di_missing_et.keys(), di_missing_et.values(), error]).T
df_jessica.columns = ["bins", "counts", "error"]
df_jessica['bins_lower'] = ptmiss[:-1]
df_jessica.to_csv(output_name)
