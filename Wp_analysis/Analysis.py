import pyhf
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from pyhf.contrib.viz import brazil
# import json

def statisticalAnalysis(Data, signal, bkg, bkgerr):
    #Creamos el modelo
    model = pyhf.simplemodels.uncorrelated_background(signal = list(signal), bkg = list(bkg), bkg_uncertainty = list(bkgerr))

    #Calculamos el mu observado y esperado
    observations = np.concatenate((Data, model.config.auxdata))
    poi_values = np.linspace(0.01, 2, 100)
    obs_limit, exp_limits, (scan, results) = pyhf.infer.intervals.upper_limits.upper_limit(
        observations, model, poi_values, level=0.05, return_results=True
    )
    print(f"Upper limit (obs): μ = {obs_limit:.4f}")
    print(f"Upper limit (exp): μ = {exp_limits[2]:.4f}")

    #p-value
    pvalue = pyhf.infer.hypotest(
        1, observations, model, return_expected_set=True, return_tail_probs = True
    )[1][0]
    
    return [obs_limit, exp_limits[2], pvalue]
    
#Data mu channel
df_mu_1 = pd.read_csv("./data/M_T_mu.csv", skiprows=83, nrows=70)
df_mu_2 = pd.read_csv("./data/M_T_mu.csv", skiprows=157, nrows=70)
df_mu_3 = pd.read_csv("./data/M_T_mu.csv", skiprows=231, nrows=70)
df_mu_4 = pd.read_csv("./data/M_T_mu.csv", skiprows=9, nrows=70)

MT_mu = df_mu_1["Transverse mass, $M_T$ [GeV]"]

bkg_mu = df_mu_1["SM"]
bkg_err_mu = df_mu_1["stat+syst uncertainty +"]

signal_mu_Wp_3_8 = df_mu_2["WPRIME 3.8 TeV"]
signal_mu_Wp_5_6 = df_mu_3["WPRIME 5.6 TeV"]

Data_mu = df_mu_4["Data"]

resultsS_3_8 = statisticalAnalysis(Data_mu, signal_mu_Wp_3_8, bkg_mu, bkg_err_mu)