#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:23:09 2020

@author: codyhuddleston
"""

import pandas as pd
import numpy as np

#%% Download Data as Pandas DataFrames
file_location = 'E:\Bring Home\SDS\\bal_iteration_Expl_180503.xlsx'
excel_sheet = 'bal(t=5)'
sds = pd.read_excel(file_location,sheet_name = excel_sheet, skiprows = 8, \
                      index_col = 'Standard Inputs and Outputs')
sds = pd.DataFrame(sds,columns = ['SDS'])
params = pd.read_excel(file_location,sheet_name = excel_sheet, nrows = 6, \
                      index_col = 'Parameter Inputs', usecols = [0,1,2,3])
drm = pd.read_excel(file_location,sheet_name = excel_sheet, skiprows = 9, \
                    nrows = 102, usecols = [7,8,9,10,11,12])

#%% Convert DataFrame Values to Variables and Matrices
bal_positiveccs = np.matrix([[params.loc['G1_']['RATIO']], \
                     [params.loc['G2_']['RATIO']], \
                     [params.loc['G3_']['RATIO']], \
                     [params.loc['G4_']['RATIO']], \
                     [params.loc['G5_']['RATIO']], \
                     [params.loc['G6_']['RATIO']]])

bal_incal = np.matrix([[params.loc['G1_']['INCAL']], \
                     [params.loc['G2_']['INCAL']], \
                     [params.loc['G3_']['INCAL']], \
                     [params.loc['G4_']['INCAL']], \
                     [params.loc['G5_']['INCAL']], \
                     [params.loc['G6_']['INCAL']]])

bal_zlos = np.matrix([[params.loc['G1_']['BUOY']], \
                     [params.loc['G2_']['BUOY']], \
                     [params.loc['G3_']['BUOY']], \
                     [params.loc['G4_']['BUOY']], \
                     [params.loc['G5_']['BUOY']], \
                     [params.loc['G6_']['BUOY']]])

bal_iccal = np.matrix([[sds.loc['BAL1G1_ICCAL']['SDS']], \
                     [sds.loc['BAL1G2_ICCAL']['SDS']], \
                     [sds.loc['BAL1G3_ICCAL']['SDS']], \
                     [sds.loc['BAL1G4_ICCAL']['SDS']], \
                     [sds.loc['BAL1G5_ICCAL']['SDS']], \
                     [sds.loc['BAL1G6_ICCAL']['SDS']]])

bal_iczero = np.matrix([[sds.loc['BAL1G1_ICZERO']['SDS']], \
                     [sds.loc['BAL1G2_ICZERO']['SDS']], \
                     [sds.loc['BAL1G3_ICZERO']['SDS']], \
                     [sds.loc['BAL1G4_ICZERO']['SDS']], \
                     [sds.loc['BAL1G5_ICZERO']['SDS']], \
                     [sds.loc['BAL1G6_ICZERO']['SDS']]])

bal_mean = np.matrix([[sds.loc['BAL1N1_MEAN']['SDS']], \
                     [sds.loc['BAL1N2_MEAN']['SDS']], \
                     [sds.loc['BAL1S1_MEAN']['SDS']], \
                     [sds.loc['BAL1S2_MEAN']['SDS']], \
                     [sds.loc['BAL1RM_MEAN']['SDS']], \
                     [sds.loc['BAL1A_MEAN']['SDS']]])

drm = np.asmatrix(drm.values)               #Convert DataFrame to Matrix

#%% Begin Balance Calcs
bal_calcorr = np.divide(bal_incal,np.subtract(bal_iccal,bal_iczero))
bal_woz = np.multiply(bal_positiveccs, np.multiply(bal_calcorr, bal_iczero))
bal = np.multiply(bal_positiveccs, np.multiply(bal_calcorr, bal_mean))
bal_dwz = np.subtract(bal_woz,bal_zlos)
bal_ddz = np.subtract(bal,bal_zlos)

#%% Balance Comp Iteration Function
def bal_iteration(drm,bal_measured):
    d0 = drm[[0,1,2,3,4,5]]
    d2 = drm[np.arange(6,102)]
    
    f0 = np.matmul(d0.transpose(),bal_measured)
    f = f0
    conv_tol = 0.0005
    max_iter = 15
    k = 0
    while k < max_iter:
        
        h16 = f
        h712 = abs(h16)
        h1318 = np.multiply(h16,h16)
        h1924 = np.multiply(h16,h712)
        h2529 = float(h16[0])*h16[slice(1,6)]
        h3033 = float(h16[1])*h16[slice(2,6)]
        h3436 = float(h16[2])*h16[slice(3,6)]
        h3738 = float(h16[3])*h16[slice(4,6)]
        h39 = float(h16[4])*h16[slice(5,6)]
        h4054 = abs(np.vstack((h2529,h3033,h3436,h3738,h39)))
        h5559 = float(h16[0])*abs(h16[slice(1,6)])
        h6063 = float(h16[1])*abs(h16[slice(2,6)])
        h6466 = float(h16[2])*abs(h16[slice(3,6)])
        h6768 = float(h16[3])*abs(h16[slice(4,6)])
        h69 = float(h16[4])*abs(h16[slice(5,6)])
        h7074 = abs(float(h16[0]))*h16[slice(1,6)]
        h7578 = abs(float(h16[1]))*h16[slice(2,6)]
        h7981 = abs(float(h16[2]))*h16[slice(3,6)]
        h8283 = abs(float(h16[3]))*h16[slice(4,6)]
        h84 = abs(float(h16[4]))*h16[slice(5,6)]
        h8590 = np.multiply(h16,np.multiply(h16,h16))
        h9196 = abs(h8590)
        H = np.vstack((h16,h712,h1318,h1924,h2529,h3033,h3436,h3738,h39,h4054,h5559,\
                       h6063,h6466,h6768,h69,h7074,h7578,h7981,h8283,h84,h8590,h9196))
        
        # Final f array
        f1 = np.subtract(f0,np.matmul(d2.transpose(),H))
        
        # Convergence Condition
        delt = np.subtract(f1,f)
        diff = abs(np.amax(delt))
        if conv_tol > diff:
            break
        # Update inputs
        f = f1
        k +=1
    return f1

#%% Balance Calcs (continued)
bal_iacp = bal_iteration(drm,bal_ddz)
bal_biascorr = bal_iteration(drm,bal_dwz)
bal_nobias = np.subtract(bal_iacp,bal_biascorr)

#%% Function Resolving Balance Loads
bal_type = 1
bal_dist = np.matrix(([4.5,4.5,3.5,3.5,0.0,0.0])).transpose()
def rslv(bal_dist,bal_nobias,bal_type):
    if bal_type == 1:
        bal_norrslvfrc = bal_nobias.item(0)+bal_nobias.item(1)
        bal_pitrslvmom = bal_dist.item(0)*bal_nobias.item(0)-bal_dist.item(1)*bal_nobias.item(1)
        bal_sidrslvfrc = bal_nobias.item(2)+bal_nobias.item(3)
        bal_yawrslvmom = bal_dist.item(2)*bal_nobias.item(2)-bal_dist.item(3)*bal_nobias.item(3)
        bal_rolrslvmom = bal_nobias.item(4)
        bal_axirslvfrc = bal_nobias.item(5)
    elif bal_type == 2:
        bal_norrslvfrc = bal_nobias.item(0)
        bal_pitrslvmom = bal_nobias.item(1)
        bal_sidrslvfrc = bal_nobias.item(2)
        bal_yawrslvmom = bal_nobias.item(3)
        bal_rolrslvmom = bal_nobias.item(4)
        bal_axirslvfrc = bal_nobias.item(5)
    else:
        bal_norrslvfrc = (bal_nobias.item(1)-bal_nobias.item(0))/(bal_dist.item(0)+bal_dist.item(1))
        bal_pitrslvmom = (bal_dist.item(1)*bal_nobias.item(0)-bal_dist.item(0)*bal_nobias.item(1))/(bal_dist.item(0)+bal_dist.item(1))
        bal_sidrslvfrc = (bal_nobias.item(3)-bal_nobias.item(2))/(bal_dist.item(2)+bal_dist.item(3))
        bal_yawrslvmom = (bal_dist.item(3)*bal_nobias.item(2)-bal_dist.item(2)*bal_nobias.item(3))/(bal_dist.item(2)+bal_dist.item(3))
        bal_rolrslvmom = bal_nobias.item(4)
        bal_axirslvfrc = bal_nobias.item(5)
        
    bal_rslv = np.matrix(([bal_norrslvfrc,bal_pitrslvmom,\
                           bal_sidrslvfrc,bal_yawrslvmom,\
                           bal_rolrslvmom,bal_axirslvfrc])).transpose()
    return bal_rslv

#%% Balance Calcs (continued)
bal_biascorr_rslv = rslv(bal_dist,bal_biascorr,bal_type)
bal_rslv = rslv(bal_dist,bal_nobias,bal_type)

#%% Tare Correction Clacs
t_corr = np.matrix([[sds.loc['BASP_NORTCORR']['SDS']], \
                     [sds.loc['BASP_PITTCORR']['SDS']], \
                     [sds.loc['BASP_SIDTCORR']['SDS']], \
                     [sds.loc['BASP_YAWTCORR']['SDS']], \
                     [sds.loc['BASP_ROLTCORR']['SDS']], \
                     [sds.loc['BASP_AXITCORR']['SDS']]])

bal_tcorrected_rslv = np.subtract(bal_rslv,t_corr)

    

