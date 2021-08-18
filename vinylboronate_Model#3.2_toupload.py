# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:02:37 2019

@author: ll17354
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import odeint
from lmfit import Model, Parameters, minimize, fit_report
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

def create_numpy_array(data_file_path,
                      data_file_name):
    """
    creates a numpy array containing the time points vs intensity that need to be plotted 

    Parameters
    ----------
    data_file_path : string that contains the path to the csv file
    data_file_name : string that contains the name of the csv file as X.csv

    Returns
    -------
    array a numpy array 
    """
    array = np.genfromtxt(os.path.join(data_file_path, data_file_name),delimiter=',') #reads csv file into a dataframe 
    return array

def rxn_fit(params, t1, t2, t3, signal1, signal2, signal3):
    """
    Model to be fit. Uses ode integration to numerically solve sets deifferntial equations which
    describe the reaction model.

    Parameters
    ----------
    params : class object which contains the parametrs wanting to be optmised. This is initialised outside of 
    the function.
    t : time trace for the function
    signal : signal trace(s) to fit to

    Returns
    -------
    residuals of the fit to be minimised 
    """
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    k4 = params['k4']
    k5 = params['k5']
    
    amp = [params['amp_Product_C_30mM'], params['amp_Radical_D_30mM']] 
    amp_40mM = [params['amp_Product_C_40mM'], params['amp_Radical_D_40mM']] 
    amp_50mM = [params['amp_Product_C_50mM'], params['amp_Radical_D_50mM']] 
    
    def rxn(X,t):
        A=X[0] #Boronate
        C=X[1] #Int C
        D=X[2] #Radical
        E=X[3] #Intermediate E
        G=X[4] #Product_IAT
        O=X[5] #Oxygen
        
        #Reaction model
        # k1 A + D -> E
        # k2 E -> C + D
        # k3 D + D -> F
        # k4 C -> G
        # k5 D + O -> DO
        
        #differential equations which describe the reaction
        dAdt=-k1*A*D
        dCdt=k2*E-k4*C
        dDdt=-k1*A*D+k2*E-2*k3*D*D-k5*D*O
        dEdt=k1*A*D-k2*E
        dGdt=k4*C
        dOdt=-k5*D*O
        return [dAdt, dCdt, dDdt, dEdt, dGdt, dOdt]
 
       
    X0=[0.03,0,params['radical'],0,0, 1e-4] #intitial concs [Boronate, Int C, Radical, Intermediate E, product, O2]
    X0_40mM=[0.04,0,params['radical'],0,0, 1e-4] #intitial concs [Boronate, Int C, Radical, Intermediate E, product, O2]
    X0_50mM=[0.05,0,params['radical'],0,0, 1e-4] #intitial concs [Boronate, Int C, Radical, Intermediate E, product, O2]
    C=odeint(rxn,X0,t1) #[Boronate, Int C, Radical, Intermediate E, product]
    C_40mM = odeint(rxn,X0_40mM,t2) #[Boronate, Int C, Radical, Intermediate E, product]
    C_50mM = odeint(rxn,X0_50mM,t3) #[Boronate, Int C, Radical, Intermediate E, product]
    
    C_selected = np.column_stack((C[:,4], C[:,2])) #[Product C, Radical]
    C_fit = C_selected*amp #[Product C, Rad]
    C_selected_40mM = np.column_stack((C_40mM[:,4], C_40mM[:,2])) #[Product C, Radical]
    C_fit_40mM = C_selected_40mM*amp_40mM #[Product C, Rad]
    C_selected_50mM = np.column_stack((C_50mM[:,4], C_50mM[:,2])) #[Product C, Radical]
    C_fit_50mM =  C_selected_50mM*amp_50mM #[Product C, Rad]
    
    return ((C_fit[:,0]-signal1[:,1], C_fit[:,1]-signal1[:,3], 
             C_fit_40mM[:,0]-signal2[:,1], C_fit_40mM[:,1]-signal2[:,3],
             C_fit_50mM[:,0]-signal3[:,1], C_fit_50mM[:,1]-signal3[:,3]))  #return the residual of: Radical, Static Product H, Product C

def create_initialised_params_plot(t, amp, X0, params, signal):
    """
    Model to be fit. Uses ode integration to numerically solve sets differntial equations which
    describe the reaction model.

    Parameters
    ----------
    t : time trace for the function
    amp : the ampplitudes to applied to the model for conversion from concentration to absorbance 
    X0 : initial concentrations to be plugged into model
    params : paramters to be plugged into model. i.e rate coeffs, initial concs
    signal : signal trace(s) to fit to
    
    Returns
    -------
    plots of the initial model over the observed transient kinetics 
    """
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    k4 = params['k4']
        
    def rxn(X,t):
        A=X[0] #Boronate
        C=X[1] #Int C
        D=X[2] #Radical
        E=X[3] #Intermediate E
        G=X[4] #Product
        
        #Reaction model
        # k1 A + D -> E
        # k2 E -> C + D
        # k3 D + D -> F
        # k4 C -> G
        
        #differential equations which describe the reaction
        dAdt=-k1*A*D
        dCdt=k2*E-k4*C
        dDdt=-k1*A*D+k2*E-2*k3*D*D
        dEdt=k1*A*D-k2*E
        dGdt=k4*C
        return [dAdt, dCdt, dDdt, dEdt, dGdt]
 
    C=odeint(rxn,X0,t1) #intitial concs [Boronate, Int C, Radical, Intermediate E, product]   
    C_selected = np.column_stack((C[:,4], C[:,2])) #[Product C, Radical]
    C_fit = C_selected*amp #[Product C, Rad]
    
    #plotting code
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(t,C_fit[:,1],label='Model Radical D', color = colors['crimson'], linewidth = 2)
    ax3.plot(t,C_fit[:,0],label='Model Product_IAT', color = 'purple', linewidth = 2)
    ax3.plot(t, signal[:,1], label='Exp_IAT', alpha = 0.5, color = 'purple', linewidth = 2)
    ax3.plot(t, signal[:,3], label='Exp_D', alpha =0.5, color = colors['crimson'], linewidth = 2)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=3, mode="expand", borderaxespad=0.)
    ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    ax3.axes.set_xlabel('Time / s')
    ax3.axes.set_ylabel('Integrated Signal')
    
def create_fitted_exp_plot_30mM(out, t, signal):  
    res=np.split(out.residual,6)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fitted_s1=signal[:,1]+res[0]
    fitted_s3=signal[:,3]+res[1]
    ax1.plot(t,signal[:,1],alpha=0.5, label = 'Product F', color = colors['crimson'], linewidth = 2)
    ax1.plot(t,signal[:,3],alpha=0.5, label = 'Radical D',  color = 'orange', linewidth = 2)
    ax1.plot(t,fitted_s1,linestyle='--', label = 'fitted F', color = colors['crimson'], linewidth = 2)
    ax1.plot(t,fitted_s3,linestyle='--', label = 'fitted D', color = 'orange', linewidth = 2) 
    ax1.legend()
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    ax1.axes.set_xlabel('Time / s')
    ax1.axes.set_ylabel('Integrated Signal')
    
def create_fitted_exp_plot_40mM(out, t, signal):  
    res=np.split(out.residual,6)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fitted_s1=signal[:,1]+res[2]
    fitted_s3=signal[:,3]+res[3]
    ax1.plot(t,signal[:,1],alpha=0.5, label = 'Product F', color = colors['crimson'], linewidth = 2)
    ax1.plot(t,signal[:,3],alpha=0.5, label = 'Radical D',  color = 'orange', linewidth = 2)
    ax1.plot(t,fitted_s1,linestyle='--', label = 'fitted F', color = colors['crimson'], linewidth = 2)
    ax1.plot(t,fitted_s3,linestyle='--', label = 'fitted D', color = 'orange', linewidth = 2) 
    ax1.legend()
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    ax1.axes.set_xlabel('Time / s')
    ax1.axes.set_ylabel('Integrated Signal')
    
def create_fitted_exp_plot_50mM(out, t, signal):  
    res=np.split(out.residual,6)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fitted_s1=signal[:,1]+res[4]
    fitted_s3=signal[:,3]+res[5]
    ax1.plot(t,signal[:,1],alpha=0.5, label = 'Product F', color = colors['crimson'], linewidth = 2)
    ax1.plot(t,signal[:,3],alpha=0.5, label = 'Radical D',  color = 'orange', linewidth = 2)
    ax1.plot(t,fitted_s1,linestyle='--', label = 'fitted F', color = colors['crimson'], linewidth = 2)
    ax1.plot(t,fitted_s3,linestyle='--', label = 'fitted D', color = 'orange', linewidth = 2) 
    ax1.legend()
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    ax1.axes.set_xlabel('Time / s')
    ax1.axes.set_ylabel('Integrated Signal')

data_file_path = r"C:\Users\ll17354\OneDrive - University of Bristol\MyFiles-Migrated\Documents\Projects\Mattia Visible Light Boronate Radical Addition\Modelling\Vinyl Boronate\Model #1 - Simple Chain with termn"
data_file_name_1 = "vinylboronate_30mM_clean.csv"
data_file_name_2 = "vinylboronate_40mM_clean.csv"
data_file_name_3 = "vinylboronate_50mM_clean.csv"
signal1 = create_numpy_array(data_file_path,data_file_name_1) #columns Time / ns, Product C, Static Product H, Total_Rad_D
signal2 = create_numpy_array(data_file_path,data_file_name_2) #columns Time / ns, Product C, Static Product H, Total_Rad_D
signal3 = create_numpy_array(data_file_path,data_file_name_3) #columns Time / ns, Product C, Static Product H, Total_Rad_D
t1 = signal1[:,0]*(1e-9)
t2 = signal2[:,0]*(1e-9)
t3 = signal3[:,0]*(1e-9)
#Initilaise parameter class and add floated parameters 
params = Parameters()
k = [1e11, 3e10, 1.5e10, 1e6, 3.1e5]
params.add('radical', value=6e-5, vary=False) #radical conc in M 
params.add('k1', value=k[0])
params.add('k2', value=k[1])
params.add('k3', value=k[2], vary = False)
params.add('k4', value=k[3])
params.add('k5', value=k[4], vary = False)
params.add('amp_Product_C_30mM', value=8.5)
params.add('amp_Radical_D_30mM', value=3000)
params.add('amp_Product_C_40mM', value=1)
params.add('amp_Radical_D_40mM', value=3143)
params.add('amp_Product_C_50mM', value=17)
params.add('amp_Radical_D_50mM', value=3143)

out = minimize(rxn_fit, params, args=(t1, t2, t3, signal1, signal2, signal3), method='leastsq')
print(fit_report(out.params))


amp = [params['amp_Product_C_30mM'], params['amp_Radical_D_30mM']] 
amp_40mM = [params['amp_Product_C_40mM'], params['amp_Radical_D_40mM']] 
amp_50mM = [params['amp_Product_C_50mM'], params['amp_Radical_D_50mM']] 
X0=[0.03,0,params['radical'],0,0,1e-4] #intitial concs [Boronate, Int C, Radical, Intermediate E, product, O2]
X0_40mM=[0.04,0,params['radical'],0,0,1e-4] #intitial concs [Boronate, Int C, Radical, Intermediate E, product, O2]
X0_50mM=[0.05,0,params['radical'],0,0,1e-4] #intitial concs [Boronate, Int C, Radical, Intermediate E, product, O2]

# create_initialised_params_plot(t1, amp, X0, params, signal1)
#30,0,0.04,0,0,0
create_fitted_exp_plot_30mM(out, t1, signal1)
create_fitted_exp_plot_40mM(out, t2, signal2)
create_fitted_exp_plot_50mM(out, t3, signal3)

