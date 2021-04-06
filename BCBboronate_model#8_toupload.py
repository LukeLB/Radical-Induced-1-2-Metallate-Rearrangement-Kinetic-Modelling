# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:02:37 2019

@author: rc13564
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import odeint
from lmfit import Model, Parameters, minimize, fit_report
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "Arial"
def create_numpy_array(data_file_path, data_file_name):
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


    Parameters
    ----------
    params : 
    t : time trace for the function
    signal : signal trace(s) to fit to

    Returns
    -------
    
    model to be fit. Uses ode integration to numerically solve sets deifferntial equations which
    describe the reaction model.

    Parameters
    ----------
    params : class object
        class object which contains the parametrs wanting to be optmised. This is initialised outside of 
        the function.
    t1, t2, and t3 : array
        Array which contain the time steps of the data being modelled.
    signal1, signal2, and signal3 : array
        Arrays which contains the signal intenisties for species to be modelled (in integrated signal / arb. units)    
    
    Returns
    -------
    list of arrays
        Each array contains the residuals of the fit for an individual species to be minimised 

    """
    
    
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    k4 = params['k4']
    k5 = params['k5']
    k6 = params['k6']
    amp = [params['amp_C'], params['amp_F'], params['amp_D']] 
    amp_50mM = [params['amp_C_50mM'], params['amp_F_50mM'], params['amp_D_50mM']] 
    amp_70mM = [params['amp_F_70mM'], params['amp_D_70mM']] 
    def rxn(X, t):
        """
        creates the reaction model to pass into the ODE intergrator odeint()        
        
        Parameters
        ----------
        X : list
            A list containing initial concentrations of reactive species in the model.
        t : array
            An array containg the tiem steps of the reaction model.

        Returns
        -------
        list
            a list containg the concentration of each species at timestep t of the model

        """    
    
        A=X[0] #Boronate
        B=X[1] #EIA
        C=X[2] #Intermediate C
        D=X[3] #Radical
        E=X[4] #Intermediate E
        F=X[5] #Product
        G=X[6] #Static rxn radical
        H=X[7] #termn product
        O=X[8] #oxygen 
        
        #Reaction model
        # k1 A + D -> E
        # k2 E + B -> C + D 
        # k3 D + D -> H
        # k4 C -> F
        # k5 G + A -> E 
        # k6 D + O -> I
        
        #differential equations which describe the reaction
        dAdt=(-(k1*A*D)-(k5*A*G))
        dBdt=(-(k2*E*B))
        dCdt=(-(k4*C)+(k2*E*B))
        dDdt=(-(k1*A*D)+(k2*E*B)-2*(k3*D*D))-(k6*O*D)
        dEdt=(k1*A*D)-(k2*E*B)+(k5*A*G)
        dFdt=(k4*C)
        dGdt=(-k5*A*G)
        dHdt=(k3*D*D)
        dOdt=(k6*O*D)
        return [dAdt, dBdt, dCdt, dDdt, dEdt, dFdt, dGdt, dHdt, dOdt] 
       
    X0=[0.03,0.04,0,params['radical'],0,0,params['static_radical'],0, 1e-4] #intitial concs [Boronate, EIA, Int C, Radical, Intermediate E, Product, Static Rxn Rad, Termn product, oxygen conc]
    X0_50mM=[0.05,0.04,0,params['radical'],0,0,params['static_radical'],0, 1e-4] #intitial concs [Boronate, EIA, Int C, Radical, Intermediate E, Product, Static Rxn Rad, Termn product, oxygen conc]
    X0_70mM=[0.07,0.04,0,params['radical'],0,0,params['static_radical'],0, 1e-4] #intitial concs [Boronate, EIA, Int C, Radical, Intermediate E, Product, Static Rxn Rad, Termn productoxygen conc]
    
    C=odeint(rxn,X0,t1) #[Boronate, EIA, Int C, Radical, Intermediate E, Product, Static Rxn Rad, Termn product]
    C_50mM = odeint(rxn,X0_50mM,t2) #[Boronate, EIA, Int C, Radical, Intermediate E, Product, Static Rxn Rad, Termn product]
    C_70mM = odeint(rxn,X0_70mM,t3) #[Boronate, EIA, Int C, Radical, Intermediate E, Product, Static Rxn Rad, Termn product]
    
    C_sumradconc = np.add(C[:,3], C[:,6]) #sum the two radical concentrations of static and diffusional radical
    C_selected = np.column_stack((C[:,2], C[:,5], C_sumradconc)) #[Int C, Product, Total_Rad]
    C_fit = C_selected*amp #[Int C, Product(F), Total_Rad(D)
    C_sumradconc_50mM = np.add(C_50mM[:,3], C_50mM[:,6])
    C_selected_50mM = np.column_stack((C_50mM[:,2], C_50mM[:,5], C_sumradconc_50mM)) #[Int C, Product, Total_Rad]
    C_fit_50mM = C_selected_50mM*amp_50mM #[Int C, Product(F), Total_Rad(D)
    C_sumradconc_70mM = np.add(C_70mM[:,3], C_70mM[:,6])
    C_selected_70mM = np.column_stack((C_70mM[:,5], C_sumradconc_70mM)) #[Product, Total_Rad]
    C_fit_70mM = C_selected_70mM*amp_70mM #[Product(F), Total_Rad(D)]
    
    return ((C_fit[:,2]-signal1[:,1], C_fit[:,0]-signal1[:,2], C_fit[:,1]-signal1[:,3], 
             C_fit_50mM[:,2]-signal2[:,1], C_fit_50mM[:,0]-signal2[:,2], C_fit_50mM[:,1]-signal2[:,3],
             C_fit_70mM[:,1]-signal3[:,1], C_fit_70mM[:,0]-signal3[:,2])) #return the residual of: Radical, Int C, Product F at each boron ate concentration

def create_fitted_exp_plot_30mM(out,  t, signal):  
    """
    
    function to plot the kinetic model onto the experimental data 
    
    Parameters
    ----------
    out : class object
        object containing the minimised least squares fit of the model
    t : array
        array containing the time series to be plotted on the x-axis
    signal : array
        experimentally observed signal

    Returns
    -------
    None.

    """
    res=np.split(out.residual, 8)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fitted_s1=signal[:,1]+res[0]
    fitted_s2=signal[:,2]+res[1]
    fitted_s3=signal[:,3]+res[2]
    ax1.plot(t,signal[:,1],alpha=0.5, label = 'Radical D', color = colors['crimson'], linewidth = 2)
    ax1.plot(t,signal[:,2],alpha=0.5, label = 'Int C', color = 'purple', linewidth = 2)
    ax1.plot(t,signal[:,3],alpha=0.5, label = 'Product F', color = 'orange', linewidth = 2)
    ax1.plot(t,fitted_s1,linestyle='--', label = 'fitted D', color = colors['crimson'], linewidth = 2)
    ax1.plot(t,fitted_s2,linestyle='--', label = 'fitted C',  color = 'purple', linewidth = 2)
    ax1.plot(t,fitted_s3,linestyle='--', label = 'fitted F',  color = 'orange', linewidth = 2) 
    ax1.legend()
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    ax1.axes.set_xlabel('Time / s')
    ax1.axes.set_ylabel('A / mOD')
    
def create_fitted_exp_plot_50mM(out, t, signal):  
    """
    
    function to plot the kinetic model onto the experimental data 
    
    Parameters
    ----------
    out : class object
        object containing the minimised least squares fit of the model
    t : array
        array containing the time series to be plotted on the x-axis
    signal : array
        experimentally observed signal

    Returns
    -------
    None.

    """
    res=np.split(out.residual,8)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fitted_s1=signal[:,1]+res[3]
    fitted_s2=signal[:,2]+res[4]
    fitted_s3=signal[:,3]+res[5]
    ax1.plot(t,signal[:,1],alpha=0.5, label = 'Radical D', color = colors['crimson'], linewidth = 2)
    ax1.plot(t,signal[:,2],alpha=0.5, label = 'Int C', color = 'purple', linewidth = 2)
    ax1.plot(t,signal[:,3],alpha=0.5, label = 'Product F',  color = 'orange', linewidth = 2)
    ax1.plot(t,fitted_s1,linestyle='--', label = 'fitted D', color = colors['crimson'], linewidth = 2)
    ax1.plot(t,fitted_s2,linestyle='--', label = 'fitted C', color = 'purple', linewidth = 2)
    ax1.plot(t,fitted_s3,linestyle='--', label = 'fitted F', color = 'orange', linewidth = 2) 
    ax1.legend()
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    ax1.axes.set_xlabel('Time / s')
    ax1.axes.set_ylabel('A / mOD')
    
def create_fitted_exp_plot_70mM(out, t, signal):  
    """
    
    function to plot the kinetic model onto the experimental data 
    
    Parameters
    ----------
    out : class object
        object containing the minimised least squares fit of the model
    t : array
        array containing the time series to be plotted on the x-axis
    signal : array
        experimentally observed signal

    Returns
    -------
    None.

    """
    res=np.split(out.residual,8)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fitted_s1=signal[:,1]+res[6]
    fitted_s2=signal[:,2]+res[7]
    ax1.plot(t,signal[:,1],alpha=0.5, label = 'Radical D', color = colors['crimson'], linewidth = 2)
    ax1.plot(t,signal[:,2],alpha=0.5, label = 'Product F',  color = 'orange', linewidth = 2)
    ax1.plot(t,fitted_s1,linestyle='--', label = 'fitted D', color = colors['crimson'], linewidth = 2)
    ax1.plot(t,fitted_s2,linestyle='--', label = 'fitted F',  color = 'orange', linewidth = 2)
    ax1.legend()
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    ax1.axes.set_xlabel('Time / s')
    ax1.axes.set_ylabel('A / mOD')


data_file_path = r"" #enter file path to experimental data
# file names
data_file_name_1 = "30mM_cleaneddata_EIA_PROD_INTC_RAD.csv"
data_file_name_2 = "50mM_cleaneddata_RAD_INTC_F.csv"
data_file_name_3 = "70mM_cleaneddata_RAD_F.csv"
#create arrays from signal data
signal1 = create_numpy_array(data_file_path,data_file_name_1) #columns Time / ns, D, C, F
signal2 = create_numpy_array(data_file_path,data_file_name_2) #columns Time / ns, D, C, F
signal3 = create_numpy_array(data_file_path,data_file_name_3) #columns Time / ns, D, F
#take time series from signal data
t1 = signal1[:,0]*(1e-9)
t2 = signal2[:,0]*(1e-9)
t3 = signal3[:,0]*(1e-9)

#Initilaise parameter class and add parameters to model 

params = Parameters()
#rate coefficients 
k = [1e9, 1.3e10, 1.5e9, 1e6, 3.03e11, 3.1e5]
params.add('k1', value=k[0])
params.add('k2', value=k[1], vary = False)
params.add('k3', value=k[2])
params.add('k4', value=k[3])
params.add('k5', value=k[4], vary = False)
params.add('k6', value=k[5], vary = False)
#static and diffusive radical conentrations
params.add('radical', value=3.8e-5, vary = False) #radical conc in M
params.add('static_radical', value=1e-5, vary = False) #static radical conc 
#amplitudes to convert from concentration to integrated signal
params.add('amp_C', value=30)
params.add('amp_F', value=17)
params.add('amp_D', value=5700)
params.add('amp_C_50mM', value=20)
params.add('amp_F_50mM', value=17)
params.add('amp_D_50mM', value=3143)
params.add('amp_F_70mM', value=17)
params.add('amp_D_70mM', value=3143)

#run fit
out = minimize(rxn_fit, params, args=(t1, t2, t3, signal1, signal2, signal3), method='leastsq')
print(fit_report(out.params))

#plot fit
create_fitted_exp_plot_30mM(out, t1, signal1)
create_fitted_exp_plot_50mM(out, t2, signal2)
create_fitted_exp_plot_70mM(out, t3, signal3)
