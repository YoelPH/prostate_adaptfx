# -*- coding: utf-8 -*-
"""
In this file are all helper functions that are needed for the adaptive fractionation calculation
"""

import numpy as np
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt



def data_fit(data):
    """
    This function fits the alpha and beta value for the prior

    Parameters
    ----------
    data : array or list
        list with n elements for each observed overlap volume

    Returns
    -------
    frozen function
        normal distribution
    """
    mu, std = norm.fit(data)
    return norm(loc = mu, scale = std)

def hyperparam_fit(data):
    """
    This function fits the alpha and beta value for the prior

    Parameters
    ----------
    data : array
        a nxk matrix with n the amount of patints and k the amount of sparing factors per patient.

    Returns
    -------
    list
        alpha and beta hyperparameter.
    """
    vars = data.var(axis=1)
    alpha, loc, beta = gamma.fit(vars, floc=0)
    return [alpha, beta]

def std_calc(measured_data, alpha, beta):
    """
    calculates the most likely standard deviation for a list of k sparing factors and a gamma conjugate prior
    measured_data: list/array with k sparing factors

    Parameters
    ----------
    measured_data : list/array
        list/array with k sparing factors
    alpha : float
        shape of gamma distribution
    beta : float
        scale of gamma distrinbution

    Returns
    -------
    std : float
        most likely std based on the measured data and gamma prior

    """
    n = len(measured_data)
    std_values = np.arange(0.00001, 0.5, 0.00001)
    likelihood_values = np.zeros(len(std_values))
    for index, value in enumerate(std_values):
        likelihood_values[index] = (
            value ** (alpha - 1)
            / value ** (n - 1)
            * np.exp(-1 / beta * value)
            * np.exp(-np.var(measured_data) / (2 * (value**2 / n)))
        )  # here i have to check whether it is good.
    std = std_values[np.argmax(likelihood_values)]
    return std





def get_state_space(distribution):
    """
    This function spans the state space for different volumes based on a probability distribution

    Parameters
    ----------
    distribution : frozen function
        normal distribution

    Returns
    -------
    state_space: Array spanning from the 2% percentile to the 98% percentile with a normalized spacing to define 100 states
        np.array
    """
    lower_bound = distribution.ppf(0.001)
    upper_bound = distribution.ppf(0.999)

    return np.linspace(lower_bound,upper_bound,200)

def probdist(X,state_space):
    """
    This function produces a probability distribution based on the normal distribution X

    Parameters
    ----------
    X : scipy.stats._distn_infrastructure.rv_frozen
        distribution function.

    Returns
    -------
    prob : np.array
        array with probabilities for each sparing factor.

    """
    prob = np.zeros(len(state_space))
    spacing = state_space[1]-state_space[0]
    for idx, state in enumerate(state_space):
        prob[idx] = X.cdf(state + spacing/2) - X.cdf(state - spacing/2)
    return np.array(prob) #note: this will only add up to roughly 96% instead of 100%

def max_action(accumulated_dose, dose_space, goal):
    """
    Computes the maximal dose that can be delivered to the tumor in each fraction depending on the actual accumulated dose

    Parameters
    ----------
    accumulated_dose : float
        accumulated tumor dose so far.
    dose_space : list/array
        array with all discrete dose steps.
    goal : float
        prescribed tumor dose.
    Returns
    -------
    sizer : integer
        gives the size of the resized actionspace to reach the prescribed tumor dose.

    """
    max_action = min(max(dose_space), goal - accumulated_dose)
    sizer = np.argmin(np.abs(dose_space - max_action))
    sizer = 1 if sizer == 0 else sizer #Make sure that at least the minimum dose is delivered
    return sizer

def actual_policy_plotter(policies_overlap: np.ndarray,volume_space: np.ndarray, probabilities: np.ndarray):
    """plots the actual policy given the overlap in volume space and the policies in policies overlap

    Args:
        policies_overlap (np.ndarray): policy for each overlap
        volume_space (np.ndarray): considered overlaps
        probabilities (np.ndarray): probability distribution of overlaps

    Returns:
        matplotlib figure: a figure with the actual policy plotted
    """
    color = 'tab:red'
    fig, ax = plt.subplots()
    ax.plot(volume_space,policies_overlap, label = 'optimal dose', color = color)
    ax.set_xlabel('Volume overlap in cc') 
    ax.set_ylabel('optimal dose')
    ax.set_title('policy of actual fraction')
    
    color = 'tab:blue'
    ax2 = ax.twinx()
    ax2.set_ylabel('probability') 
    ax2.plot(volume_space,probabilities, label = 'probabilities', color = color)
    fig.legend()
    return fig

def analytic_plotting(fraction, number_of_fractions, values, volume_space, dose_space):
    values[values < -10000000000] = 1.111111111111111
    min_Value = np.min(values)
    values[values == 1.111111111111111] = 1.1*min_Value
    colormap = plt.cm.get_cmap('jet')
    number_of_plots = number_of_fractions - fraction
    fig, axs = plt.subplots(1,number_of_plots, figsize = (number_of_plots*10,10))
    for index, ax in enumerate(axs): 
        img = ax.imshow(values[number_of_plots - index-1],extent = [volume_space.min(), volume_space.max(), dose_space.max(),dose_space.min()],cmap=colormap,aspect = 'auto')
        ax.set_title(f'value of fraction {fraction + index + 1}', fontsize = 24)
        ax.set_xlabel('overlap volume', fontsize = 24)
        ax.set_ylabel('accumulated dose', fontsize = 24)
        ax.tick_params(axis='both', which='both', labelsize=20)
    cbar = plt.colorbar(img, ax = ax)  
    cbar.set_label('state value', fontsize = 24)
    return fig