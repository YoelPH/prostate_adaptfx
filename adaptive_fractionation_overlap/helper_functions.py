# -*- coding: utf-8 -*-
"""
In this file are all helper functions that are needed for the adaptive fractionation calculation
"""

import numpy as np
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt



def data_fit(data):
    """
    This function fits a normal distribution for the given data

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
    calculates the most likely standard deviation for a list of k overlap volumes and a gamma prior
    measured_data: list/array with k sparing factors

    Parameters
    ----------
    measured_data : list/array
        list/array with k overlap volumes
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
    std_values = np.arange(0.001, 10, 0.001)
    likelihood_values = np.zeros(len(std_values))
    for index, value in enumerate(std_values):
        likelihood_values[index] = (
            value ** (alpha - 1)
            / value ** (n - 1)
            * np.exp(-1 / beta * value)
            * np.exp(-np.var(measured_data) / (2 * (value**2 / n)))
        )
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

def penalty_calc_single(physical_dose, min_dose, mean_dose, actual_volume, steepness):
    """
    This function calculates the penalty for the given dose and volume by adding the triangle arising from the dose gradient
    if the dose delivered is larger than the uniform fractionated dose.
    """
    steepness = np.abs(steepness)
    if physical_dose > mean_dose:
        penalty_added = (physical_dose - min_dose) * (actual_volume) + (physical_dose - mean_dose)**2*steepness/2
    else:
        penalty_added = actual_volume * (physical_dose - min_dose)
    return penalty_added

def benefit_calc_single(physical_dose, mean_dose, actual_volume, steepness):
    """
    This function calculates the benefit for the given dose and volume by adding the triangle arising from the dose gradient
    if the dose delivered is smaller than the uniform fractionated dose.
    """
    steepness = np.abs(steepness)
    if physical_dose < mean_dose:
        # benefit_added = (mean_dose - physical_dose) * (actual_volume + (mean_dose - physical_dose)*steepness/2)
        benefit_added = (mean_dose - physical_dose) * ((mean_dose - physical_dose)*steepness/2)
    else:
        benefit_added = 0
    return benefit_added    


def penalty_calc_single_volume(delivered_doses, min_dose, mean_dose, actual_volume, steepness):
    """
    This function calculates the penalty for the given doses and single volume by adding the triangle arising from the dose gradient
    if the dose delivered is larger than the uniform fractionated dose.
    """
    steepness = np.abs(steepness)
    overlap_penalty_linear = (delivered_doses - min_dose) * actual_volume
    overlap_penalty_quadratic = (delivered_doses - mean_dose)**2*steepness/2
    overlap_penalty_quadratic[delivered_doses <= mean_dose] = 0
    overlap_penalty = overlap_penalty_linear + overlap_penalty_quadratic
    return overlap_penalty


def benefit_calc_single_volume(delivered_doses, mean_dose, actual_volume, steepness):
    """
    This function calculates the benefit for the given doses and single volume by adding the triangle arising from the dose gradient
    if the dose delivered is smaller than the uniform fractionated dose.
    """
    steepness = np.abs(steepness)
    # overlap_benefit_linear = (mean_dose - delivered_doses) * actual_volume
    overlap_benefit_quadratic = (mean_dose - delivered_doses)**2*steepness/2
    overlap_benefit_quadratic[delivered_doses >= mean_dose] = 0
    # overlap_benefit_linear[delivered_doses >= mean_dose] = 0
    # overlap_benefit = overlap_benefit_linear + overlap_benefit_quadratic
    overlap_benefit = overlap_benefit_quadratic
    return overlap_benefit


def penalty_calc_matrix(delivered_doses, volume_space, min_dose, mean_dose, steepness):
    """
    This function calculates the penalty for the given dose and volume by adding the triangle arising from the dose gradient
    if the dose delivered is larger than the uniform fractionated dose.
    """
    steepness = np.abs(steepness)
    overlap_penalty_linear = (np.outer(volume_space, (delivered_doses - min_dose)))
    overlap_penalty_quadratic = (delivered_doses - mean_dose)**2*steepness/2
    overlap_penalty_quadratic[delivered_doses <= mean_dose] = 0
    overlap_penalty = overlap_penalty_linear + overlap_penalty_quadratic
    return overlap_penalty


def benefit_calc_matrix(delivered_doses, volume_space, mean_dose, steepness):
    """
    This function calculates the benefit for the given dose and volume by adding the triangle arising from the dose gradient
    if the dose delivered is smaller than the uniform fractionated dose.
    """
    steepness = np.abs(steepness)
    # overlap_benefit_linear = (np.outer(volume_space, (mean_dose - delivered_doses)))
    overlap_benefit_quadratic = (mean_dose - delivered_doses)**2*steepness/2
    overlap_benefit_quadratic[delivered_doses >= mean_dose] = 0
    # overlap_benefit_linear[:,delivered_doses >= mean_dose] = 0
    # overlap_benefit = overlap_benefit_linear + overlap_benefit_quadratic
    overlap_benefit = overlap_benefit_quadratic
    return overlap_benefit

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

def actual_policy_plotter(policies_overlap: np.ndarray,volume_space: np.ndarray, probabilities: np.ndarray = None):
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
    
    if probabilities is not None:
        color = 'tab:blue'
        ax2 = ax.twinx()
        ax2.set_ylabel('probability')
        ax2.plot(volume_space,probabilities, label = 'probabilities', color = color)
    fig.legend()
    return fig

def analytic_plotting(fraction: int, number_of_fractions: int, values: np.ndarray, volume_space: np.ndarray, dose_space: np.ndarray):
    """plots all future values given the values calculated by adaptive_fractionation_core.
    Only available for fractions 1 - (number of fractions - 1)

    Args:
        fraction (int): number of actual fraction
        number_of_fractions (int): total number of fractions
        values (np.ndarray): remaining_fractions x volume_space x dose_space dimensional array with values for each volume/dose pair
        volume_space (np.ndarray): 1 dimensional array with all considered volume overlaps
        dose_space (np.ndarray): 1 dimensional array with all considered future accumulated doses

    Returns:
        matplotlib.fig: returns a figure with all values plotted as subfigures
    """
    values[values < -10000000000] = 10000000000
    min_Value = np.min(values)
    values[values == 10000000000] = 1.1*min_Value
    colormap = plt.cm.get_cmap('jet')
    number_of_plots = number_of_fractions - fraction
    fig, axs = plt.subplots(1,number_of_plots, figsize = (number_of_plots*10,10))
    if number_of_plots > 1:
        for index, ax in enumerate(axs): 
            img = ax.imshow(values[number_of_plots - index-1],extent = [volume_space.min(), volume_space.max(), dose_space.max(),dose_space.min()],cmap=colormap,aspect = 'auto')
            ax.set_title(f'value of fraction {fraction + index + 1}', fontsize = 24)
            ax.set_xlabel('overlap volume', fontsize = 24)
            ax.set_ylabel('accumulated dose', fontsize = 24)
            ax.tick_params(axis='both', which='both', labelsize=20)
        cbar = plt.colorbar(img, ax = ax)  
        cbar.set_label('state value', fontsize = 24)
    else:
        img = axs.imshow(values[0], extent = [volume_space.min(), volume_space.max(), dose_space.max(),dose_space.min()],cmap=colormap,aspect = 'auto')
        axs.set_title(f'value of fraction {fraction + 1}', fontsize = 24)
        axs.set_xlabel('overlap volume', fontsize = 24)
        axs.set_ylabel('accumulated dose', fontsize = 24)
        axs.tick_params(axis='both', which='both', labelsize=20)
        cbar = plt.colorbar(img, ax = axs)  
        cbar.set_label('state value', fontsize = 24) 

    return fig