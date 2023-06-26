"""
This is the core of adaptive fractionation that computes the optimal dose for each fraction
"""


import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

from .helper_functions import std_calc
from .helper_functions import get_state_space
from .helper_functions import probdist
from .helper_functions import max_action





def adaptive_fractionation_core(fraction, volumes, accumulated_dose, number_of_fractions = 5, min_dose = 7.25, max_dose = 9.25, mean_dose = 8, alpha = 4.476900395372424, beta = 0.10723772034080617):
    goal = number_of_fractions * mean_dose #dose to be reached
    actual_volume = volumes[-1]
    minimum_future = accumulated_dose + min_dose 
    if all(x == volumes[0] for x in volumes) == True: #if all sparing factors are identical we just deliver the mean dose. Can be changed to a very small std in theory (then we deliver min_dose)
        actual_policy = mean_dose
        volume_space = np.ones(200)*actual_volume
        probabilities = np.ones(200)*1/200
        dose_space = np.arange(minimum_future,goal, 0.25) #spans the dose space delivered to the tumor
        dose_space = np.concatenate((dose_space, [goal, goal + 0.25])) # add an additional state that overdoses and needs to be prevented
        bound = goal + 0.25
        delivered_doses = np.arange(min_dose,max_dose + 0.01,0.05) #spans the action space of all deliverable doses

        values = np.zeros(((number_of_fractions - fraction), len(dose_space), len(volume_space))) # 2d values list with first indice being the accumulated dose and second being the overlap volume
        policies = np.zeros(((number_of_fractions - fraction), len(dose_space), len(volume_space)))
        policies_overlap = np.zeros(len(volume_space))
    else:
        std = std_calc(volumes, alpha, beta)
        distribution = norm(loc = volumes.mean(), scale = std)
        volume_space = get_state_space(distribution)
        probabilities = probdist(distribution,volume_space) #produce probabilities of the respective volumes
        dose_space = np.arange(minimum_future,goal, 0.25) #spans the dose space delivered to the tumor
        dose_space = np.concatenate((dose_space, [goal, goal + 0.25])) # add an additional state that overdoses and needs to be prevented
        bound = goal + 0.25
        delivered_doses = np.arange(min_dose,max_dose + 0.01,0.05) #spans the action space of all deliverable doses
        policies_overlap = np.zeros(len(volume_space))
        values = np.zeros(((number_of_fractions - fraction), len(dose_space), len(volume_space))) # 2d values list with first indice being the accumulated dose and second being the overlap volume
        policies = np.zeros(((number_of_fractions - fraction), len(dose_space), len(volume_space)))
        for state, fraction_state in enumerate(np.arange(number_of_fractions, fraction-1, -1)):
            
            if (state == number_of_fractions - 1):  # first fraction with no prior dose delivered so we dont loop through dose_space
                overlap_penalty = (np.outer(volume_space, (delivered_doses - min_dose).clip(0))).clip(0) #This means only values over min_dose get a penalty. Values below min_dose do not get a reward
                actual_penalty = (actual_volume * (delivered_doses - min_dose).clip(0)).clip(0)
                future_values_func = interp1d(dose_space, (values[state - 1] * probabilities).sum(axis=1))
                future_values = future_values_func(delivered_doses)  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions
                values_actual_frac = -overlap_penalty + future_values
                policies_overlap = delivered_doses[values_actual_frac.argmax(axis = 1)]
                actual_value =-actual_penalty + future_values
                actual_policy = delivered_doses[actual_value.argmax()]

            elif (fraction_state == fraction and fraction != number_of_fractions):  # actual fraction but not first fraction
                delivered_doses_clipped = delivered_doses[0 : max_action(accumulated_dose, delivered_doses, goal)+1]
                overlap_penalty = (np.outer(volume_space, (delivered_doses - min_dose).clip(0))).clip(0) #This means only values over min_dose get a penalty. Values below min_dose do not get a reward
                actual_penalty = (actual_volume * (delivered_doses_clipped - min_dose).clip(0)).clip(0)
                future_doses = accumulated_dose + delivered_doses_clipped
                future_doses[future_doses > goal] = bound
                penalties = np.zeros(future_doses.shape)
                penalties[future_doses > goal] = -1000000000000
                future_values_func = interp1d(dose_space, (values[state - 1] * probabilities).sum(axis=1))
                future_values = future_values_func(future_doses)  # for each dose and volume overlap calculate the penalty of the action and add the future value. We will only have as many future values as we have doses (not volumes dependent)
                values_actual_frac = -overlap_penalty + future_values + penalties
                policies_overlap = delivered_doses[values_actual_frac.argmax(axis = 1)]
                actual_value =-actual_penalty + future_values + penalties
                actual_policy = delivered_doses[actual_value.argmax()]
        
            elif (fraction == number_of_fractions):  #actual fraction is also the final fraction we do not need to calculate any penalty as the last action is fixed. 
                best_action = goal - accumulated_dose
                if accumulated_dose > goal:
                    best_action = 0
                if best_action < min_dose:
                    best_action = min_dose
                if best_action > max_dose:
                    best_action = max_dose
                actual_policy = best_action
        
            else: #any fraction that is not the actual one
                future_value_prob = (values[state - 1] * probabilities).sum(axis=1)
                future_values_func = interp1d(dose_space, future_value_prob)
                for tumor_index, tumor_value in enumerate(dose_space):  # this and the next for loop allow us to loop through all states
                    delivered_doses_clipped = delivered_doses[0 : max_action(tumor_value, delivered_doses, goal)+1]  # we only allow the actions that do not overshoot
                    overlap_penalty = (np.outer(volume_space, (delivered_doses_clipped - min_dose).clip(0))).clip(0) #This means only values over 8 get a penalty. Values below 8 do not get a reward. This results in a volume_space x doses space
                    if state != 0:
                        future_doses = tumor_value + delivered_doses_clipped
                        future_doses[future_doses > goal] = bound #all overdosing doses are set to the penalty state
                        future_values = future_values_func(future_doses)  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
                        penalties = np.zeros(future_doses.shape)
                        penalties[future_doses > goal] = -1000000000000
                        vs = -overlap_penalty + future_values + penalties
                        best_action = delivered_doses[vs.argmax(axis=1)]
                        valer = vs.max(axis=1)

                    else:  # last fraction when looping, only give the final penalty
                        best_action = goal - tumor_value
                        if best_action > max_dose:
                            best_action = max_dose
                        if best_action < min_dose:
                            best_action = min_dose
                        future_accumulated_dose = tumor_value + best_action
                        last_penalty = ((best_action - min_dose) * volume_space).clip(0) if (best_action - min_dose) > 0 else 0
                        underdose_penalty = 0
                        overdose_penalty = 0
                        if np.round(future_accumulated_dose,2) < goal:
                            underdose_penalty = -1000000000000 #in theory one can change this such that underdosing is penalted linearly
                        if np.round(future_accumulated_dose,2) > goal:
                            overdose_penalty = -1000000000000 
                        valer = (- last_penalty + underdose_penalty * np.ones(volume_space.shape) + overdose_penalty * np.ones(volume_space.shape))  # gives the value of each action for all sparing factors. elements 0-len(sparingfactors) are the Values for
                    policies[state][tumor_index] = best_action
                    values[state][tumor_index] = valer
                
    physical_dose = np.round(actual_policy,2)
    penalty_added = actual_volume * (physical_dose - min_dose).clip(0)
    return [policies, policies_overlap, volume_space, physical_dose, penalty_added, values, dose_space, probabilities]
    
   
def adaptfx_full(volumes, number_of_fractions = 5, min_dose = 7.25, max_dose = 9.25, mean_dose = 8):
    physical_doses = np.zeros(number_of_fractions)
    accumulated_doses = np.zeros(number_of_fractions)
    for index, frac in enumerate(range(1,6)):
        if frac != number_of_fractions:
            [policies, policies_overlap, volume_space, physical_dose, penalty_added, Values, dose_space, probabilities]  = adaptive_fractionation_core(frac,volumes[:-number_of_fractions+frac],accumulated_doses[index], min_dose = min_dose, max_dose = max_dose, mean_dose = mean_dose)
            accumulated_doses[index+1] = accumulated_doses[index] + physical_dose
        else:
            [policies, policies_overlap,  volume_space, physical_dose, penalty_added, Values, dose_space, probabilities]  = adaptive_fractionation_core(frac,volumes,accumulated_doses[index], min_dose = min_dose, max_dose = max_dose, mean_dose = mean_dose)
        physical_doses[index] = physical_dose
    total_penalty = ((physical_doses - 7.25) * volumes[-5:]).sum()
    return physical_doses, accumulated_doses, total_penalty



