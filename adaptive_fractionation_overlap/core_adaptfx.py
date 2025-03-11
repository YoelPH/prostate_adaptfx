"""
Core of adaptive fractionation computing optimal dose per fraction.
Refactored for readability and structure.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import pandas as pd

from .helper_functions import std_calc, get_state_space, probdist, max_action

# Constants for penalties and dose calculations
PENALTY = -1e12
DOSE_EPSILON = 1e-9  # To include max_dose due to floating-point precision


def adaptive_fractionation_core(
    fraction: int,
    volumes: np.ndarray,
    accumulated_dose: float,
    number_of_fractions: int = 5,
    min_dose: float = 7.5,
    max_dose: float = 9.5,
    mean_dose: float = 8,
    dose_steps: float = 0.25,
    alpha: float = 0.8909285040669036,
    beta: float = 0.522458969735114,
):
    """Compute optimal dose for a single fraction using adaptive fractionation."""
    goal = number_of_fractions * mean_dose
    actual_volume = volumes[-1]
    remaining_fractions = number_of_fractions - fraction + 1

    # Initialize accumulated dose for first fraction
    if fraction == 1:
        accumulated_dose = 0

    # Check if remaining dose is outside feasible range
    dose_remaining = goal - accumulated_dose
    if _is_dose_unfeasible(dose_remaining, remaining_fractions, min_dose, max_dose):
        return _handle_unfeasible_case(min_dose, max_dose, remaining_fractions, number_of_fractions, fraction)

    # Calculate volume distribution and probabilities
    std = std_calc(volumes, alpha, beta)
    distribution = norm(loc=volumes.mean(), scale=std)
    volume_space = get_state_space(distribution)
    probabilities = probdist(distribution, volume_space)

    # Initialize dose spaces and policy arrays
    dose_space = _create_dose_space(accumulated_dose + min_dose, goal, dose_steps)
    delivered_doses = _create_delivered_doses(min_dose, max_dose, dose_steps)
    policies_overlap = np.zeros(len(volume_space))
    values, policies = _initialize_policy_arrays(number_of_fractions, fraction, dose_space, volume_space)

    # Main loop to compute optimal policies
    for state, fraction_state in enumerate(np.arange(number_of_fractions, fraction - 1, -1)):
        if _is_first_state(state, number_of_fractions):
            policies_overlap, actual_value, optimal_dose = _handle_first_state(
                volume_space, actual_volume, delivered_doses, min_dose, dose_space,
                values[state - 1], probabilities
            )
        elif _is_current_fraction(fraction_state, fraction, number_of_fractions):
            optimal_dose, actual_value = _handle_current_fraction(
                accumulated_dose, volume_space, actual_volume, delivered_doses, min_dose,
                goal, dose_space, values[state - 1], probabilities
            )
        elif _is_last_fraction(fraction, number_of_fractions):
            optimal_dose, actual_value = _handle_last_fraction(accumulated_dose, goal, min_dose, max_dose)
        else:
            _update_policies_for_state(state, values, policies, dose_space, volume_space, delivered_doses,
                                       min_dose, goal, probabilities)

    # Prepare final results
    physical_dose = np.round(optimal_dose, 2)
    penalty_added = actual_volume * max(physical_dose - min_dose, 0)
    final_penalty = np.max(actual_value) - penalty_added

    return [policies, policies_overlap, volume_space, physical_dose, penalty_added,
            values, dose_space, probabilities, final_penalty]


def _is_dose_unfeasible(dose_remaining, remaining_fractions, min_dose, max_dose):
    """Check if remaining dose can't be met with min/max doses."""
    return (dose_remaining < remaining_fractions * min_dose) or (dose_remaining > remaining_fractions * max_dose)


def _handle_unfeasible_case(min_dose, max_dose, remaining_fractions, number_of_fractions, fraction):
    """Handle case where dose is not feasible with min/max constraints."""
    optimal_dose = min_dose if (remaining_fractions * min_dose >= number_of_fractions * min_dose) else max_dose
    policies = np.full((number_of_fractions - fraction, 200, 200), optimal_dose)
    policies_overlap = np.full(200, optimal_dose)
    values = np.full(((number_of_fractions - fraction), 200, 200), PENALTY)
    return [policies, policies_overlap, None, optimal_dose, 0, values, None, None, PENALTY]


def _create_dose_space(start, goal, step):
    """Create dose space array with overflow buffer."""
    dose_space = np.arange(start, goal, step)
    return np.concatenate([dose_space, [goal, goal + 0.05]])


def _create_delivered_doses(min_dose, max_dose, step):
    """Generate possible doses for delivery."""
    return np.arange(min_dose, max_dose + DOSE_EPSILON, step)


def _initialize_policy_arrays(number_of_fractions, fraction, dose_space, volume_space):
    """Initialize policy and value arrays."""
    values_shape = ((number_of_fractions - fraction), len(dose_space), len(volume_space))
    policies_shape = ((number_of_fractions - fraction), len(dose_space), len(volume_space))
    return np.zeros(values_shape), np.zeros(policies_shape)


def _is_first_state(state, number_of_fractions):
    """Check if processing the first state."""
    return state == number_of_fractions - 1


def _handle_first_state(volume_space, actual_volume, delivered_doses, min_dose, dose_space, prev_values, probabilities):
    """Calculate policies for the first state."""
    overlap_penalty = _calculate_overlap_penalty(volume_space, delivered_doses, min_dose)
    actual_penalty = _calculate_overlap_penalty(np.array([actual_volume]), delivered_doses, min_dose)
    future_values = _compute_future_values(prev_values, dose_space, probabilities, delivered_doses)
    values_actual_frac = -overlap_penalty + future_values
    policies_overlap = delivered_doses[values_actual_frac.argmax(axis=1)]
    actual_value = -actual_penalty.squeeze() + future_values
    optimal_dose = delivered_doses[actual_value.argmax()]
    return policies_overlap, actual_value, optimal_dose


def _calculate_overlap_penalty(volumes, doses, min_dose):
    """Calculate penalty based on overlap volumes and doses."""
    return np.outer(volumes, (doses - min_dose).clip(0)).clip(0)


def _compute_future_values(prev_values, dose_space, probabilities, future_doses):
    """Interpolate future values based on previous state."""
    future_value_prob = (prev_values * probabilities).sum(axis=1)
    future_values_func = interp1d(dose_space, future_value_prob, fill_value="extrapolate")
    return future_values_func(future_doses)


def _is_current_fraction(fraction_state, current_fraction, total_fractions):
    """Check if processing current fraction."""
    return fraction_state == current_fraction and current_fraction != total_fractions


def _handle_current_fraction(accumulated_dose, volume_space, actual_volume, delivered_doses, min_dose, goal,
                             dose_space, prev_values, probabilities):
    """Calculate policies for the current fraction."""
    max_act = max_action(accumulated_dose, delivered_doses, goal)
    delivered_doses_clipped = delivered_doses[:max_act + 1]
    overlap_penalty = _calculate_overlap_penalty(volume_space, delivered_doses_clipped, min_dose)
    actual_penalty = _calculate_overlap_penalty(np.array([actual_volume]), delivered_doses_clipped, min_dose)
    future_doses = accumulated_dose + delivered_doses_clipped
    future_doses = np.where(future_doses > goal, goal + 0.05, future_doses)
    penalties = np.where(future_doses > goal, PENALTY, 0)
    future_values = _compute_future_values(prev_values, dose_space, probabilities, future_doses)
    values_actual_frac = -overlap_penalty + future_values + penalties
    optimal_dose = delivered_doses_clipped[values_actual_frac.argmax()]
    actual_value = -actual_penalty.squeeze() + future_values + penalties
    return optimal_dose, actual_value


def _is_last_fraction(current_fraction, total_fractions):
    """Check if processing last fraction."""
    return current_fraction == total_fractions


def _handle_last_fraction(accumulated_dose, goal, min_dose, max_dose):
    """Determine dose for last fraction."""
    optimal_dose = goal - accumulated_dose
    optimal_dose = np.clip(optimal_dose, min_dose, max_dose)
    actual_value = np.zeros(1)
    return optimal_dose, actual_value


def _update_policies_for_state(state, values, policies, dose_space, volume_space, delivered_doses,
                               min_dose, goal, probabilities):
    """Update policies and values for non-special states."""
    future_value_prob = (values[state - 1] * probabilities).sum(axis=1)
    future_values_func = interp1d(dose_space, future_value_prob, fill_value="extrapolate")
    for tumor_index, tumor_value in enumerate(dose_space):
        max_act = max_action(tumor_value, delivered_doses, goal)
        delivered_doses_clipped = delivered_doses[:max_act + 1]
        overlap_penalty = _calculate_overlap_penalty(volume_space, delivered_doses_clipped, min_dose)
        future_doses = tumor_value + delivered_doses_clipped
        future_doses = np.where(future_doses > goal, goal + 0.05, future_doses)
        penalties = np.where(future_doses > goal, PENALTY, 0)
        future_values = future_values_func(future_doses)
        vs = -overlap_penalty + future_values + penalties
        best_action = delivered_doses_clipped[vs.argmax(axis=1)]
        policies[state][tumor_index] = best_action
        values[state][tumor_index] = vs.max(axis=1)


def adaptfx_full(volumes: list, number_of_fractions: int = 5, min_dose: float = 7.5,
                max_dose: float = 9.5, mean_dose: float = 8, dose_steps: float = 0.25,
                alpha: float = 0.8909285040669036, beta: float = 0.522458969735114):
    """Compute full adaptive fractionation plan."""
    physical_doses = np.zeros(number_of_fractions)
    accumulated_doses = np.zeros(number_of_fractions)
    for idx in range(number_of_fractions):
        frac = idx + 1
        current_volumes = volumes[:-number_of_fractions + frac] if frac != number_of_fractions else volumes
        result = adaptive_fractionation_core(
            fraction=frac, volumes=current_volumes, accumulated_dose=accumulated_doses[idx],
            number_of_fractions=number_of_fractions, min_dose=min_dose, max_dose=max_dose,
            mean_dose=mean_dose, dose_steps=dose_steps, alpha=alpha, beta=beta
        )
        physical_dose = result[3]
        physical_doses[idx] = physical_dose
        if idx < number_of_fractions - 1:
            accumulated_doses[idx + 1] = accumulated_doses[idx] + physical_dose
    total_penalty = ((physical_doses - min_dose) * volumes[-number_of_fractions:]).sum()
    return physical_doses, accumulated_doses, total_penalty


def precompute_plan(fraction: int, volumes: np.ndarray, accumulated_dose: float,
                   number_of_fractions: int = 5, min_dose: float = 7.5, max_dose: float = 9.5,
                   mean_dose: float = 8, dose_steps=0.25, alpha: float = 0.8909285040669036,
                   beta: float = 0.522458969735114):
    """Precompute possible doses for next fraction based on volume variations."""
    std = std_calc(volumes, alpha, beta)
    distribution = norm(loc=volumes.mean(), scale=std)
    volume_space = get_state_space(distribution)
    distribution_max = max(6.5, volume_space.max())
    volumes_to_check = np.arange(0, distribution_max, 0.1)
    predicted_policies = np.zeros(len(volumes_to_check))
    for idx, volume in enumerate(volumes_to_check):
        result = adaptive_fractionation_core(
            fraction=fraction, volumes=np.append(volumes, volume), accumulated_dose=accumulated_dose,
            number_of_fractions=number_of_fractions, min_dose=min_dose, max_dose=max_dose,
            mean_dose=mean_dose, dose_steps=dose_steps, alpha=alpha, beta=beta
        )
        predicted_policies[idx] = result[3]
    df = pd.DataFrame({'volume': volumes_to_check, 'dose': predicted_policies})
    return df, volumes_to_check, predicted_policies