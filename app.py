import streamlit as st
import adaptive_fractionation_overlap as af
import numpy as np
st.set_page_config(layout="wide")
st.title('Adaptive Fractionation for prostate cancer interface')
st.markdown('## info \n This web app is supposed to be used as user interface to compute the optimal dose to be delivered in prostate adaptive fractionation if you have any questions please contact [yoel.perezhaas@usz.ch](mailto:yoel.perezhaas@usz.ch)')

st.header('User Input')
function = st.radio('Type of adaptive fractionation', ['actual fraction calculation','full plan calculation'])


left, right = st.columns(2)  
with left:
    fractions = st.text_input('total number of fractions', '5', help = 'insert the number of total fractions in the treatment')
    overlaps_str = st.text_input('observed overlap volumes in cc separated by spaces', help = 'insert ALL observed overlaps for the patient. for a full plan at least (number of fractions - 1) volumes are required')
    actual_fraction = st.text_input('number of actual fraction', disabled = (function =='full plan calculation'), help = 'the actual fraction number is only needed for the actual fraction calculation')
with right:
    minimum_dose = st.text_input('minimum dose', '7.25', help = 'insert the minimum dose in Gy')
    maximum_dose = st.text_input('maximum dose', '9.25', help = 'insert the maximum dose in Gy')
    mean_dose = st.text_input('mean dose to be delivered over all fractions', '8', help = 'insert mean dose in Gy')
    accumulated_dose = st.text_input('accumulated physical dose in previous fractions', disabled = (function =='full plan calculation'), help = 'the accumulated dose is only needed in the actual fraction calculation set to 0 if actual fraction is 1')


st.header('Results')


if st.button('compute optimal dose', help = 'takes the given inputs from above to compute the optimal dose'):
    overlaps_str = overlaps_str.split()
    overlaps = [float(i) for i in overlaps_str]
    if function == 'actual fraction calculation':
        [policies, policies_overlap, volume_space, physical_dose, penalty_added, values, dose_space, probabilities, final_penalty] = af.adaptive_fractionation_core(int(actual_fraction),np.array(overlaps), float(accumulated_dose), int(fractions), float(minimum_dose), float(maximum_dose), float(mean_dose))
        left2, right2 = st.columns(2)  
        with left2:
            actual_value = 'Goal can not be reached' if final_penalty <= -100000000000 else str(np.round(final_penalty,1)) + 'ccGy'
            st.metric(label="optimal dose for actual fraction", value= str(physical_dose) + 'Gy', delta = (physical_dose - float(mean_dose)))
            st.metric(label="expected final penalty from this fraction", value = actual_value)
            if final_penalty <= -100000000000:
                st.write('the minimal dose is delivered if we overdose, the maximal dose is delivered if we underdose')
        with right2:
            st.pyplot(af.actual_policy_plotter(policies_overlap,volume_space,probabilities))
        figure = af.analytic_plotting(int(actual_fraction),int(fractions),values, volume_space, dose_space)    
        with st.expander('see Analytics'):
            st.header('Analytics')
            st.pyplot(figure)
            st.write('The figures above show the value function for each future fraction. These functions help to identify whether a potential mistake has been made in the calculation.')
    else:
        [physical_doses, accumulated_doses, total_penalty] = af.adaptfx_full(np.array(overlaps), int(fractions), float(minimum_dose), float(maximum_dose), float(mean_dose))
        col1, col2, col3, col4, col5 = st.columns(5)  
        with col1:
            st.metric(label="**overlap**", value = str(overlaps[-5]) + 'cc')
            st.metric(label="**first fraction**", value = str(physical_doses[0]) + 'Gy', delta = (physical_doses[0] - float(mean_dose)))
        with col2:
            st.metric(label="**overlap**", value = str(overlaps[-4]) + 'cc')
            st.metric(label="**second fraction**", value = str(physical_doses[1]) + 'Gy', delta = (physical_doses[1] - float(mean_dose)))
        with col3:
            st.metric(label="**overlap**", value = str(overlaps[-3]) + 'cc')
            st.metric(label="**third fraction**", value = str(physical_doses[2]) + 'Gy', delta = (physical_doses[2] - float(mean_dose)))
        with col4:
            st.metric(label="**overlap**", value = str(overlaps[-2]) + 'cc')
            st.metric(label="**fourth fraction**", value = str(physical_doses[3]) + 'Gy', delta = (physical_doses[3] - float(mean_dose)))
        with col5:
            st.metric(label="**overlap**", value = str(overlaps[-1]) + 'cc')
            st.metric(label="**fifth fraction**", value = str(physical_doses[4]) + 'Gy', delta = (physical_doses[4] - float(mean_dose)))
        st.header('Plan summary')
        st.markdown('The adaptive plan achieved a total penalty of:')
        st.metric(label = "penalty", value = str(total_penalty) + 'ccGy', delta = np.round(total_penalty - (np.array(overlaps[-5:])*0.75).sum(),2), delta_color= 'inverse')
        st.markdown('The arrow shows the comparison to standard fractionation, i.e. (number of fractions x mean dose). A green arrow shows an improvement.')

