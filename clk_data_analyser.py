import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import warnings
import numpy as np
import io
import re
from ipywidgets import interact, RadioButtons

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Clock Data Analyser", page_icon=":stopwatch:", layout="wide")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
st.title(":clock10: Clock Data Analyser ")


st.sidebar.image("https://www.bipm.org/documents/20126/27072194/Logo+BIPM+blue.png/797bb4e6-8cfb-5d78-b480-e460ad5eeec2?t=1583920059345", width=200)
#One line of gap 
st.sidebar.write("")
#IEEE UFFSC logo
st.sidebar.image("https://www.fusfoundation.org/images/IEEE-UFFC.jpg", width=200)

st.sidebar.header("Time & Frequency Capacity Building")


# Function to parse the clock data files
def parse_clock_data(file_objects):
    data_frames = []
    step_frames = []
    for file_object in file_objects:
        try:
            string_io = io.StringIO(file_object.getvalue().decode('utf-8'))

            for line in string_io:
                split_line = line.strip().split()

                # Check for step data format
                if len(split_line) >= 2 and re.match(r'^\d{5}\.\d{2}$', split_line[0]) and re.match(r'^\d{7}$', split_line[1]):
                    # Extract data with regex
                    step_match = re.match(r'(\d{5}\.\d{2})\s+(\d{7})\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+(\w+)\s+(\d{5})', line)
                    if step_match:
                        step_frames.append(step_match.groups())
                    else:
                        st.error(f"Step correction format is incorrect in file: {file_object.name}")
                        continue
                elif len(split_line) >= 2:
                    # Regular clock data parsing
                    colspecs = [(0, 5), (6, 11)]
                    i = 12
                    while i < len(line) and line[i:i+5].strip() != '':
                        colspecs.extend([(i, i+7), (i+7, i+18)])
                        i += 18
                    df = pd.read_fwf(io.StringIO(line), colspecs=colspecs, header=None)
                    if not df.empty:
                        col_names = ['MJD', 'Lab_Code']
                        for j in range((len(df.columns) - 2) // 2):
                            col_names.extend([f'TAI_Code_{j+1}', f'Clock_Diff_{j+1}'])
                        df.columns = col_names
                        data_frames.append(df)

        except Exception as e:
            st.error(f"Error processing file {file_object.name}: {e}")

    combined_data = pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
    combined_steps = pd.DataFrame(step_frames, columns=['MJD', 'Clock_Code', 'Time_Step', 'Frequency_Step', 'Lab_Acronym', 'Lab_Code']) if step_frames else pd.DataFrame()

    return combined_data, combined_steps


# Method to apply corrections to the data

def apply_corrections(combined_df, step_data, time_step_selection, freq_step_selection, counter):
    # Ensure MJD in combined_df is numeric
    combined_df['MJD'] = pd.to_numeric(combined_df['MJD'], errors='coerce')

    if step_data is not None:
        # Apply corrections up to the specified counter
        for index, step_row in step_data.iterrows():
            if index >= counter:  # Stop applying corrections once the counter limit is reached
                break

            # Convert step MJD and correction values to numeric
            step_mjd = pd.to_numeric(step_row['MJD'], errors='coerce')
            time_step = pd.to_numeric(step_row['Time_Step'], errors='coerce')
            freq_step = pd.to_numeric(step_row['Frequency_Step'], errors='coerce')

            # Select data points before the step correction MJD
            before_step = combined_df['MJD'] < step_mjd

            if time_step_selection != "Not Apply":
                # Apply or reverse the Time Step correction
                correction_factor = -time_step if time_step_selection == "Apply Reverse" else time_step
                combined_df.loc[before_step, 'Clock_Diff'] += correction_factor

            if freq_step_selection != "Not Apply":
                # Apply or reverse the Frequency Step correction
                correction_factor = -freq_step if freq_step_selection == "Apply Reverse" else freq_step
                # Calculate delta_t in days and apply Frequency Step correction
                combined_df.loc[before_step, 'Clock_Diff'] += correction_factor * (step_mjd - combined_df.loc[before_step, 'MJD'])

    return combined_df


# Function to plot MJD vs Clock Differences for a given TAI code
def plot_clock_differences(df, tai_code, step_data=None, step_correction_options= None,step_correction_counter=0):
    fig = go.Figure()

    # Dynamically find TAI code columns
    tai_code_columns = [col for col in df.columns if 'TAI_Code' in col]
    clock_diff_columns = [col.replace('TAI_Code', 'Clock_Diff') for col in tai_code_columns]

    mjd_combined = []
    clock_diff_combined = []

    for tai_col, clock_diff_col in zip(tai_code_columns, clock_diff_columns):
        if tai_col in df.columns and clock_diff_col in df.columns:
            subset = df[df[tai_col] == tai_code]
            if not subset.empty:
                mjd_combined.extend(subset['MJD'])
                clock_diff_combined.extend(subset[clock_diff_col])
    
    # Add step correction data if requested
    if step_correction_options and step_data is not None:
        step_subset = step_data[step_data['Clock_Code'] == str(tai_code)]
        for _, step_row in step_subset.iterrows():
            mjd_combined.append(step_row['MJD'])
            clock_diff_combined.append(step_row['Time_Step'])

    # Convert MJD and Clock_Diff to numeric and sort the data
    combined_df = pd.DataFrame({
        'MJD': pd.to_numeric(mjd_combined, errors='coerce'), 
        'Clock_Diff': pd.to_numeric(clock_diff_combined, errors='coerce')
    })
    combined_df = combined_df.drop_duplicates().sort_values(by='MJD')

    if step_data is not None and step_correction_options is not None:
        # Apply the step corrections based on user selection
        combined_df = apply_corrections(combined_df, step_data, *step_correction_options, step_correction_counter)


    # Linear and quadratic fit
    linear_fit = np.polyfit(combined_df['MJD'], combined_df['Clock_Diff'], 1)
    quadratic_fit = np.polyfit(combined_df['MJD'], combined_df['Clock_Diff'], 2)

    # Calculating residuals
    combined_df['Linear_Residuals'] = combined_df['Clock_Diff'] - np.polyval(linear_fit, combined_df['MJD'])
    combined_df['Quadratic_Residuals'] = combined_df['Clock_Diff'] - np.polyval(quadratic_fit, combined_df['MJD'])

    # Streamlit widget for selecting residual type
    # Streamlit widget for selecting residual type
    # Streamlit widget for selecting residual type in column format
    # with st.container():
    #     st.write("Select the type of residual to display:")  # Label
    #     residual_type = st.radio( "",('None', 'Linear Residuals', 'Quadratic Residuals'), key="residual_type" )

    # with st.container():
    #     # Use markdown with HTML to control spacing
    #     st.markdown("#### Select the type of residual to display:", unsafe_allow_html=True)
    #     residual_type = st.radio("", ('None', 'Linear Residuals', 'Quadratic Residuals'), key="residual_type")
        
    residual_type = st.radio(
        "Select the type of Data to display:",
        ('Clock Data', 'Linear Residuals', 'Quadratic Residuals'),
        key="residual_type"
    )
    # Use the residual_type to plot the data
    if residual_type == 'Clock Data':
        fig.add_trace(go.Scatter(x=combined_df['MJD'], y=combined_df['Clock_Diff'], mode='lines+markers', name='Clock Data'))
    elif residual_type == 'Linear Residuals':
        fig.add_trace(go.Scatter(x=combined_df['MJD'], y=combined_df['Linear_Residuals'], mode='lines+markers', name='Linear Residuals'))
    elif residual_type == 'Quadratic Residuals':
        fig.add_trace(go.Scatter(x=combined_df['MJD'], y=combined_df['Quadratic_Residuals'], mode='lines+markers', name='Quadratic Residuals'))

    fig.update_layout(
        title=f'MJD vs {residual_type} for the clock {tai_code}',
        xaxis=dict(
            title='MJD',
            title_font=dict(size=18),  # Adjust size as needed
            tickfont=dict(size=14)     # Adjust size as needed
        ),
        yaxis=dict(
            title=f'{residual_type} (ns)' if residual_type != 'None' else 'Clock Difference (ns)',
            title_font=dict(size=18),  # Adjust size as needed
            tickfont=dict(size=14)     # Adjust size as needed
        ),
        legend_title='Data Series',
        font=dict(size=44)  # This sets the font size for the rest of the figure components
    )
    fig.update_xaxes(tickformat="05d")
    return fig

# Main Streamlit app
def main():
    
    # Initialize 'step_correction_counter' in st.session_state
    if 'step_correction_counter' not in st.session_state:
        st.session_state['step_correction_counter'] = 0
        
        
    with st.form("my-form1", clear_on_submit=True):
        files_01 = st.file_uploader("Upload the clock data files", accept_multiple_files=True)
        submitted1 = st.form_submit_button("Submit")

    if submitted1 and files_01:
        combined_data, combined_steps = parse_clock_data(files_01)
        st.session_state['combined_data'] = combined_data
        st.session_state['combined_steps'] = combined_steps

    selected_code = None  # Initialize selected_code
    step_correction_options = None
    
    if 'combined_data' in st.session_state and not st.session_state['combined_data'].empty:
        tai_code_columns = [col for col in st.session_state['combined_data'].columns if 'TAI_Code' in col]
        all_tai_codes = pd.unique(st.session_state['combined_data'][tai_code_columns].fillna(0).astype(int).values.ravel('K'))
        all_tai_codes = [code for code in all_tai_codes if code != 0]

        selected_code = st.radio("Select a clock", all_tai_codes, key='tai_code_selection', horizontal=True)
        # step_correction_options= None

    if 'combined_steps' in st.session_state and not st.session_state['combined_steps'].empty:
        # Arrange step correction radio buttons in columns

        
        # Filter step corrections for the selected clock
        filtered_steps = st.session_state['combined_steps'][st.session_state['combined_steps']['Clock_Code'] == str(selected_code)]

        
       
        # Check if the selected clock matches any Clock_Code in step data
        if str(selected_code) in st.session_state['combined_steps']['Clock_Code'].unique():
            # st.write("Step Corrections:")
            # st.write(st.session_state['combined_steps'])
            # Ensure the types are consistent
            st.session_state['combined_steps']['Clock_Code'] = st.session_state['combined_steps']['Clock_Code'].astype(str)

            if str(selected_code) in st.session_state['combined_steps']['Clock_Code'].unique():
                st.write("Steps corrections mentioned in the data for this clock:")
                st.write(st.session_state['combined_steps'][st.session_state['combined_steps']['Clock_Code'] == str(selected_code)])
                st.write("Verify the step corrections in this clock")
                          
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        time_step_selection = st.radio("Time Step", ["Not Apply", "Apply", "Apply Reverse"], key='time_step')
                    with col2:
                        freq_step_selection = st.radio("Freq Step", ["Not Apply", "Apply", "Apply Reverse"], key='freq_step')
                step_correction_options = (time_step_selection, freq_step_selection)
        
        if 'step_correction_counter' not in st.session_state:
                st.session_state['step_correction_counter'] = 0
                
        # Check if there are more than one step corrections for the selected clock
        if len(filtered_steps) > 1:
            # Initialize the step correction counter if not already done
            if 'step_correction_counter' not in st.session_state:
                st.session_state['step_correction_counter'] = 0

            # Display increment and decrement buttons
            st.write("Apply corrections one after the other sequencially ")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Apply Previous Step Correction"):
                    st.session_state['step_correction_counter'] = max(0, st.session_state['step_correction_counter'] - 1)
            with col2:
                if st.button("Apply Next Step Correction"):
                    st.session_state['step_correction_counter'] = min(len(filtered_steps) - 1, st.session_state['step_correction_counter'] + 1)

            st.write(f"Step Corrections applied sequentially: {st.session_state['step_correction_counter']}")
        
    # Check if a code is selected before plotting
    if selected_code is not None:
        fig = plot_clock_differences(st.session_state['combined_data'], selected_code,st.session_state['combined_steps'], step_correction_options, st.session_state['step_correction_counter'])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No data available for the selected TAI Code.")
    

if __name__ == "__main__":
    main()
