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
                if len(split_line) >= 1 and "." in split_line[0] and re.match(r'^\d{5}\.\d{2}$', split_line[0]):
                    if len(split_line) >= 2 and re.match(r'^\d{7}$', split_line[1]):
                        step_match = re.match(r'(\d{5}\.\d{2})\s+(\d{7})\s+(-?\d+\.\d)\s+(-?\d+\.\d{3})\s+(\w+)\s+(\d{5})', line)
                        if step_match:
                            step_frames.append(step_match.groups())
                        else:
                            st.error(f"Step correction format is incorrect in file: {file_object.name}")
                            break
                    else:
                        st.error(f"Incorrect step correction format in file: {file_object.name}")
                        break

                # Check for empty or incomplete clock data lines
                elif len(split_line) < 2 or (len(split_line) == 2 and not split_line[1].isdigit()):
                    st.error(f"Please remove the empty or incomplete lines in the clock data format in file: {file_object.name}")
                    break

                else:
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

def apply_corrections(combined_df, step_data, time_step_selection, freq_step_selection):
    # Iterate over each step correction entry
    for index, step_row in step_data.iterrows():
        step_mjd = step_row['MJD']
        time_step = step_row['Time_Step']
        freq_step = step_row['Frequency_Step']

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


# Function to plot MJD vs Clock Differences for a given TAI code
def plot_clock_differences(df, tai_code, step_data=None, step_correction_options = None):
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
        combined_df = apply_corrections(combined_df, step_data, *step_correction_options)


    # Linear and quadratic fit
    linear_fit = np.polyfit(combined_df['MJD'], combined_df['Clock_Diff'], 1)
    quadratic_fit = np.polyfit(combined_df['MJD'], combined_df['Clock_Diff'], 2)

    # Calculating residuals
    combined_df['Linear_Residuals'] = combined_df['Clock_Diff'] - np.polyval(linear_fit, combined_df['MJD'])
    combined_df['Quadratic_Residuals'] = combined_df['Clock_Diff'] - np.polyval(quadratic_fit, combined_df['MJD'])

    # Streamlit widget for selecting residual type
    # Streamlit widget for selecting residual type
    residual_type = st.radio(
        "Select the type of residual to display:",
        ('None', 'Linear Residuals', 'Quadratic Residuals'),
        key="residual_type"
    )

    # Use the residual_type to plot the data
    if residual_type == 'None':
        fig.add_trace(go.Scatter(x=combined_df['MJD'], y=combined_df['Clock_Diff'], mode='lines+markers', name='Clock Data'))
    elif residual_type == 'Linear Residuals':
        fig.add_trace(go.Scatter(x=combined_df['MJD'], y=combined_df['Linear_Residuals'], mode='lines+markers', name='Linear Residuals'))
    elif residual_type == 'Quadratic Residuals':
        fig.add_trace(go.Scatter(x=combined_df['MJD'], y=combined_df['Quadratic_Residuals'], mode='lines+markers', name='Quadratic Residuals'))

    fig.update_layout(
        title=f'MJD vs {residual_type} for the clock {tai_code}',
        xaxis_title='MJD',
        yaxis=dict(title=f'{residual_type} (ns)' if residual_type != 'None' else 'Clock Difference (ns)'),
        legend_title='Data Series',
        font=dict(size=44)
    )

    return fig

# Main Streamlit app
def main():
    with st.form("my-form1", clear_on_submit=True):
        files_01 = st.file_uploader("Upload the clock data files", accept_multiple_files=True)
        submitted1 = st.form_submit_button("Submit")

    if submitted1 and files_01:
        combined_data, combined_steps = parse_clock_data(files_01)
        st.session_state['combined_data'] = combined_data
        st.session_state['combined_steps'] = combined_steps

    selected_code = None  # Initialize selected_code
    
    if 'combined_data' in st.session_state and not st.session_state['combined_data'].empty:
        tai_code_columns = [col for col in st.session_state['combined_data'].columns if 'TAI_Code' in col]
        all_tai_codes = pd.unique(st.session_state['combined_data'][tai_code_columns].fillna(0).astype(int).values.ravel('K'))
        all_tai_codes = [code for code in all_tai_codes if code != 0]

        selected_code = st.radio("Select a clock", all_tai_codes, key='tai_code_selection', horizontal=True)

    if 'combined_steps' in st.session_state and not st.session_state['combined_steps'].empty:
        st.write("Step Corrections:")
        st.write(st.session_state['combined_steps'])
        st.write("Verify the clock step corrections:")
        # Table of radio buttons for step correction options
        time_step_selection = st.radio("Time Step", ["Not Apply", "Apply", "Apply Reverse"], key='time_step')
        freq_step_selection = st.radio("Freq Step", ["Not Apply", "Apply", "Apply Reverse"], key='freq_step')

        step_correction_options = (time_step_selection, freq_step_selection)


    # Check if a code is selected before plotting
    if selected_code is not None:
        fig = plot_clock_differences(st.session_state['combined_data'], selected_code,st.session_state['combined_steps'], step_correction_options)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No data available for the selected TAI Code.")
    

if __name__ == "__main__":
    main()