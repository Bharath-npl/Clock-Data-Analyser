# *********************************************
# ****************** Clock Data Analyser ******

# to tune this code in your local PC use the follwoing command **
# streamlit run .\clk_data_analyser.py --server.port 8888

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import warnings
import numpy as np
import io
import re
#from ipywidgets import interact, RadioButtons
import warnings 

# Ignore specific warnings by message, category, etc.
warnings.filterwarnings("ignore", message="`label` got an empty value.*")
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Clock Data Analyser", page_icon=":stopwatch:", layout="wide")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
st.title(":clock10: Clock Data Analyser ")


st.sidebar.image("https://www.bipm.org/documents/20126/27072194/Logo+BIPM+blue.png/797bb4e6-8cfb-5d78-b480-e460ad5eeec2?t=1583920059345", width=200)
#One line of gap 
st.sidebar.write("")
#IEEE UFFSC logo
st.sidebar.image("https://www.fusfoundation.org/images/IEEE-UFFC.jpg", width=200)
st.sidebar.write("")
st.sidebar.write("")

st.sidebar.header("Time & Frequency Capacity Building")

st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
# Sidebar button to open PDF
# if st.sidebar.button('Clock Data Format'):
#     st.sidebar.markdown('BIPM clock data format:', unsafe_allow_html=True)
#     st.sidebar.markdown('[Open PDF](https://webtai.bipm.org/database/documents/clock-data_format.pdf)', unsafe_allow_html=True)

# if st.sidebar.button('Clock Data Format'):
#     # st.sidebar.markdown('BIPM clock data format:', unsafe_allow_html=True)
#     st.sidebar.markdown('[BIPM clock data format](https://webtai.bipm.org/database/documents/clock-data_format.pdf)', unsafe_allow_html=True)


# Function to parse the clock data files

# Updated Function to parse the clock data files
def parse_clock_data(file_objects):
    data_frames = []
    step_frames = []
    for file_object in file_objects:
        try:
            string_io = io.StringIO(file_object.getvalue().decode('utf-8'))
            data_started = False  # Flag to indicate we're in the data section

            for line in string_io:
                
                # Skip lines starting with #
                if line.startswith('#'):
                    continue
                
                if not data_started:
                    # Check if the line represents the start of the data section
                    if re.match(r'^\d{5}', line.strip()):
                        data_started = True
                    else:
                        continue  # Skip the line if data hasn't started

                # Skip empty lines or lines not containing data
                if line.strip() == '':
                    data_started = False  # Reset flag if an empty line is encountered
                    continue

                split_line = line.strip().split()
                
                # Check for step data format
                # Regex format to correctly parse step data
                step_match = re.match(r'(\d{5}(?:\.\d{2})?)\s+(\d{7})\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+(\w+)\s+(\d{5})', line.strip())
                if step_match:
                    # If a match is found, process the step data
                    step_frames.append(step_match.groups())
                    continue  # Skip the rest of the loop if step data is processed

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
    if combined_steps.empty:
        # st.write("No correction steps are mentioned in the Uploaded files")
        combined_steps = pd.DataFrame()
        
    # st.write(combined_steps)
    return combined_data, combined_steps


def apply_corrections(combined_df, step_data, user_selected_steps, counter):
    # Ensure MJD in combined_df is numeric and sorted
    combined_df['MJD'] = pd.to_numeric(combined_df['MJD'], errors='coerce')
    combined_df.sort_values(by='MJD', inplace=True)

    # The following code should check, if there are any non zero counter number then apply that many corrections 
    if counter > 0 and not step_data.empty:
        # Sort step_data by MJD to ensure corrections are applied in order
        step_data = step_data.sort_values(by='MJD')
        
        # Iterate over the steps up to the counter value
        for step_index in range(counter):
            step_row = step_data.iloc[step_index]  # Get the step row by index
            step_mjd = pd.to_numeric(step_row['MJD'], errors='coerce')
            before_step = combined_df['MJD'] < step_mjd  # Identify points before the step
            


            # Apply time step correction based on the selection for this step
            if user_selected_steps[step_index][0] != "Not Apply":
                time_step = -pd.to_numeric(step_row['Time_Step'], errors='coerce') # if step is potivive it is removed from the pqst hence the negqtive sign
                correction_factor = (-time_step if user_selected_steps[step_index][0] == "Apply Reverse"
                                     else time_step)
                combined_df.loc[before_step, 'Clock_Diff'] += correction_factor

            # Apply frequency step correction based on the selection for this step
            if user_selected_steps[step_index][1] != "Not Apply":
                freq_step = pd.to_numeric(step_row['Frequency_Step'], errors='coerce')
                # st.write(f"Frequency step: {freq_step}")
                correction_direction = (-1 if user_selected_steps[step_index][1] == "Apply Reverse" else 1)
                day_difference = step_mjd - combined_df.loc[before_step, 'MJD']  # Days difference from the day of correction (MJD)
               
                correction_amount = (day_difference* correction_direction * (freq_step)) # Assuming freq_step is in seconds/day, and correction amount is calculated per day
                combined_df.loc[before_step, 'Clock_Diff'] += correction_amount  # Apply the correction

    return combined_df


# Function to plot MJD vs Clock Differences for a given TAI code
def plot_clock_differences(df, tai_code, step_data, step_correction_options, step_correction_counter):
    fig = go.Figure()
    
    # Dynamically find TAI code columns
    tai_code_columns = [col for col in df.columns if 'TAI_Code' in col]
    clock_diff_columns = [col.replace('TAI_Code', 'Clock_Diff') for col in tai_code_columns]

    # Filter step data for the given TAI code
    if not step_data.empty:
        step_data = step_data.loc[step_data['Clock_Code'] == str(tai_code)]

    # st.write(step_data)
    mjd_combined = []
    clock_diff_combined = []

    for tai_col, clock_diff_col in zip(tai_code_columns, clock_diff_columns):
        if tai_col in df.columns and clock_diff_col in df.columns:
            subset = df[df[tai_col] == tai_code]
            if not subset.empty:
                mjd_combined.extend(subset['MJD'])
                clock_diff_combined.extend(subset[clock_diff_col])
   
    # Convert MJD and Clock_Diff to numeric and sort the data
    combined_df = pd.DataFrame({
        'MJD': pd.to_numeric(mjd_combined, errors='coerce'), 
        'Clock_Diff': pd.to_numeric(clock_diff_combined, errors='coerce')
    }).drop_duplicates().sort_values(by='MJD')


     # Create a copy to preserve original 'Clock_Diff' before corrections
    combined_df['original_Clock_Diff'] = combined_df['Clock_Diff'].copy()


    if step_data is not None and step_correction_options is not None:
        # st.write(f"step_correction_option zero Time Step: {step_correction_options[3][0]}")
        # st.write(f"step_correction_option one Freqeuncy step: {step_correction_options[3][1]}")        
        combined_df = apply_corrections(combined_df, step_data, step_correction_options, step_correction_counter)
       
    # Create a DataFrame for comparison
    comparison_df = combined_df[['MJD', 'original_Clock_Diff', 'Clock_Diff']].copy()
    comparison_df.rename(columns={'original_Clock_Diff': 'clk_data', 'Clock_Diff': 'Corrected_Clk_data'}, inplace=True)

    # Linear and quadratic fit
    linear_fit = np.polyfit(combined_df['MJD'], combined_df['Clock_Diff'], 1)
    quadratic_fit = np.polyfit(combined_df['MJD'], combined_df['Clock_Diff'], 2)

    # Calculating residuals
    combined_df['Linear_Residuals'] = combined_df['Clock_Diff'] - np.polyval(linear_fit, combined_df['MJD'])
    combined_df['Quadratic_Residuals'] = combined_df['Clock_Diff'] - np.polyval(quadratic_fit, combined_df['MJD'])

    # Consecutive differences in 'Clock_Diff' column
    combined_df['First_Derivative'] = combined_df['Clock_Diff'].diff().fillna(0)
    
    
    residual_type = st.radio(":white_square_button: **Select the type of Data to display:**",
        ('Clock Data', 'Linear Residuals', 'Quadratic Residuals', 'First Derivative'),
        key="residual_type", horizontal = True )
    
    fig = go.Figure()  # Initialize the figure object
    
    # Use the residual_type to plot the data
    if residual_type == 'Clock Data':
        fig.add_trace(go.Scatter(x=combined_df['MJD'], y=combined_df['Clock_Diff'], mode='lines+markers', name='Clock Data'))
    elif residual_type == 'Linear Residuals':
        fig.add_trace(go.Scatter(x=combined_df['MJD'], y=combined_df['Linear_Residuals'], mode='lines+markers', name='Linear Residuals'))
    elif residual_type == 'Quadratic Residuals':
        fig.add_trace(go.Scatter(x=combined_df['MJD'], y=combined_df['Quadratic_Residuals'], mode='lines+markers', name='Quadratic Residuals'))
    elif residual_type == 'First Derivative':
        fig.add_trace(go.Scatter(x=combined_df['MJD'][1:], y=combined_df['First_Derivative'][1:], mode='lines+markers', name='First Derivative'))


    fig.update_layout(
        title=f'{residual_type} for the clock {tai_code}',
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
    return fig, comparison_df

# Main Streamlit app
def main():
    
        # CSS to reduce the spacing between the radio button rows
    st.markdown("""
        <style>
        .stRadio > div > div {
            margin-top: -25px;
            display: flex;
            margin-bottom: -25px;
            margin-top: -5px;
            margin-right: -10px;
            height: 20px; /* Adjust height as needed */
            flex-wrap: nowrap;
        }
        </style>
        """, unsafe_allow_html=True)



    valid_filenames = []
    step_correction_options = []
    step_counter= 0
    # Initialize 'step_correction_counter' in st.session_state
    if 'step_correction_counter' not in st.session_state:
        st.session_state['step_correction_counter'] = 0
        
        
    with st.form("my-form1", clear_on_submit=True):
        files_01 = st.file_uploader("**Upload the clock data files**", accept_multiple_files=True)
        submitted1 = st.form_submit_button("Submit")
    
    if files_01:
        
        # Extract filenames and store them in a list
        filenames = [file.name for file in files_01]
        
        # Append the filenames
       
        for filename in filenames:
            valid_filenames.append(filename)

        # Write the list of valid filenames in a row
        if valid_filenames:
            st.write(f"Files uploaded: {', '.join(valid_filenames)}")
        else:
            st.write("No valid files found.")

        combined_data, combined_steps = parse_clock_data(files_01)
        st.session_state['combined_data'] = combined_data
        st.session_state['combined_steps'] = combined_steps

    selected_code = None  # Initialize selected_code
    # st.session_state["step_correction_options"] = None
    
    if 'combined_data' in st.session_state and not st.session_state['combined_data'].empty:
        tai_code_columns = [col for col in st.session_state['combined_data'].columns if 'TAI_Code' in col]
        all_tai_codes = pd.unique(
            st.session_state['combined_data'][tai_code_columns]
            .fillna('0')  # Fill NaNs with '0'
            .applymap(lambda x: str(x).replace(' ', ''))  # Convert to string and remove spaces for all elements
            .astype(int)  # Convert to integer
            .values.ravel('K')
            )
        all_tai_codes = [code for code in all_tai_codes if code != 0]

        selected_code = st.radio(":white_square_button: **Select a clock**", all_tai_codes, key='tai_code_selection', horizontal=True )
        # step_correction_options= None

    if 'combined_steps' in st.session_state and not st.session_state['combined_steps'].empty:
              
        # Filter step corrections for the selected clock
        filtered_steps = st.session_state['combined_steps'][st.session_state['combined_steps']['Clock_Code'] == str(selected_code)]

        if len(filtered_steps) > 0:
            step_counter = len(filtered_steps)
            if 'step_correction_counter' not in st.session_state:
                st.session_state['step_correction_counter'] = 0

        # Check if the selected clock matches any Clock_Code in step data
        if str(selected_code) in st.session_state['combined_steps']['Clock_Code'].unique():
            st.session_state['combined_steps']['Clock_Code'] = st.session_state['combined_steps']['Clock_Code'].astype(str)

            if str(selected_code) in st.session_state['combined_steps']['Clock_Code'].unique():
                filtered_steps = filtered_steps.sort_values(by=filtered_steps.columns[0]) # Arrange the filtered steps in ascending order of their MJD 
                # st.write(":white_square_button: Steps corrections mentioned in the data for this clock")
                filtered_steps = st.session_state['combined_steps'][st.session_state['combined_steps']['Clock_Code'] == str(selected_code)].reset_index(drop=True)

                # Create a new index starting from 1
                filtered_steps.index = range(1, len(filtered_steps) + 1)

                
                # Custom CSS to inject into the Streamlit app
                # This CSS is to change the background color
                
                st.markdown("""
                <style>
                .css-1v3fvcr {
                    background-color: #f0ad4e; /* Bootstrap's button warning color */
                    color: white !important;
                }
                .css-1v3fvcr:hover {
                    background-color: #ec971f; /* Darker shade for hover effect */
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Layout for the main columns
                main_col1, maincol_space, main_col2 = st.columns([6,0.5, 8])  # Adjust the ratio as needed

                with main_col1:
                    with st.expander(":white_square_button: **Steps corrections mentioned in the data for this clock**"):
                        # st.write(filtered_steps)
                        st.table(filtered_steps)

                with main_col2:

                    with st.expander(":large_orange_diamond: **Apply the step corrections to this clock**"):
                
                        if len(filtered_steps) > 0:
                            # Display labels once, above the first row
                            col1_space, col1, col2_space, col2, col3_space = st.columns([1.5, 2, 3, 2, 4])
                            with col1:
                                st.write(":white_square_button: **Time Step** 👇")
                            with col2:
                                st.write(":white_square_button: **Freq Step** 👇")

                            # Initialize manual index for display purposes
                            display_index = 0
                            
                            st.markdown("""
                                <style>
                                /* Attempt to reduce spacing between Streamlit widgets, including radio buttons */
                                .stRadio > div {
                                    margin-bottom: -20px !important;
                                }
                                /* Further reduce spacing around the markdown used for indexes */
                                .stMarkdown {
                                    margin-bottom: -15px !important;
                                    padding-top: 0px !important;
                                    padding-bottom: 0px !important;
                                }
                                </style>
                                """, unsafe_allow_html=True)
                            
                            # Iterate over the rows of the DataFrame to create dynamic rows of radio buttons without repeating labels
                            for _, row in filtered_steps.iterrows():
                                col_index, col1, col2, _ = st.columns([0.5, 4, 4, 2])
                                with col_index:
                                    # Display the manual serial number/index for each row
                                    st.markdown(f"<div style='margin-top: 28px;'>{display_index+1}.</div>", unsafe_allow_html=True)
                                with col1:
                                    # Use the display index to create a unique key for each set of radio buttons for Time Step
                                    key_time_step = f'time_step_{display_index}'
                                    time_step_selection = st.radio(
                                        "",  # The label is now managed above, so this is left empty
                                        ["Not Apply", "Apply", "Apply Reverse"], 
                                        key=key_time_step, 
                                        horizontal=True, 
                                        label_visibility="hidden"  # Hides the label, assuming your Streamlit version supports it
                                    )

                                with col2:
                                    # Use the index to create a unique key for each set of radio buttons for Freq Step
                                    key_freq_step = f'freq_step_{display_index}'
                                    freq_step_selection = st.radio(
                                        "Frequency",  # Label removed as it's already displayed above
                                        ["Not Apply", "Apply", "Apply Reverse"], 
                                        key=key_freq_step, 
                                        horizontal=True,
                                        label_visibility = "hidden"
                                    )
                                
                                # Increment the manual index after processing each row
                                display_index += 1

                                # Store selections for each step
                                step_correction_options.append((time_step_selection, freq_step_selection))

    # Check if a code is selected before plotting
    if selected_code is not None:
        fig, verify_data = plot_clock_differences(st.session_state['combined_data'], selected_code,st.session_state['combined_steps'], step_correction_options, step_counter)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No data available for the selected TAI Code.")

            # Option to, print the DataFrame to output for verification
        if not verify_data.empty:
            
            verify_colm1, verify_colm2 = st.columns([3,7])
            
            with verify_colm1:
                st.write(':white_square_button: **Clock Data Overview**')
                # Create a new index starting from 1
                verify_data.index = range(1, len(verify_data) + 1)
                st.table(verify_data)
    

if __name__ == "__main__":
    main()


st.sidebar.markdown('---')  # Add a horizontal line for separation
st.sidebar.markdown('**Contact:   tf.cbkt@bipm.org**')
st.sidebar.markdown('**For more Information visit:**')
st.sidebar.markdown("[**BIPM e-Learning Platform**](https://e-learning.bipm.org/)")
