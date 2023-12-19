import streamlit as st
import pandas as pd
import math
import numpy as np

st.set_page_config(
    page_title="Recommendation Samsung Smartphone ",
    layout='wide'
)

st.title('Samsung Recommendation using Simple Additive Weighting')
url_editor = "https://www.linkedin.com/in/marselius-agus-dhion-374106226/"
github_url = "https://github.com/TheOX7/Handphone-SAW-Algorithm-Streamlit"
st.markdown(f'Streamlit App by [Marselius Agus Dhion]({url_editor})', unsafe_allow_html=True)
st.markdown(f'GitHub Repository → {github_url}', unsafe_allow_html=True)

# Load dataset
df = pd.read_csv('cleaned_data_df.csv')
df_weight = pd.DataFrame()

def display_size_weight(p, l):
    width = int(p)
    height = int(l)
    sqrt_value = math.sqrt(width * height)
    rounded_value = np.round(sqrt_value, 0)
    return rounded_value

# Fungsi untuk mengubah nilai
def map_removability(value):
    return "Removable" if value == 1 else "Not Removable"

# Menggunakan fungsi map untuk mengubah nilai kolom
df['Battery Removable'] = df['Battery Removable'].map(map_removability)


# Define weights for each column
weights = {
    'Weight (Gram)': lambda x: x * 0.25,  # Smaller weight is better
    'Price (£)': lambda x: x * 0.25,  # Smaller price is better
    'Length (mm)': lambda x: x * 0.25,  # Smaller length is better
    'Width (mm)': lambda x: x * 0.25,  # Smaller width is better
    'Thickness (mm)': lambda x: x * 0.25,  # Smaller thickness is better
    'Display Size Pixels': lambda x: display_size_weight(*x.split(' x ')) * 0.25,  # Use the display_size_weight function
    'Internal Memory (GB)': lambda x: x * 0.25,  # Larger internal memory is better
    'Primary Camera MP': lambda x: x * 0.25,  # Larger MP is better
    'Secondary Camera MP': lambda x: x * 0.25,  # Larger MP is better
    'Battery (mAh)': lambda x: x * 0.20,  # Larger mAh is better
    'Battery Removable': lambda x: 5 if x == "Removable" else 0, 
    'Screen to Body Ratio': lambda x: x * 0.20,  # Larger ratio is better
    'CPU (MB)': lambda x: x * 0.20,  # Larger speed is better
    'RAM (MB)': lambda x: x * 0.20,  # Larger RAM size is better
    'Display Type': lambda x: 4 if x == 'Super AMOLED' else 3 if x == 'AMOLED' else 2 if x == 'IPS LCD' else 1 if x == 'TFT' else 0,
    'Color Support': lambda x: 3 if x == '16M' else 2 if x == '256K' else 1 if x == '65K' else 0,  # Higher weight for 16M color support
    'Additional Features': lambda x: 3 if 'Super AMOLED Plus' in x else 2 if 'Super Flexible AMOLED' in x else 1 if 'SC-LCD' in x else 0,
}

# Streamlit web app

col_1, col_2, col_3 = st.tabs(['Overview Smartphone Dataset', 'Weighting Smartphone', 'CRUD Smartphone Data']) 

with col_1:
    # Display the original dataset
    st.subheader('Original Dataset')
    df_original = df.copy()
    st.dataframe(df_original.drop(['Brand', 'Image URL'], axis=1))

    
    col_1_1, col_1_2, col_1_3 = st.columns([1,2,1])
    with col_1_2 :
        st.subheader('Selected Smartphone Specifications')
        # Select a row from the DataFrame based on Model
        selected_model = st.selectbox('Select a phone model', df['Model'].unique())

        # Get the selected row based on the selected model
        selected_row = df[df['Model'] == selected_model].index[0]

        col_1_2_1, col_1_2_2, col_1_2_3 = st.columns([2,3,4])
        
        with col_1_2_1 :
            # Display the image of the selected row
            selected_phone_image_url = df.at[selected_row, 'Image URL']
            st.image(selected_phone_image_url, caption=f"{df.at[selected_row, 'Brand']} {df.at[selected_row, 'Model']}",
                    width=175,  # Set the desired width
                    use_column_width=False)  # Set to False to use the specified width

        with col_1_2_2:            
            # Keenam spesifikasi pertama
            for feature, weight_function in weights.items():
                if feature in ['Length (mm)', 'Width (mm)', 'Thickness (mm)', 'Battery Removable', 'Additional Features', 'Screen to Body Ratio',
                               'CPU (MB)', 'RAM (MB)', 'Display Size Pixels', 'Display Type', 'Color Support']:
                    # Skip features that are not numerical
                    continue
                
                current_value = df.at[selected_row, feature]
                st.text(f"{feature}: {current_value}")
                            
        with col_1_2_3:    
            # Keenam spesifikasi terakhir
            for feature, weight_function in weights.items():
                if feature in ['Length (mm)', 'Width (mm)', 'Thickness (mm)', 'Battery Removable', 'Additional Features', 'Screen to Body Ratio',
                               'Weight (Gram)', 'Price (£)', 'Internal Memory (GB)', 'Primary Camera MP', 'Secondary Camera MP', 'Battery (mAh)']:
                    # Skip features that are not numerical
                    continue

                current_value = df.at[selected_row, feature]
                st.text(f"{feature}: {current_value}")       
                 
            # Display Phone Dimension
            length_value = df.at[selected_row, 'Length (mm)']
            width_value = df.at[selected_row, 'Width (mm)']
            thickness_value = df.at[selected_row, 'Thickness (mm)']
            st.text(f"Dimensions : {length_value} mm x {width_value} mm x {thickness_value} mm")


    
with col_2:
    with st.expander('Edit Weights (Click Here)') :
        # Edit weights for each column
        st.subheader('Edit Weights for Each Columns (Decimals to Percentage)')

        # Create a dictionary to store edited weights
        edited_weights = {}

        # Split the features into groups of 4
        filter_keys = ['Battery Removable', 'Display Size Pixels', 'Display Type', 'Color Support', 'Additional Features']
        filtered_weights = {key: value for key, value in weights.items() if key not in filter_keys}
        feature_groups = [list(filtered_weights.keys())[i:i+4] for i in range(0, len(filtered_weights), 4)]

        for group in feature_groups:
            # Create columns for each group
            weight_columns = st.columns(len(group))

            for col, weight_column in zip(group, weight_columns):
                if col not in ['Battery Removable', 'Display Size Pixels', 'Display Type', 'Color Support', 'Additional Features']:
                    # Use spinner to edit weights
                    current_weight = weights[col](0)  # Get the current weight value by applying the lambda function to a dummy value
                    weight_value = weight_column.number_input(f'{col}', min_value=0.0, max_value=1.0, value=current_weight, step=0.01)
                    edited_weights[col] = weight_value

        # Update weights dictionary with edited values
        for col in edited_weights:
            weights[col] = lambda x, col=col: x * edited_weights[col]

        # Apply updated weights to each column
        for col in weights:
            if col in df.columns:
                df_weight[col + ' Score'] = df[col].apply(weights[col])
                
    # Checkbox to select models
    selected_models = st.multiselect('Select models for scoring', df['Model'].unique())

    # Filter DataFrame based on selected models
    filtered_df_weight = df_weight[df_weight.index.isin(df[df['Model'].isin(selected_models)].index)]

    # Calculate overall score for the filtered DataFrame
    overall_score_column_names = [col + ' Score' for col in weights]
    filtered_df_weight['Overall Score'] = filtered_df_weight[overall_score_column_names].sum(axis=1)

    # Display the filtered DataFrame and overall score
    st.subheader('Weight/Scoring Dataset')
    st.dataframe(filtered_df_weight)

    # Display the sorted dataset based on overall score for the filtered DataFrame
    sorted_filtered_df = df.loc[df['Model'].isin(selected_models)].merge(
        filtered_df_weight[['Overall Score']], left_index=True, right_index=True
    ).sort_values(by='Overall Score', ascending=False)

    st.subheader('Sorted Dataset based on Overall Score (Filtered)')
    st.dataframe(sorted_filtered_df[['Brand', 'Model', 'Overall Score']])


with col_3:
    st.subheader('CRUD Smartphone Data')

    with st.expander('Create New Smartphone Data (Click Here)'):
        st.subheader('Create New Smartphone Data')
        
        col_3_1, col_3_2, col_3_3, col_3_4, col_3_5, col_3_6 = st.columns(6)
        with col_3_1 :            
            model = st.text_input('Model Name', '')
            weight = st.number_input('Weight (Gram)', min_value=0.0, step=0.1)
            price = st.number_input('Price (£)', min_value=0)
            
        with col_3_2 :
            length = st.slider('Length (mm)', min_value=0.0, max_value=500.0, step=0.1)
            width = st.slider('Width (mm)', min_value=0.0, max_value=500.0, step=0.1)
            thickness = st.slider('Thickness (mm)', min_value=0.0, max_value=500.0, step=0.1)
            
        with col_3_3 :
            internal_memory = st.number_input('Internal Memory (GB)', min_value=0)
            primary_camera = st.number_input('Primary Camera MP', min_value=0)
            secondary_camera = st.number_input('Secondary Camera MP', min_value=0)

        with col_3_4 :
            cpu = st.number_input('CPU (MB)', min_value=0)
            ram = st.number_input('RAM (MB)', min_value=0)
            display_type = st.selectbox('Display Type', ('Super AMOLED', 'AMOLED', 'PLS', 'IPS LCD', 'TFT'), key='display_type_update')
            
            

        with col_3_5 :
            size_pixels = st.text_input('Display Size Pixels (P x L)', '')
            battery = st.number_input('Battery (mAh)', min_value=0)
            screen_ratio = st.slider('Screen to Body Ratio (%)', min_value=0.0, max_value=100.0, step=0.1)

        with col_3_6 :
            battery_removable = st.radio('Battery Removable', ['Removable', 'Not Removable'])
            color_support  = st.radio('Color Support', ['16M', '256K', '65K'])
            additional_information = st.text_input('Additional Information', '')

            
            
        if st.button('Add Smartphone'):
            # Add the new smartphone to the DataFrame
            new_data = {
                'Model': model, 'Brand': 'Samsung',
                'Weight (Gram)' : weight, 'Price (£)' : price,
                'Length (mm)' : length, 'Width (mm)' : width, 'Thickness (mm)': thickness,
                'Display Size Pixels' : size_pixels, 'Internal Memory (GB)' : internal_memory,
                'Primary Camera MP' : primary_camera, 'Secondary Camera MP' : secondary_camera,
                'Battery (mAh)' : battery, 'Battery Removable' : battery_removable,
                'Screen to Body Ratio' : screen_ratio, 'CPU (MB)' : cpu, 'RAM (MB)' : ram,
                'Display Type' : display_type, 'Color Support' : color_support,
                }
            df = df.append(new_data, ignore_index=True)
            st.success(f'Smartphone {model} added successfully!')

    with st.expander('Update Smartphone Data (Click Here)'):
        # Update
        st.subheader('Update Smartphone Data')
        selected_update_model = st.selectbox('Select a phone model to update', df['Model'].unique())
        selected_update_row = df[df['Model'] == selected_update_model].index[0]

        # Display the current details of the selected smartphone
        st.text(f"Current Brand: {df.at[selected_update_row, 'Brand']}")

        col_3_2_1, col_3_2_2, col_3_2_3, col_3_2_4, col_3_2_5, col_3_2_6 = st.columns(6)
        with col_3_2_1 :            
            model_update = st.text_input('Model Name', df.at[selected_update_row, 'Model'])
            weight_update = st.number_input('Weight (Gram)', min_value=0.0, step=0.1, value=df.at[selected_update_row, 'Weight (Gram)'])
            price_update = st.number_input('Price (£)', min_value=0, value=df.at[selected_update_row, 'Price (£)'])
            
        with col_3_2_2 :
            length_update = st.slider('Length (mm)', min_value=0.0, max_value=500.0, step=0.1, value=df.at[selected_update_row, 'Length (mm)'])
            width_update = st.slider('Width (mm)', min_value=0.0, max_value=500.0, step=0.1, value=df.at[selected_update_row, 'Width (mm)'])
            thickness_update = st.slider('Thickness (mm)', min_value=0.0, max_value=500.0, step=0.1, value=df.at[selected_update_row, 'Thickness (mm)'])
            
        with col_3_2_3 :
            internal_memory_update = st.number_input(
                'Internal Memory (GB)', min_value=float(0), max_value=float(10000),  
                step=1.0, value=float(df.at[selected_update_row, 'Internal Memory (GB)']))
            primary_camera_update = st.number_input(
                'Primary Camera MP', min_value=float(0), max_value=float(16), 
                value=float(df.at[selected_update_row, 'Primary Camera MP']))
            secondary_camera_update = st.number_input(
                'Secondary Camera MP', min_value=float(0), max_value=float(16), 
                value=float(df.at[selected_update_row, 'Secondary Camera MP']))

        with col_3_2_4 :
            cpu_update = st.number_input(
                'CPU (MB)', min_value=int(0), max_value=int(10000),
                value=int(df.at[selected_update_row, 'CPU (MB)']))
            ram_update = st.number_input(
                'RAM (MB)', min_value=int(0), max_value=int(10000),
                value=int(df.at[selected_update_row, 'RAM (MB)']))
            
            display_type_options = ['Super AMOLED', 'AMOLED', 'PLS', 'IPS LCD', 'TFT']
            default_display_type = df.at[selected_update_row, 'Display Type']

            if default_display_type not in display_type_options:
                default_display_type = display_type_options[0]

            selected_display_update = st.selectbox(
                'Display Type',
                options=display_type_options,
                index=display_type_options.index(default_display_type)
            )

        with col_3_2_5 :
            size_pixels_update = st.text_input('Display Size Pixels (P x L)', value=df.at[selected_update_row, 'Display Size Pixels'])
            battery_update = st.number_input('Battery (mAh)', min_value=0, value=int(df.at[selected_update_row, 'Battery (mAh)']))
            screen_ratio_update = st.slider('Screen to Body Ratio (%)', min_value=0.0, max_value=100.0, step=0.1, value=df.at[selected_update_row, 'Screen to Body Ratio'])

        with col_3_2_6:
            battery_removable_update = st.radio(
                'Battery Removable',
                ['Removable', 'Not Removable'],
                index=1 if df.at[selected_update_row, 'Battery Removable'] == 'Not Removable' else 0,
                key='battery_removable_update'
            )
            color_support_update = st.radio(
                'Color Support',
                ['16M', '256K', '65K'],
                index=['16M', '256K', '65K'].index(df.at[selected_update_row, 'Color Support']),
                key='color_support_update'
            )
            additional_information_update = st.text_input(
                'Additional Information',
                value=df.at[selected_update_row, 'Additional Features'],
                key='additional_information_update'
            )

        # Allow user to update fields
        new_brand_update = st.text_input('New Brand:', df.at[selected_update_row, 'Brand'])

        if st.button('Update Smartphone'):
            # Update the selected smartphone with new values
            df.at[selected_update_row, 'Brand'] = new_brand_update
            df.at[selected_update_row, 'Model'] = model_update
            df.at[selected_update_row, 'Weight (Gram)'] = weight_update
            df.at[selected_update_row, 'Price (£)'] = price_update
            df.at[selected_update_row, 'Length (mm)'] = length_update
            df.at[selected_update_row, 'Width (mm)'] = width_update
            df.at[selected_update_row, 'Thickness (mm)'] = thickness_update
            df.at[selected_update_row, 'Display Size Pixels'] = size_pixels_update
            df.at[selected_update_row, 'Internal Memory (GB)'] = internal_memory_update
            df.at[selected_update_row, 'Primary Camera MP'] = primary_camera_update
            df.at[selected_update_row, 'Secondary Camera MP'] = secondary_camera_update
            df.at[selected_update_row, 'Battery (mAh)'] = battery_update
            df.at[selected_update_row, 'Screen to Body Ratio'] = screen_ratio_update
            df.at[selected_update_row, 'CPU (MB)'] = cpu_update
            df.at[selected_update_row, 'RAM (MB)'] = ram_update
            df.at[selected_update_row, 'Display Type'] = selected_display_update
            df.at[selected_update_row, 'Color Support'] = color_support_update
            df.at[selected_update_row, 'Battery Removable'] = 'Not Removable' if battery_removable_update == 1 else 'Removable'
            df.at[selected_update_row, 'Additional Features'] = additional_information_update

            st.success(f'Smartphone {selected_update_model} updated successfully!')
    with st.expander('Delete Smartphone Data (Click Here)') :
        # Delete
        st.subheader('Delete Smartphone Data')
        selected_delete_model = st.selectbox('Select a phone model to delete', df['Model'].unique())

        if st.button('Delete Smartphone'):
            # Delete the selected smartphone from the DataFrame
            df = df[df['Model'] != selected_delete_model]
            st.success(f'Smartphone {selected_delete_model} deleted successfully!')
  
    # Display the updated DataFrame
    st.subheader('Updated Dataset')
    st.dataframe(df.drop(['Brand', 'Image URL'], axis=1))
