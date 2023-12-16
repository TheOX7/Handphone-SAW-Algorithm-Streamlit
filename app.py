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
    'Internal Memory (GB)': lambda x: x * 0.25,  # Larger internal memory is better
    'Primary Camera MP': lambda x: x * 0.25,  # Larger MP is better
    'Secondary Camera MP': lambda x: x * 0.25,  # Larger MP is better
    'Battery (mAh)': lambda x: x * 0.20,  # Larger mAh is better
    'Screen to Body Ratio': lambda x: x * 0.20,  # Larger ratio is better
    'CPU (MB)': lambda x: x * 0.20,  # Larger speed is better
    'RAM (MB)': lambda x: x * 0.20,  # Larger RAM size is better
    'Battery Removable': lambda x: 5 if x == "Removable" else 0, 
    'Display Size Pixels': lambda x: display_size_weight(*x.split(' x ')) * 0.25,  # Use the display_size_weight function
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

    
    col_1_1, col_1_2 = st.columns(2)
    with col_1_1 :
        # Select a row from the DataFrame based on Model
        selected_model = st.selectbox('Select a phone model', df['Model'].unique())

        # Get the selected row based on the selected model
        selected_row = df[df['Model'] == selected_model].index[0]

        # Display the image of the selected row
        selected_phone_image_url = df.at[selected_row, 'Image URL']
        st.image(selected_phone_image_url, caption=f"Selected Phone - {df.at[selected_row, 'Brand']} {df.at[selected_row, 'Model']}",
                width=200,  # Set the desired width
                use_column_width=False)  # Set to False to use the specified width    with col_1_2 :
        st.text(f"{selected_model}")
    
with col_2:
    with st.expander('Edit Weights') :
        # Edit weights for each column
        st.subheader('Edit Weights for Each Columns')

        # Create a dictionary to store edited weights
        edited_weights = {}

        # Split the features into groups of 4
        feature_groups = [list(weights.keys())[i:i+4] for i in range(0, len(weights), 4)]

        for group in feature_groups:
            # Create columns for each group
            weight_columns = st.columns(len(group))

            for col, weight_column in zip(group, weight_columns):
                if col not in ['Battery Removable', 'Display Size Pixels', 'Display Type', 'Color Support', 'Additional Features']:
                    # Use spinner to edit weights
                    current_weight = weights[col](0)  # Get the current weight value by applying the lambda function to a dummy value
                    weight_value = weight_column.number_input(f'{col} Weight', min_value=0.0, max_value=1.0, value=current_weight, step=0.01)
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

    with st.expander('Create New Smartphone Data'):
        # Create
        st.subheader('Create New Smartphone Data')
        new_model = st.text_input('Model:', '')
        new_brand = st.text_input('Brand:', '')
        # ... add other input fields for the remaining columns

        if st.button('Add Smartphone'):
            # Add the new smartphone to the DataFrame
            new_data = {'Model': new_model, 'Brand': new_brand}
            # ... add other fields to new_data dictionary
            df = df.append(new_data, ignore_index=True)
            st.success(f'Smartphone {new_model} added successfully!')

    with st.expander('Update Smartphone Data') :
        # Update
        st.subheader('Update Smartphone Data')
        selected_update_model = st.selectbox('Select a phone model to update', df['Model'].unique())
        selected_update_row = df[df['Model'] == selected_update_model].index[0]

        # Display the current details of the selected smartphone
        st.text(f"Current Brand: {df.at[selected_update_row, 'Brand']}")
        # ... display other current details

        # Allow user to update fields
        new_brand_update = st.text_input('New Brand:', df.at[selected_update_row, 'Brand'])
        # ... add other input fields for the remaining columns

        if st.button('Update Smartphone'):
            # Update the selected smartphone with new values
            df.at[selected_update_row, 'Brand'] = new_brand_update
            # ... update other fields
            st.success(f'Smartphone {selected_update_model} updated successfully!')

    with st.expander('Delete Smartphone Data') :
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
