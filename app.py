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
    return int(rounded_value / 40)

# Fungsi untuk mengubah nilai
def map_removability(value):
    return "Removable" if value == 1 else "Not Removable"

# Menggunakan fungsi map untuk mengubah nilai kolom
df['Battery Removable'] = df['Battery Removable'].map(map_removability)


# Define weights for each column
weights = {
    'Weight (Gram)': lambda x: 1 / x,  # Smaller weight is better
    'Price (£)': lambda x: 1 / x,  # Smaller price is better
    'Length (mm)': lambda x: 1 / x,  # Smaller length is better
    'Width (mm)': lambda x: 1 / x,  # Smaller width is better
    'Thickness (mm)': lambda x: 1 / x,  # Smaller thickness is better
    'Display Size Pixels': lambda x: display_size_weight(*x.split(' x ')),  # Use the display_size_weight function
    'Internal Memory (GB)': lambda x: x,  # Larger internal memory is better
    'Primary Camera MP': lambda x: x,  # Larger MP is better
    'Secondary Camera MP': lambda x: x,  # Larger MP is better
    'Battery (mAh)': lambda x: x,  # Larger mAh is better
    'Battery Removable': lambda x: 5 if x == "Removable" else 0,  # Higher weight if removable
    'Screen to Body Ratio': lambda x: x,  # Larger ratio is better
    'CPU (MB)': lambda x: x,  # Larger speed is better
    'RAM (MB)': lambda x: x,  # Larger RAM size is better
    'Display Type': lambda x: 4 if x == 'Super AMOLED' else 3 if x == 'AMOLED' else 2 if x == 'IPS LCD' else 1 if x == 'TFT' else 0,
    'Color Support': lambda x: 3 if x == '16M' else 2 if x == '256K' else 1 if x == '65K' else 0,  # Higher weight for 16M color support
    'Additional Features': lambda x: 3 if 'Super AMOLED Plus' in x else 2 if 'Super Flexible AMOLED' in x else 1 if 'SC-LCD' in x else 0,
}

# Streamlit web app

col_1, col_2, col_3 = st.tabs(['Overview Smartphone Dataset', 'Weighting Smartphone', 'Add New Smartphone']) 

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
    # Apply weights to each column
    for col in weights:
        if col in df.columns:
            df_weight[col + ' Score'] = df[col].apply(weights[col])

    df_weight['Display Pixels Score'] = df_weight[['Width (mm) Score', 'Length (mm) Score', 'Thickness (mm) Score']].sum(axis=1)
    df_weight.drop(['Width (mm) Score', 'Length (mm) Score', 'Thickness (mm) Score'], inplace=True, axis=1)

    # Checkbox to select models
    selected_models = st.multiselect('Select models for scoring', df['Model'].unique())

    # Filter DataFrame based on selected models
    filtered_df_weight = df_weight[df_weight.index.isin(df[df['Model'].isin(selected_models)].index)]

    # Calculate overall score for the filtered DataFrame
    filtered_df = df[df['Model'].isin(selected_models)]
    
    # Convert selected columns to numeric if not already
    numeric_columns = list(weights.keys())
    filtered_df[numeric_columns] = filtered_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Ensure there are no NaN values after conversion
    filtered_df = filtered_df.dropna(subset=numeric_columns, how='all')

    # Calculate overall score
    filtered_df['Overall Score'] = filtered_df[numeric_columns].sum(axis=1)

    # Display the filtered DataFrame and overall score
    st.subheader('Weight/Scoring Dataset')
    st.dataframe(filtered_df_weight)

    # Display the sorted dataset based on overall score for the filtered DataFrame
    sorted_filtered_df = filtered_df.sort_values(by='Overall Score', ascending=False)
    st.subheader('Sorted Dataset based on Overall Score (Filtered)')
    st.dataframe(sorted_filtered_df[['Brand', 'Model', 'Overall Score']])

