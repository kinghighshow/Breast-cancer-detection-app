import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

# Load the dataset
df = pd.read_csv("Breast cancer dataset cleaned.csv")

# Assuming 'Diagnosis' column contains the labels for whether a person has breast cancer or not
X = df.drop(columns=['Diagnosis'])  # Features
Y = df['Diagnosis']  # Target labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the deep learning model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # For binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0015), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train_scaled, y_train, 
          epochs=100, 
          batch_size=32, 
          validation_split=0.15, 
          verbose=1, 
          callbacks=[early_stopping])

# Evaluate the model
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test F1 Score: {test_f1:.2f}")

# Streamlit App
st.image("logo.png", use_column_width=True)

st.title("BreastGuard: Machine Learning-Based Breast Cancer Prediction System")
st.write("Please Input Corresponding Patient Data")

st.header('Patient Input Parameters')

def user_input_features():
    # Input fields
    Age = st.number_input('Enter Patient Age', min_value=1, max_value=120, step=1)
    
    Gender = st.selectbox('Gender', ["Male", "Female"])
    Gender = 0 if Gender == "Male" else 1

    Laterality = st.selectbox('Laterality', ["Left", "Right"])
    Laterality = 1 if Laterality == "Left" else 0

    Lymph_Node = st.selectbox('Lymph Node', ["No", "Yes"])
    Lymph_Node = 1 if Lymph_Node == "Yes" else 0

    Familial_cancer = st.selectbox('Familial cancer', ["No", "Yes"])
    Familial_cancer = 1 if Familial_cancer == "Yes" else 0

    # Aspirate nature
    aspirate_options = {
        "Colloid Aspirate": "colloid_Aspirate",
        "Creamy Aspirate": "creamy_Aspirate",
        "Hemorrhagic Aspirate": "hemorrhagic_Aspirate",
        "Milky Aspirate": "milky_Aspirate",
        "Mucoid Aspirate": "mucoid_Aspirate",
        "Oily Aspirate": "oily_Aspirate",
        "Proteinaceous Aspirate": "proteinaceous_Aspirate",
        "Sanguineous Aspirate": "sanguineous_Aspirate",
        "Serous Aspirate": "serous_Aspirate",
        "Turbid Aspirate": "turbid_Aspirate"
    }
    Nature_of_Aspirate = st.selectbox('Nature of Aspirate', list(aspirate_options.keys()))
    Nature_of_Aspirate = aspirate_options[Nature_of_Aspirate]

    # Tumor shape
    shape_options = {
        "Lobulated": "lobulated",
        "Nodular": "nodular",
        "Oval": "oval",
        "Round": "round",
        "Stellate": "stellate"
    }
    Tumor_Shape = st.selectbox('Tumor Shape', list(shape_options.keys()))
    Tumor_Shape = shape_options[Tumor_Shape]

    # Create input dictionary
    data = {
        'Age': Age,
        'Gender': Gender,
        'Laterality': Laterality,
        'Lymph_Node': Lymph_Node,
        'Familial_cancer': Familial_cancer,
        'Nature_of_Aspirate': Nature_of_Aspirate,
        'Tumor_Shape': Tumor_Shape
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

if st.button('Submit'):
    # Ensure input matches the training set columns
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    prob_malignant = prediction[0][0]
    prob_benign = 1 - prob_malignant

    # Display results
    st.subheader('Prediction')
    if prob_malignant >= 0.5:
        st.write("Predicted Diagnosis: **Malignant**")
    else:
        st.write("Predicted Diagnosis: **Benign**")

    st.subheader('Prediction Probability')
    st.write(f"Probability of Malignant: **{prob_malignant:.2f}**")
    st.write(f"Probability of Benign: **{prob_benign:.2f}**")
