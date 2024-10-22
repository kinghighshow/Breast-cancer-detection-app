
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
df = pd.read_csv("Downloads/Breast cancer dataset cleaned.csv")

# Assuming 'Diagnosis' column contains the labels for whether a person has breast cancer or not
X = df.drop(columns=['Diagnosis'])  # Features
Y = df['Diagnosis']  # Target labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the deep learning model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())  # Optional: add batch normalization for stable learning
model.add(Dropout(0.2))  # Lower dropout rate
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # For binary classification

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0015), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss',  # Metric to monitor
                               patience=5,         # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True)

# Train the model
model.fit(X_train_scaled, y_train, 
          epochs=100, 
          batch_size=32, 
          validation_split=0.15, 
          verbose=1, 
          callbacks=[early_stopping])

# Evaluate the model on the test set and calculate F1 score manually
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test F1 Score: {test_f1:.2f}")

# Streamlit App
logo_path = r"Downloads/logo.png" 
st.image(logo_path, use_column_width='auto')

# Streamlit app title
st.title("BreastGuard: Machine Learning-Based Breast Cancer Prediction System")

# App description
st.write("""
Please Input Corresponding Patient Data
""")

# Sidebar for user inputs
st.header('Patient Input Parameters')

def user_input_features():
    Age = st.number_input('Enter Patient Age', min_value=1, max_value=120, step=1)  # Integer input for age
    gender_options = {
        "Male": 0,
        "Female": 1
    }
    Gender = st.selectbox('Gender', list(gender_options.keys()))  # Display gender options

    # Get the selected gender value
    Gender = gender_options[Gender]  # Map to numeric value

    # Create a dictionary to map labels to values for laterality
    laterality_options = {
        "Left": 1,
        "Right": 0
    }
    laterality = st.selectbox('Laterality', list(laterality_options.keys()))  # Display laterality options

    # Get the selected laterality value
    laterality = laterality_options[laterality]  # Map to numeric value

    # Create a dictionary to map labels to values for lymph node
    lymph_node_options = {
        "No": 0,
        "Yes": 1
    }
    Lymph_Node = st.selectbox('Lymph Node', list(lymph_node_options.keys()))  # Display lymph node options

    # Get the selected lymph node value
    Lymph_Node = lymph_node_options[Lymph_Node]

    
    
    Nature_of_Aspirate = st.selectbox('Nature of Aspirate', 
                                       ["colloid_Aspirate", "creamy_Aspirate", "hemorrhagic_Aspirate", 
                                        "milky_Aspirate", "mucoid_Aspirate", "oily_Aspirate", 
                                        "proteinaceous_Aspirate", "sanguineous_Aspirate", 
                                        "serous_Aspirate", "turbid_Aspirate"])
    
    data = {
        'Age': Age,
        'Gender': Gender,
        'laterality': laterality,
        'Lymph_Node': Lymph_Node,
        'Nature_of_Aspirate': Nature_of_Aspirate
    }

    # Convert the input data into a DataFrame
    features = pd.DataFrame(data, index=[0])
    
    # Align the features with the training data (adding missing columns if necessary)
    features = features.reindex(columns=X_train.columns, fill_value=0)
    
    return features

input_df = user_input_features()

if st.button('Submit'):
    # Scale the user input
    input_scaled = scaler.transform(input_df)
    st.write(input_df)
    
    # Make predictions based on user input
    prediction = model.predict(input_scaled)
    prob_malignant = prediction[0][0]  # Probability of malignant (1)
    prob_benign = 1 - prob_malignant   # Probability of benign (0)

    # Display the prediction results
    st.subheader('Prediction')
    
    if prob_malignant < 0.70 and prob_malignant > 0.5:
        st.write("Most likely malignant, further testing advised.")
    elif prob_malignant > 0.70:
        st.write(f"Predicted Diagnosis: {'Malignant' if prob_malignant >= 0.5 else 'Benign'}")

    if prob_benign < 0.70 and prob_benign > 0.5:
        st.write("Most likely benign, further testing advised.")
    elif prob_benign > 0.70:
        st.write(f"Predicted Diagnosis: {'Benign' if prob_malignant < 0.5 else 'Malignant'}")

    st.subheader('Prediction Probability')
    st.write(f"Probability of Malignant: {prob_malignant:.2f}")
    st.write(f"Probability of Benign: {prob_benign:.2f}")
    