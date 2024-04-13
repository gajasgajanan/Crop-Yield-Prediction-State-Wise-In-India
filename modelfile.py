import pickle
import pandas as pd

# Load the pre-trained model
model_loaded = pickle.load(open('./dtree_model','rb'))
print(model_loaded)

# Load the preprocessor
preprocessor = pickle.load(open('./pp','rb'))

def prediction(State_Name, District_Name, Crop_Year, Season, Crop, Area):
    # Organize input features into a DataFrame with a single row
    input_data = pd.DataFrame({
        'State_Name': [State_Name],
        'District_Name': [District_Name],
        'Crop_Year': [Crop_Year],
        'Season': [Season],
        'Crop': [Crop],
        'Area': [Area]
    })
    
    # Ensure all features are present during transformation
    # (Some preprocessor implementations require all features to be present)
    all_features = ['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 'Area']
    input_data = input_data.reindex(columns=all_features)
    
    # Transform the input data using the preprocessor
    transformed_features = preprocessor.transform(input_data)
    
    # Make predictions using the model
    predicted_value = model_loaded.predict(transformed_features)
    
    return predicted_value


State_Name = 'Madhya Pradesh'
District_Name = 'DINDORI'
Crop_Year = 2019
Season = 'Rabi'
Crop = 'Rapeseed &Mustard'
Area = 17209.00
    
print(int(prediction(State_Name, District_Name, Crop_Year, Season, Crop, Area)[0]))





