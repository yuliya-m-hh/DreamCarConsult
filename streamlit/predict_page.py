import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

########################################
# Function load mL
def load_model():
  with open ('pkl/model_rf.pkl', 'rb') as file:
    data = pickle.load(file)
  return data

data = load_model()

# get data
def get_explore_data():
    df = pd.read_csv('data/df_explore.csv')
    return df

df = get_explore_data()
df['First registration'] = df['First registration'].astype(int)
df['First registration'] = pd.to_datetime(df['First registration'].map('{:.0f}'.format), format='%Y').dt.year

# label decoder vars for ML Model
regressor = data["model"]
le_make = data["le_make"]
le_model = data["le_model"]
le_fuel = data["le_fuel"]
le_gear = data["le_gear"]
le_car_class = data["le_car_class"]

# global Variables and listst
makes =('-',  'Audi', 'BMW', 'Mercedes-Benz',  'smart', 'Volkswagen')
models_audi = ('-','A1', 'A3', 'A5', 'Q3')
models_bmw = ('-','i3', '120', '320', 'X1')
models_mb = ('-','A 200', 'C 200', 'GLA 200')
models_vw = ('-','Polo GTI', 'Golf GTI', 'Passat Variant', 'Tiguan')
models_smart = ('-','forFour')
smallCars=('A1', 'i3','Polo GTI', 'forFour')
smallFamCars=('A3', '120', 'A 200', 'Golf GTI')
largeFamCars=('A5', '320', 'C 200', 'Passat Variant')
compactSUV=('Q3', 'X1', 'GLA 200', 'Tiguan')
fuels = (' - ', 'Petrol', 'Diesel', 'Electro')
transmissions = (' - ', 'Automatic', 'Manual', 'Semi-automatic')
car_classes = ('-', 'Small car', 'Small family car', 'Large family car', 'Compact SUV')


########################################
# LAYOUT
def show_predict_page():

  ##########################################################
  # Form fields to fill in 
  # col1 - Forms
  # col2 - summary

  col1, col2 = st.columns((6, 3), gap="large")

  with col1:
    st.subheader("Fill in the forms to estimate the car value")

    # INPUT FIELDS
    fields1, fields2 = st.columns(2, gap="large")

    with fields1:

      # Make
      make = st.selectbox("Car make", makes)

      # Model
      model = '-'
      modelLocation = st.empty()
      if make == 'Audi':
        model = modelLocation.selectbox("Car model", models_audi)
      elif make == 'BMW':
        model = modelLocation.selectbox("Car model", models_bmw)
      elif make == 'Mercedes-Benz':
        model = modelLocation.selectbox("Car model", models_mb)
      elif make == 'Volkswagen':
        model = modelLocation.selectbox("Car model", models_vw)
      elif make == 'smart':
        model = modelLocation.selectbox("Car model", models_smart)
      else:
        modelLocation = st.empty()
        
      #mileage
      mileage = st.number_input("Current mileage in km", min_value=0, step=10)

      # registration 
      year_range = range(2011, 2024) # 2024 not inclusive
      default_year = 2018
      registration = st.slider("Select first registration", min_value=min(year_range), max_value=max(year_range), value=default_year)

      # Car Class
      car_class = ' '
      if model in smallCars:
        car_class = car_classes[1]
      elif model in smallFamCars:
        car_class = car_classes[2]
      elif model in largeFamCars:
        car_class = car_classes[3]
      elif model in compactSUV:
        car_class = car_classes[4]

      st.write(f"Car class: {car_class}")

    with fields2:
      # fuel
      fuel = st.selectbox("Fuel type", fuels)
      # HP
      hp = st.number_input("HP of car", min_value=40, step=10)
      # transmission
      gear = st.selectbox("Type of transmission", transmissions)
     
    # prediction button
    btn = st.button("Calculate estimated preis")
    if btn:
      X = np.array([[make, model, fuel, mileage, gear, registration, hp, car_class]])
      X[:,0] = le_make.transform(X[:,0])
      X[:,1] = le_model.transform(X[:,1])
      X[:,2] = le_fuel.transform(X[:,2])
      X[:,4] = le_gear.transform(X[:,4])
      X[:,7] = le_car_class.transform(X[:,7])
      X = X.astype(float)
      X

      price = regressor.predict(X)
      st.subheader(f"The estimated price is €{(price[0]).round(2)}")

  ##########################################################
  # col2 - selections summary
  with col2:
    st.subheader("Your selections:")
    
    # show car logo
    imageLocation = st.empty()
    if make == 'Mercedes-Benz':
      imageLocation.image('images/m.png', width=100)
    elif make == 'Audi':
      imageLocation.image('images/au.png', width=100)
    elif make == 'BMW':
      imageLocation.image('images/b.png', width=100)
    elif make == 'Volkswagen':
      imageLocation.image('images/v.png', width=100)
    else:
      imageLocation = st.empty()
    
    # variables0selections
    st.write(f"###### Car make: {make}")
    st.write(f"###### Model: {model}")
    st.write(f"###### Mileage: {mileage} km")
    st.write(f"###### First registration: {registration}")
    st.write(f"###### Type of fuel: {fuel}")
    st.write(f"###### HP of car: {hp}")
    st.write(f"###### Transmission: {gear}")
    st.write(f"###### Car class: {car_class}")