import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Function load mL
def load_model():
  with open ('pkl/saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)
  return data

data = load_model()

def get_explore_data():
    df = pd.read_csv('data/df_explore.csv')
    return df

df = get_explore_data()
df['First registration'] = df['First registration'].astype(int)
df['First registration'] = pd.to_datetime(df['First registration'].map('{:.0f}'.format), format='%Y').dt.year

# label decoder vars for ML Model
regressor = data["model"]
le_fuel = data["le_fuel"]
le_gear = data["le_gear"]
le_body = data["le_body"]
le_car_condition = data["le_car_condition"]
le_color = data["le_color"]
le_drive_type = data["le_drive_type"]
le_car_make_model = data["le_car_make_model"]

# global Variables and listst
car_make_models = (' - ', 'Audi A1', 'Audi A3', 'Audi A5', 'Audi Q3',
                       'BMW 120', 'BMW 320', 'BMW X1', 'BMW i3',
                       'smart forFour', 'Mercedes-Benz A 200', 'Mercedes-Benz C 200', 'Mercedes-Benz GLA 200',
                       'Volkswagen Golf GTI', 'Volkswagen Passat Variant', 'Volkswagen Polo GTI', 'Volkswagen Tiguan')

fuels = (' - ', 'Petrol', 'Diesel', 'Electro/Petrol', 'Electro', 'Electro/Diesel')
transmissions = (' - ', 'Automatic', 'Manual', 'Semi-automatic')
colors = (' - ', 'Beige', 'Black', 'Blue', 'Bronze', 'Brown', 'Gold', 'Gray', 'Green', 'Orange', 'Purple', 'Red', 'Silver', 'White', 'Yellow', 'missing')
drive_types = (' - ', 'Front', 'Rear', 'Four.w.d')
body_types = (' - ', 'Sedan', 'Small car', 'Station wagon', 'Convertible', 'Coupe', 'SUV')
car_condition_list = (' - ', 'Demonstration vehicle', 'Day admission', 'Annual car', 'Used', 'New')

# car image lists
img_mb = ('Mercedes-Benz A 200', 'Mercedes-Benz C 200', 'Mercedes-Benz GLA 200', 'smart forFour')
img_audi = ('Audi A1', 'Audi A3', 'Audi A5', 'Audi Q3')
img_bmw = ('BMW 120', 'BMW 320', 'BMW X1', 'BMW i3')
img_vw = ('Volkswagen Golf GTI', 'Volkswagen Passat Variant', 'Volkswagen Polo GTI', 'Volkswagen Tiguan')


# LAYOUT
def show_predict_page_full():

  #HEADER
  head1, head2 = st.columns((5, 3), gap="large")

  with head1:
    st.title("Software car price prediction")
    st.subheader("Estimate the current value of your car")
    st.write("1. To the right, in the image, you can find the representation of car classes, on which data the model was trained to make predictions.")
    st.write("2. Below you find the form. Please fill in all the fields for the AI to estimate the approximate value of your car on the market for used cars.")
    with st.expander("Disclamer"):
      st.write("""
          The chart above shows some numbers I picked for you.
          I rolled actual dice for these, so they're *guaranteed* to
          be random.
      """)
      st.image("images/logo.png")
    with st.expander("Check Data"):
      st.dataframe(data=df,use_container_width=False)
    
  with head2:
    st.image("images/4cars.png", use_column_width=True)
    




  # FILL FORMs
  col1, col2 = st.columns((5, 3), gap="large")

  with col1:
    st.header("Fill in the form and get teh price calculated")
    placeholder = st.empty()

    
    # INPUT FIELDS

    # row A
    A1, A2 = st.columns(2, gap="large")

    with A1:
      # Input car make and model
      car_make_model = st.selectbox("Car make and model", car_make_models)
      # Input condition
      car_condition = st.selectbox("Car condition", car_condition_list)
      #inpit transmission type
      transmission = st.selectbox("Type of transmission", transmissions)
      # Input fuel type
      fuel = st.selectbox("Fuel type", fuels)
      #input drive type
      drive_type = st.selectbox("Type of drive", drive_types)
      #input
      body = st.selectbox("Car body type ", body_types)
      # input registration slider
      year_range = range(2007, 2024) # 2024 not inclusive
      default_year = 2020
      car_registration = st.slider("Select car 1st registration", min_value=min(year_range), max_value=max(year_range), value=default_year)
      st.write("#")

    with A2:
      # input mileage of a car
      mileage = st.number_input("Current mileage in km", min_value=0, step=10)
      # input mileage of a car
      displacement = st.number_input("Engine size")
      # input mileage of a car
      hp = st.number_input("HP of car", min_value=1, step=1)
      #owner
      owner = st.number_input("How many owners (incl.you) have possessed the car", min_value=0, step=1)
      #consumption
      consumption = st.number_input("Combined consumption / if Electro input 0")
      #emission
      emission = st.number_input("Emission / if Electro input 0")
      color = st.selectbox("Car color", colors)
      
      
    
    # prediction button
    btn = st.button("Calculate Preis")
    if btn:
      X = np.array([[fuel, mileage, transmission, car_registration, hp, owner, body, car_condition, consumption, emission, color, displacement, drive_type, car_make_model]])
      X[:,0] = le_fuel.transform(X[:,0])
      X[:,2] = le_gear.transform(X[:,2])
      X[:,6] = le_body.transform(X[:,6])
      X[:,7] = le_car_condition.transform(X[:,7])
      X[:,10] = le_color.transform(X[:,10])
      X[:,12] = le_drive_type.transform(X[:,12])
      X[:,13] = le_car_make_model.transform(X[:,13])
      X = X.astype(float)
      X

      price = regressor.predict(X)
      st.subheader(f"The estimated price is €{price[0]}")

  with col2:
    st.header("Your current selections:")
    
    # show car logo, depending on user car_make_model
    imageLocation = st.empty()
    if car_make_model in img_mb:
      imageLocation.image('images/m.png', width=100)
    elif car_make_model in img_audi:
      imageLocation.image('images/au.png', width=100)
    elif car_make_model in img_bmw:
      imageLocation.image('images/b.png', width=100)
    elif car_make_model in img_vw:
      imageLocation.image('images/v.png', width=100)
    else:
      imageLocation = st.empty()
    
    
    st.write(f"###### Car make: {car_make_model}")
    st.write(f"###### Mileage: {mileage} km")
    st.write(f"###### Car condition: {car_condition}")
    st.write(f"###### Transmission: {transmission}")
    st.write(f"###### Type of fuel: {fuel}")
    st.write(f"###### Drive type: {drive_type}")
    st.write(f"###### Car color: {color}")

    st.write(f"###### Engine size: {displacement}")
    st.write(f"###### HP of car: {hp}")
    st.write(f"###### Owner: {owner}")
    st.write(f"###### consumption comb.: {consumption}")
    st.write(f"###### Emission: {emission}")
    st.write(f"###### Car body type: {body}")
    st.write(f"###### Registration year: {car_registration}")



