# Imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

########################################
# Load data
@st.cache_data
# DF explore
def get_data():
    df = pd.read_csv('data/df_ml.csv')
    df.car_age = df.car_age.astype(int)
    return df
df = get_data()

# DF aggregated
def get_df_agg():
    df_agg = pd.read_csv('data/df_explore.csv')
    return df
df_agg = get_df_agg()

#DF deprication
def load_data():
    df_dep = pd.read_csv('data/df_dep.csv')
    return df_dep

df_dep = load_data()

########################################
#calculate loss value and create class-DFs
df_grouped = df.groupby(['car_class', 'car', 'registration', 'car_age']).mean()['price'].round(2).reset_index()
df_grouped['loss'] = df_grouped['price']
def split_all_cars(df, car):
    return df_grouped[df_grouped['car'] == car].sort_values(['registration'], ascending = False)

cars_name = ['Audi Q3', 'BMW X1', 'Mercedes-Benz GLA 200', 'Volkswagen Tiguan',
    'Audi A5', 'BMW 320', 'Mercedes-Benz C 200',
    'Volkswagen Passat Variant', 'Audi A1', 'BMW i3',
    'Volkswagen Polo GTI', 'smart forFour', 'Audi A3', 'BMW 120',
    'Mercedes-Benz A 200', 'Volkswagen Golf GTI']

dfs = {car: split_all_cars(df, car) for car in cars_name}

# iterate over Dfs in teh dictionary and calculate loss in %
for key, data in dfs.items():
    data['loss'] = ((1-(data['loss'])/data['price'].max())*100).round(2)
df_loss = pd.concat(dfs.values(), ignore_index=True)
#df_loss['loss'] = df_loss['loss'].map(lambda n: '{:,.2f}'.format(n))
#df_loss['price'] = df_loss['price'].map(lambda n: '{:,.2f}'.format(n))


# function for splitting df into DFs per class
def split_all_classes(df, car_class):
    return df_loss[df_loss['car_class'] == car_class]

# assign keys/names for Dfs in teh dictionary
classes_name = ['Small car', 'Small family car', 'Large family car', 'Compact SUV']
dfs = {car_class: split_all_classes(df, car_class) for car_class in classes_name}

# acces Dfs as class name for later EDA
smallCar = dfs['Small car']
smallFamCar = dfs['Small family car']
largeFamCar = dfs['Large family car']
compactSUV = dfs['Compact SUV']

# REName Columns, change Dtypes


########################################
# SET seaborn styles
sns.set_style("whitegrid",
            {"grid.color": "#EBEBEB",
            "grid.linestyle": ":",
             
            'axes.facecolor': 'white',
            'axes.edgecolor': '#00135D',
             
            'text.color': '#00135D',
            'xtick.color': '#212121',
            'ytick.color': '#212121',
             
            'axes.grid': True,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': False,
            'axes.spines.top': False,
             
            'xtick.bottom': True,
            'xtick.top': False,
            'ytick.left': True,
            'ytick.right': False
            })

# set color theme
sns_colors = ["#FF6D43", "#00135D", '#FF9C36', '#1D8DB3', '#949494', '#A1A1A1', '#ADADAD', '#BABABA', '#C7C7C7', '#D4D4D4', '#D4D4D4', '#D4D4D4', '#D4D4D4', '#D4D4D4', '#D4D4D4', '#D4D4D4', '#D4D4D4', '#D4D4D4']
sns.set_palette(sns.color_palette(sns_colors))



########################################
# LAYOUT
def show_explore_page_one_car():
    
    ########################################
    # Sidebar
    #car-filter
    car_filter = st.sidebar.selectbox(
        "Select the car model:",
        options = df.car.sort_values().unique(),
    )
    # Df loss
    df_car_loss = df_loss[['car', 'car_age', 'price', 'loss']].query(
        'car == @car_filter'
    )
    # Df of selected car
    df_car_selection = df.query(
        'car == @car_filter'
    )

    df_car_selection.car_age = df_car_selection.car_age.astype(int) 

    ########################################
    # header KPIs
    st.subheader(f"Your selection: {car_filter}")
    st.markdown(" ")

    # Varibale sKPIs
    total_cars = df_car_selection.car.count()
    percentof_total_cars = ((total_cars/df['car'].count())*100).round(2)
    average_new_price = df_car_loss.price.max()
    max_loss = df_car_loss.sort_values(by=['loss'], ascending=False).iloc[0,3]

    # KPIs 3 columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nr.of cars", f"# {total_cars}")
    col2.metric("Ratio of set", f"% {percentof_total_cars}")
    col3.metric("Avg. price new car", f"€ {average_new_price}")
    col4.metric("Max loss", f"% {max_loss}")
    
    st.markdown(" ")
    st.markdown(" ")

    ########################################
    # Linechart price loss

    colLoss1, colLoss2 = st.columns((3,2), gap="large")
    with colLoss1:
        st.subheader('Price loss from OP, %')

        fig_line_loss = px.line(df_car_loss, x="car_age", y="loss", text="loss", markers=True)
        fig_line_loss.update_traces(textposition="bottom right", line=dict(color='#FF6D43', width=2))
        fig_line_loss.update_layout(xaxis_title='Car age', yaxis_title='Price % loss', xaxis = dict(tickmode = 'array', tickvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),)
        st.plotly_chart(fig_line_loss, use_container_width=True)

    with colLoss2:
        st.table(df_car_loss)   
    
    #########################################
    # Depreciation
    #vars
    df_r = df_dep.query('car == @car_filter')
    r = df_r.iloc[0,2].round(2)
    # Layout
    st.subheader(f"Calculate value of your {car_filter} in future") 
    st.markdown(" ") 

    col_dep1,col_dep2 = st.columns(2,gap = 'large')
    
    with col_dep1:
        # input price
        cost = int(st.text_input(f'Type in the current price:','0'))
        st.write(f"You've selected: {cost}€")
    with col_dep2:
        # input years
        years = int(st.slider('Select years:', 1, 15, 1))
        st.write( f"You've selected: {years} years") 
    
    st.caption("*Calculation bases on average price deprecation, which doesn't consider the individual features of cars and serves as orientation with ±3%")

    # func to calculate deprecation
    def dep_value(cost,years):
        #df_r = df_dep.query('car == @car_filter')
        #r = df_r['depreciation_rate']
        new_value = cost*((1- (r/100))**(years))
        return new_value.round(2)
    # Button calculate & Output
    btn = st.button("Calculate price deprecation")
    if btn:
        depreciated_value = dep_value(cost,years)
        st.markdown(f"""#### The value of your {car_filter} in {years} years will be app. {depreciated_value}€""")
    st.divider()
    #########################################
    #tabs Prices and Losses for same make dif Models
    tab1, tab2, tab3, tab4 = st.tabs(["Volkswagen", "Mercedes-Benz", "BMW", "Audi"])

    df_v = df_loss[df_loss['car'].str.contains('^Volkswagen.*')]
    df_mb = df_loss[df_loss['car'].str.contains('^Mercedes-Benz.*')]
    df_bmw = df_loss[df_loss['car'].str.contains('^BMW.*')]
    df_audi = df_loss[df_loss['car'].str.contains('^Audi.*')]

    with tab1:
        fig_df_v = px.line(df_v, x="car_age", y="loss", text="loss", markers=True, color="car", color_discrete_sequence=('#FF6D43', '#00135D', '#1D8DB3', '#FFA84F'))
        fig_df_v.update_traces(textposition="bottom right")
        fig_df_v.update_layout(title='Volkswagen Models Price Loss, %', xaxis_title='age', yaxis_title='loss', xaxis = dict(tickmode = 'array', tickvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
        st.plotly_chart(fig_df_v, use_container_width=True)


    with tab2:
        fig_df_mb = px.line(df_mb, x="car_age", y="loss", text="loss", markers=True, color="car", color_discrete_sequence=('#FF6D43', '#00135D', '#1D8DB3', '#FFA84F'))
        fig_df_mb.update_traces(textposition="bottom right")
        fig_df_mb.update_layout(title='Mercedes-Benz Models Price Loss, %', xaxis_title='age', yaxis_title='loss', xaxis = dict(tickmode = 'array', tickvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
        st.plotly_chart(fig_df_mb, use_container_width=True)
    
    with tab3:
        fig_df_bmw = px.line(df_bmw, x="car_age", y="loss", text="loss", markers=True, color="car", color_discrete_sequence=('#FF6D43', '#00135D', '#1D8DB3', '#FFA84F'))
        fig_df_bmw.update_traces(textposition="bottom right")
        fig_df_bmw.update_layout(title='BMW Models Price Loss, %', xaxis_title='age', yaxis_title='loss', xaxis = dict(tickmode = 'array', tickvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
        st.plotly_chart(fig_df_bmw, use_container_width=True)

    with tab4:
        fig_df_audi = px.line(df_audi, x="car_age", y="loss", text="loss", markers=True, color="car", color_discrete_sequence=('#FF6D43', '#00135D', '#1D8DB3', '#FFA84F'))
        fig_df_audi.update_traces(textposition="bottom right")
        fig_df_audi.update_layout(title='Audi Models Price Loss, %', xaxis_title='age', yaxis_title='loss', xaxis = dict(tickmode = 'array', tickvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
        st.plotly_chart(fig_df_audi, use_container_width=True)
