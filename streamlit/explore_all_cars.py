# Imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Car Prices Predict & Explore", page_icon = ":bar_chart:", layout="wide")

########################################
# Load data
@st.cache_data
# DF explore
def get_data():
    df = pd.read_csv('data/df_ml.csv')
    return df
df = get_data()

# DF aggregated
def get_df_agg():
    df_agg = pd.read_csv('data/df_explore.csv')
    return df
df_agg = get_df_agg()

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
def show_explore_page_all_cars():
    
    #########################################
    #tabs Prices and Losses
    tab1, tab2, tab3, tab4 = st.tabs(["Max loss", "Max price", "Avg Loss car class", "Avg price car class"])
    

    with tab1:
        st.subheader('Max. avg. price loss decade, %')

        loss_max = df_loss.groupby(['car']).max()['loss'].reset_index()
        loss_max.sort_values('loss', ascending=False, inplace=True)

        fig_max_loss = px.bar(loss_max, x='car', y='loss', text="loss",  height=550)
        fig_max_loss.update_traces(textfont_size=14, textposition="outside", marker_color='#FF6D43', marker_line_color='#00135D', marker_line_width=1.5, opacity=0.7)
        fig_max_loss.update_layout(xaxis_title='Price loss, %', yaxis_title=None)
        st.plotly_chart(fig_max_loss, use_container_width=True)
        st.caption("*Max price loss for small cars only for ca. 6 years")

    with tab2:
        st.subheader('Max. avg. price new car, €')

        df_prices_max = df_loss.groupby(["car_class", 'car']).max()['price'].reset_index()
        df_prices_max.sort_values('price', ascending=False, inplace=True)
        
        fig_max_price = px.bar(df_prices_max, x='car', y='price', text="price", height=550)
        fig_max_price.update_traces(textfont_size=14, textposition="outside", marker_color='#FF6D43', marker_line_color='#00135D', marker_line_width=1.5, opacity=0.7)
        fig_max_price.update_layout(xaxis_title='Price, €', yaxis_title=None)
        st.plotly_chart(fig_max_price, use_container_width=True)

    with tab3:
        st.subheader('Price loss with years, %')

        fig_avg_loss = px.histogram(df_loss, x='car_age', y='loss', color="car_class", barmode="group", histfunc='avg', text_auto='.2f', height=500, hover_data=df_loss.columns, color_discrete_sequence=('#FF6D43', '#00135D', '#1D8DB3', '#FFA84F'), opacity=0.85)
        fig_avg_loss.update_layout(xaxis = dict(tickmode = 'array', tickvals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
)
        st.plotly_chart(fig_avg_loss, use_container_width=True)

    with tab4:
        st.subheader('Avg price per car class vs. car age, €')

        age_price_plot = px.histogram(df_loss, x='car_age', y='price', color="car_class", barmode="group", histfunc='avg', text_auto='.2f', height=500, hover_data=df_loss.columns, color_discrete_sequence=('#FF6D43', '#00135D', '#1D8DB3', '#FFA84F'), opacity=0.85)
        age_price_plot.update_layout(xaxis = dict(tickmode = 'array', tickvals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
)
        st.plotly_chart(age_price_plot, use_container_width=True)

    ########################################
    # Explore cars and features - Barplot counts
    st.subheader("Dataset features and values")
    st.markdown(" ")
    col_barplot1, col_barplot2 = st.columns((1,3), gap="large")
    
    columns = ['car', 'registration', 'fuel', 'gear', 'body', 'color', 'owner', 'drive_type', 'car_class', 'car_age' ]
    with col_barplot1:
        bar_plot_filter = st.selectbox(label='Select car model:', options=columns)

    count_df = df[bar_plot_filter].value_counts().rename_axis(bar_plot_filter).reset_index(name='counts')
    fig_bar_count = px.bar(count_df, x=count_df[bar_plot_filter], y="counts", text="counts")
    fig_bar_count.update_traces(textfont_size=14, textposition="outside", cliponaxis=False, marker_color='#FF6D43', marker_line_color='#00135D', marker_line_width=1.5, opacity=0.7)
    st.plotly_chart(fig_bar_count, use_container_width=True)
    
    st.markdown(" ")
    st.markdown(" ")

   #########################################
    #tabs Prices and Losses
    tab1, tab2, tab3, tab4 = st.tabs(["Small Car", "Small family car", "Large family car", "Compact SUV"])
    
    with tab1:
        fig_smallCar_loss = px.line(smallCar, x="car_age", y="loss", text="loss", markers=True, color="car", color_discrete_sequence=('#FF6D43', '#00135D', '#1D8DB3', '#FFA84F'))
        fig_smallCar_loss.update_traces(textposition="bottom right")
        fig_smallCar_loss.update_layout(title='Small Cars Price Loss, %', xaxis_title='age', yaxis_title='loss', xaxis = dict(tickmode = 'array', tickvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
        st.plotly_chart(fig_smallCar_loss, use_container_width=True)
        st.caption("*BMW i3 & smart forFour start with second 0, which corresponds to the first registration of 2022")


    with tab2:
        fig_smallFamCar_loss = px.line(smallFamCar, x="car_age", y="loss", text="loss", markers=True, color="car", color_discrete_sequence=('#FF6D43', '#00135D', '#1D8DB3', '#FFA84F'))
        fig_smallFamCar_loss.update_traces(textposition="top right")
        fig_smallFamCar_loss.update_layout(title='Small Family Cars Price Loss, %', xaxis_title='age', yaxis_title='loss', xaxis = dict(tickmode = 'array', tickvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
        st.plotly_chart(fig_smallFamCar_loss, use_container_width=True)
    
    with tab3:
        fig_largeFamCar_loss = px.line(largeFamCar, x="car_age", y="loss", text="loss", markers=True, color="car", color_discrete_sequence=('#FF6D43', '#00135D', '#1D8DB3', '#FFA84F'))
        fig_largeFamCar_loss.update_traces(textposition="top right")
        fig_largeFamCar_loss.update_layout(xaxis = dict(tickmode = 'array', tickvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), title='Large Family Cars Price Loss, %', xaxis_title='age', yaxis_title='loss')
        st.plotly_chart(fig_largeFamCar_loss, use_container_width=True)

    with tab4:
        fig_SUV_loss = px.line(compactSUV, x="car_age", y="loss", text="loss", markers=True, color="car", color_discrete_sequence=('#FF6D43', '#00135D', '#1D8DB3', '#FFA84F'))
        fig_SUV_loss.update_traces(textposition="top left")
        fig_SUV_loss.update_layout(title='Compact SUV Price Loss, %', xaxis_title='age', yaxis_title='loss', xaxis = dict(tickmode = 'array', tickvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
        st.plotly_chart(fig_SUV_loss, use_container_width=True)
