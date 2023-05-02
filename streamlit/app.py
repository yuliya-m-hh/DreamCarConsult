# Imports
import streamlit as st
from predict_page import show_predict_page
from explore_all_cars import show_explore_page_all_cars
from explore_one_car import show_explore_page_one_car

#########################################
# header
head1, head2 = st.columns((5,1))

with head1:
  st.title("Dashboard to Explore and Predict Car Prices")
  st.image("images/car-classes.png")

#########################################
# Sidebar
st.sidebar.image("images/Logo.png", use_column_width=True)
st.sidebar.subheader("Explore or predict used cars' prices")
page = st.sidebar.radio("Select Page:",("Estimate current value", "Explore all cars", "Explore individual car"))

# Radiobuttons

if page == "Estimate current value":
  show_predict_page()
#elif page == "Predict Full":
#  show_predict_page_full()
elif page == "Explore individual car":
  show_explore_page_one_car()
else:
  show_explore_page_all_cars()

st.sidebar.divider()
########################################
#Expander Sidebar
with st.sidebar.expander("Car Classes by Make and Model"):
    st.write("1) **Small car**:")
    st.caption("Audi A1 - BMW i3 - smart forFour - VW Polo")
    st.write("2) **Small family car**:")
    st.caption("Audi A3 - BMW 120 - MB  A200 - VW Golf")
    st.write("3) **Large family car**:")
    st.caption("Audi A5 - BMW 320 - MB  C200 - VW Passat")
    st.write("3) **Compact SUV**:")
    st.caption("Audi Q3 - BMW X1 - MB GLA 200 - VW Tiguan")
########################################

 