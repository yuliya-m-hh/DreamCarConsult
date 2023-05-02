# DreamCarConsult
**Manual: how a Data Scientist buys a car?**

**Project Core Points**:
- Exploratative Analyses of used cars markets for Germany and UK -> **EDA**
- Evaluate the current value of a used car -> **Price modelling (only DE)**
- Predict car value depreciation for future -> **Prediction modelling (only DE)**

**Data Sources**:
1. German Car Market: the data got via webscraping from [Web site AutoScout24](https://www.autoscout24.de/)
2. UK Car Market: [Kaggle Datasource](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes?select=audi.csv), which also was scraped by the author, from download link. 

## Data Set Characteristics:

- Obtained: scraped in the 2. half of April 2023 (german Market), Kaggle Download (UK market)
- Multivariate dataset
- Shape of the dataset: 10976 rows, 20 columns
- Area: busines, used cars markets UK and Germany
- Attribute Characteristics: Categorical, Integer, Float, String
- Date priods: cars offerd for buying in April 2023, with first registration from 2023 to 2012.
- Associated Tasks: EDA, Regression (Car price prediction)
- Missing Values?: Yes

**Variables Description:**

|Variable|Definition   | Key  |
|---|---|---|
|make |Car Manufacturer  |   Audi,BMW, Volkswagon,Mercedes|
|model|Car Model within each Manufacturer| |
|fuel|Fuel Type|'Petrol', 'Diesel', 'Electro'|
|mileage|Km stand of car|in KM|
|gear|Gear Type |Automatic,manual|
|registration|year of first registration of car|Year|
|hp|Engine Power|kW|
|owner|no. of Previous Owner ||
|body|Car Type|Sedan, Small car, Station wagon, Convertible, Coupe, SUV|
|car_condition|Demonstration vehicle, Day admission, Annual car, Used, New||
|consumption|Fuel Consumption|in l/Km|
|emission|Exhaust emission|in g/Km|
|color|car color||
|car_id|unique carid||
|displacement|Engine Size|in cm3|
|drive_type|Types of drivetrain|Front,Rear, Four wheel drive(four w.d.)|
|link|link of car description||
|price|price of car on Autoscout24.de| in EUR, €|

## Requirements Setup
pyenv with Python: 3.9.8
```python
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
