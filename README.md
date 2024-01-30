### Executive Summary:

As Data Scientists at `Second Wind Wheels`, a leading online used car retailer, we propose a project to develop a machine learning model for predicting the accurate market value of used cars. This model will revolutionize our pricing strategy, boosting profits and enhancing customer satisfaction.

**Problem Statement**:

Currently, used car pricing relies heavily on manual valuations by human appraisers. This method is prone to subjectivity, inconsistencies, and delays, leading to:
Overpricing: Cars priced too high stagnate on the lot, incurring storage costs and lost sales.

1. Underpricing: We miss out on maximizing profits by selling cars for less than their true market value.

2. Customer dissatisfaction: Unfairly priced cars may discourage buyers and damage brand reputation.

To address this business challenge, we will leverage Second Wind Wheels existing used car sales data along with additional data scraped from various automotive websites to develop a machine learning model that can accurately predict used car prices.

### Data Pertinence & Attribution:
The model will be developed using up-to-date used car listing data scraped from major automotive sales sites to incorporate broader market pricing trends. It includes details on used car sales across the Japan.

### Data Collection
Wescraped our data from SBT Japan. And cleaned it.
Some of the steps included:
 1. Dealing with rare categories. If they are too many to list we either group them into a single category or drop the categories.

 2. Dealing with outliers greater than a certain upper bound or a certain lower bound. We either capped them or removed them.

 3. Dropping redundant columns.

 4. Feature Engineering new coulns form the already existing ones.

 5. Dealing with missing values.

 6. Dealing with duplicates and ensuring that they are indeed duplicate values.

 7. Dealing with typos and mislabeled classes.


### EDA
Some of our findings were: 
**Popularity Of Models**

 In the Toyota category, Prius, Landcruiser Prado, and Crown hybrid are the top three most frequently encountered models.

 For Mercedes, the E, C, and GLC models are prevalent.

 Among BMW models, the 5 Series, 7 Series, and X5 stand out.

**Year of Manufacture and mileage**
Most vehicles  were manufactured in 2021, followed by 2020 and 2022. Fewer vehicles were manufactured in 2024.

Vehicles manufactured in earlier years tend to have higher mileage compared to recent years.

**Fuel Types**

Petrol-powered vehicles are predominant, followed by hybrid vehicles. Electric and other fuel types are the least popular.

**Transmission and Mileage Impact on Price**

Vehicles with automatic transmission generally have higher mean prices compared to those with manual transmission.

Vehicles with lower mileage tend to have higher mean prices, while those with higher mileage are relatively cheaper.

### Modelling
Our methodology involved leveraging both Random Forest Regressor and XGBoost Regressor, accompanied by grid searches to optimize on their respective parameters, thereby enhancing their performance.

Initially, the Random Forest Regressor Model yielded an r2 score of 0.8514. However, after hyperparameter tuning the score improved to 0.8535, indicating a slight improvement in predictive accuracy.

Due to the improved scores we proceeded to evaluate our XGBoost model using the refined parameters obtained from the hyperparameter tuning process. Impressively, the XGBoost model demonstrated a superior performance with an r2 score of 0.8616.

Due to the improved scores we designated the XGBOOST model as our final model for subsequent analyses and applications.

### Deployment

We carried out Deployment and we used streamlit to build our web application. Our webapplication allows users to predict car prices. It also has personalization where the user can pick the kind if features he would want on the car.

To gain a comprehensive understanding of our project, we invite you to explore the presentation overview by executing the following command:run **streamlit run main.py**

To experience the full functionality of our application, allowing users to predict car prices and customize their preferred car features, please run the following command: **streamlit run app3.py**

Our interactive application provides a seamless interface for users to input car-related and receive predictions and also tailor their preferences to suit their individual needs
