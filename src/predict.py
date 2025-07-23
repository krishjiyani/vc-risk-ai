import pandas as pd
import pickle
import joblib

model = joblib.load("src/models/big_small_classifier.pkl")
#model = pickle.load("big_small_classifier.pkl")
new_data = pd.DataFrame([{
    
     "FundingRaisedUSD_millions": 12.5,
    "MonthlyRevenueUSD_thousands": 450,
    "MonthlyGrowthPercent": 35,
    "MarketSizeUSD_billions": 3.5,
    "TAMGrowthPercent": 12
}])

prediction = model.predict(new_data)

label = "big" if prediction[0] == 1 else "Small"
print(f"Prediction: {prediction[0]} => startup is: {label}")