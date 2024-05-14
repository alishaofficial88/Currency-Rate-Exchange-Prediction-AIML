#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from seaborn import regression
sns.set()


data = pd.read_csv("INR=X (2).csv")
print(data.head())


# In[4]:


data.info()


# In[5]:


print(data.isnull().sum())


# In[ ]:





# In[6]:


figure = px.line(data, x="Date", 
                 y="Close", 
                 title='USD - INR Conversion Rate over the year')
figure.show()


# In[7]:


data.shape


# In[8]:


x = data[["Open", "High", "Low"]]
y = data["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)


# In[9]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)


# In[10]:


data = pd.DataFrame(data={"Predicted Rate": ypred.flatten()})
print(data.head())


# In[11]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Compute Mean Squared Error (MSE)
mse = mean_squared_error(ytest, ypred)

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(ytest, ypred)

# Compute R-squared (R2) score
r2 = r2_score(ytest, ypred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)


# ## INR/EUR

# In[13]:


d = pd.read_csv("EURINR=X.csv")
print(d.head())


# In[14]:


d.info()


# In[15]:


print(d.isnull().sum())


# In[16]:


fig_eur = px.line(d, x="Date", 
                 y="Close", 
                 title='INR - EUR Conversion Rate over the years')
fig_eur.show()


# In[19]:


x_eur = d[["Open", "High", "Low"]]
y_eur = d["Close"]
x_eur = x_eur.to_numpy()
y_eur = y_eur.to_numpy()
y_eur= y_eur.reshape(-1, 1)


# In[20]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_eur, y_eur, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(Xtrain, Ytrain)
Ypred = model.predict(Xtest)


# In[21]:


d = pd.DataFrame(data={"Predicted Rate": Ypred.flatten()})
print(d.head())


# In[22]:


mse_eur = mean_squared_error(Ytest, Ypred)

# Compute Mean Absolute Error (MAE)
mae_eur = mean_absolute_error(Ytest, Ypred)

# Compute R-squared (R2) score
r2_eur = r2_score(Ytest, Ypred)

print("Mean Squared Error (MSE):", mse_eur)
print("Mean Absolute Error (MAE):", mae_eur)
print("R-squared (R2) Score:", r2_eur)


# In[24]:


import tkinter as tk
from tkinter import ttk
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# Load the model
model_usd = DecisionTreeRegressor()
model_usd.fit(xtrain, ytrain)  # Assuming Xtrain_usd, Ytrain_usd are already defined

model_eur = DecisionTreeRegressor()
model_eur.fit(Xtrain, Ytrain)  # Assuming Xtrain_eur, Ytrain_eur are already defined

def predict_rates():
    # Get user input
    open_usd = float(entry_open_usd.get())
    high_usd = float(entry_high_usd.get())
    low_usd = float(entry_low_usd.get())
    
    open_eur = float(entry_open_eur.get())
    high_eur = float(entry_high_eur.get())
    low_eur = float(entry_low_eur.get())
    
    # Make predictions
    pred_usd = model_usd.predict([[open_usd, high_usd, low_usd]])
    pred_eur = model_eur.predict([[open_eur, high_eur, low_eur]])
    
    # Update result labels
    label_result_usd.config(text=f"Predicted close rate (USD): {pred_usd[0]}")
    label_result_eur.config(text=f"Predicted close rate (EUR): {pred_eur[0]}")

# Create main window
window = tk.Tk()
window.title("Currency Exchange Rate Prediction")

# Create USD input fields
label_usd = ttk.Label(window, text="Enter rates for USD:")
label_usd.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

label_open_usd = ttk.Label(window, text="Open rate:")
label_open_usd.grid(row=1, column=0, padx=10, pady=5, sticky="e")
entry_open_usd = ttk.Entry(window)
entry_open_usd.grid(row=1, column=1, padx=10, pady=5)

label_high_usd = ttk.Label(window, text="High rate:")
label_high_usd.grid(row=2, column=0, padx=10, pady=5, sticky="e")
entry_high_usd = ttk.Entry(window)
entry_high_usd.grid(row=2, column=1, padx=10, pady=5)

label_low_usd = ttk.Label(window, text="Low rate:")
label_low_usd.grid(row=3, column=0, padx=10, pady=5, sticky="e")
entry_low_usd = ttk.Entry(window)
entry_low_usd.grid(row=3, column=1, padx=10, pady=5)

# Create EUR input fields
label_eur = ttk.Label(window, text="Enter rates for EUR:")
label_eur.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

label_open_eur = ttk.Label(window, text="Open rate:")
label_open_eur.grid(row=5, column=0, padx=10, pady=5, sticky="e")
entry_open_eur = ttk.Entry(window)
entry_open_eur.grid(row=5, column=1, padx=10, pady=5)

label_high_eur = ttk.Label(window, text="High rate:")
label_high_eur.grid(row=6, column=0, padx=10, pady=5, sticky="e")
entry_high_eur = ttk.Entry(window)
entry_high_eur.grid(row=6, column=1, padx=10, pady=5)

label_low_eur = ttk.Label(window, text="Low rate:")
label_low_eur.grid(row=7, column=0, padx=10, pady=5, sticky="e")
entry_low_eur = ttk.Entry(window)
entry_low_eur.grid(row=7, column=1, padx=10, pady=5)

# Create button to make predictions
button_predict = ttk.Button(window, text="Predict", command=predict_rates)
button_predict.grid(row=8, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Create result labels
label_result_usd = ttk.Label(window, text="")
label_result_usd.grid(row=9, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

label_result_eur = ttk.Label(window, text="")
label_result_eur.grid(row=10, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

# Configure grid weights to center content
for i in range(11):
    window.grid_rowconfigure(i, weight=1)
    window.grid_columnconfigure(0, weight=1)
    window.grid_columnconfigure(1, weight=1)

# Run the main event loop
window.mainloop()


# In[ ]:




