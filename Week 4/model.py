from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load the data
housing = fetch_california_housing()
X, y = housing.data, housing.target


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

#%%
# Save the model
with open('linear_model_.pkl', 'wb') as f:
    pickle.dump(model, f)