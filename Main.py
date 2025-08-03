import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df= pd.read_csv("housing.csv")
X= df[['bedrooms','sqft']]
y=df['price']*83
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print("Predicted prices on test set (₹):",[f"₹{p:,.2f}"for p in predictions])
print("Actual prices on test set (₹):",[f"₹{a:,.2f}"for a in y_test.values])
print("\nModel Coefficients:",model.coef_)
print("Model Intercept:",model.intercept_)

plt.scatter(y_test,predictions,color='blue')
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.xlabel("Actual Prices (₹)")
plt.ylabel("Predicted Prices (₹)")
plt.title("Actual vs Predicted House Prices in INR")
plt.grid(True)
plt.show()

try:
 bed_input=input("Enter number of bedrooms: ").strip()
 sqft_input=input("Enter square footage: ").strip()
 if not bed_input or not sqft_input:
  raise ValueError("Empty input")
 bed=int(bed_input)
 sqft=int(sqft_input)
except ValueError:
 print("\nInvalid input detected. Using default test values.")
 bed=3
 sqft=1500

new_data=pd.DataFrame([[bed,sqft]],columns=['bedrooms','sqft'])
predicted_price_inr=model.predict(new_data)[0]
print(f"\nPredicted house price for {bed} bedrooms and {sqft} sqft: ₹{predicted_price_inr:,.2f}")
