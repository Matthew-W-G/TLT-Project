import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import func_library



bond_yields = pd.read_csv("/Users/Matthew/Downloads/30-year-treasury-bond-rate-yield-chart.csv", header=8)
bond_yields.columns = ['Date', 'Bond Yield']
bond_futures = pd.read_csv("/Users/Matthew/Downloads/30-year-treasury-futures.csv", header=10)
bond_futures.columns = ['Date', 'Bond Price']
TLT_price = pd.read_csv("/Users/Matthew/Downloads/TLT-2.csv")
FED_Funds = pd.read_csv("/Users/Matthew/Downloads/FEDFUNDS.csv")
FED_Funds.columns = ['Date', 'Fed Rate']
FED_Funds = FED_Funds.fillna(method='ffill')
inflation = pd.read_csv("/Users/Matthew/Downloads/Inflation Data - CPIAUCSL.csv")




bond_comparisons = pd.merge(pd.merge(
    pd.merge(pd.merge(bond_yields, bond_futures, how="outer", on="Date"),
             TLT_price[['Date','Close']].copy(),how="outer",on="Date"),
                    FED_Funds,how="outer",on="Date"),inflation,how="outer", on="Date")


bond_comparisons = bond_comparisons.dropna(subset=['Bond Yield','Bond Price'])
bond_comparisons['Fed Rate'].fillna(method='ffill', inplace=True)
bond_comparisons['Inflation Rate'].fillna(method='ffill', inplace=True)
bond_comparisons['Fed Rate'].fillna(method='bfill', inplace=True)





fill_missing = bond_comparisons[['Bond Price', 'Close']]
fill_missing.dropna(inplace=True)
print(fill_missing)
X1 = fill_missing[['Bond Price']]
Y1 = fill_missing[['Close']]
missing_regr = linear_model.LinearRegression()
missing_regr.fit(X1, Y1)

print('Intercept: ', missing_regr.intercept_)
print('Coefficients: ', missing_regr.coef_)
print('Prediction', missing_regr.predict([[160]]))


def regress_price(bond_price):
    return bond_price*missing_regr.coef_+missing_regr.intercept_

bond_comparisons['Close'].fillna(bond_comparisons['Bond Price'].apply(regress_price), inplace=True)

print(bond_comparisons)

def regression():
    X = bond_comparisons[['Bond Yield']]
    Y = bond_comparisons[['Close']]

    regr = linear_model.LinearRegression()
    regr.fit(X,Y)

    print('Intercept: ', regr.intercept_)
    print('Coefficients: ', regr.coef_)
    print('Prediction', regr.predict([[5.43239032]]))

regression()


def plot_creator():
    compare_FED_to_yield = pd.merge(bond_yields, FED_Funds, how="outer", on="Date")
    compare_FED_to_yield.plot.scatter(x='Bond Yield', y='Fed Rate')
    bond_comparisons.plot.line(x='Date')
    bond_comparisons.plot.scatter(x='Bond Yield', y='Bond Price')
    bond_comparisons.plot.scatter(x='Bond Yield', y='Close')

plot_creator()

bond_comparisons.plot.line(x='Date')


plt.show()


