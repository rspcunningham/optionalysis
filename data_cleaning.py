import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

historic_df = pd.read_csv("data/TSLA_historic.csv")
options_df = pd.read_csv("data/TSLA_options.csv")

historic_df = historic_df.drop(columns=["Open", "High", "Low", "Close", "Volume"])
options_df = options_df.drop(
    columns=[
        "lastTradeDate",
        "inTheMoney",
        "contractSize",
        "currency",
        "volume",
        "openInterest",
        "impliedVolatility",
        "change",
        "percentChange",
        "contractSymbol",
        "Unnamed: 0",
        "lastPrice",
    ]
)
print(options_df)


def get_mid_price(contract) -> float:
    return (contract["bid"] + contract["ask"]) / 2


options_df["price"] = options_df.apply(get_mid_price, axis=1)
options_df = options_df.drop(columns=["bid", "ask"])
# print(options_df)

calls = options_df[options_df["type"] == 0].drop(columns="type")

calls_dict = {}

for date in calls["expirationDate"].unique():
    filtered = calls[calls["expirationDate"] == date].drop(columns="expirationDate")
    calls_dict[date] = filtered.reset_index(drop=True)

test_df = calls_dict["2023-07-21"]
new_df = pd.DataFrame(
    {
        "strike": np.arange(test_df["strike"].min(), test_df["strike"].max() + 1, 1),
        "price": np.nan,
    }
)
merged_df = pd.merge(new_df, test_df, on="strike", how="left").drop(columns="price_x")
test_df = merged_df.interpolate(method="cubicspline", order=5).rename(
    columns={"price_y": "price"}
)
print(test_df.loc[0:20])


avg_window = 20

test_df["price"] = test_df["price"].rolling(window=avg_window, center=True).mean()
test_df["dC/dK"] = test_df["price"].diff() / test_df["strike"].diff()
test_df["dC/dK"] = test_df["dC/dK"].rolling(window=avg_window, center=True).mean()
test_df["ddC/dKK"] = test_df["dC/dK"].diff() / test_df["strike"].diff()
test_df["ddC/dKK"] = test_df["ddC/dKK"].rolling(window=avg_window, center=True).mean()
test_df["dC/dK"] = test_df["dC/dK"].clip(lower=-1, upper=0)
test_df["ddC/dKK"] = test_df["ddC/dKK"].clip(lower=0)
# print(test_df)

plt.subplot(1, 3, 1)
plt.scatter(test_df["strike"], test_df["price"])
plt.xlabel("Strike Price")
plt.ylabel("Spot Price")
plt.title("Price")

plt.subplot(1, 3, 2)
plt.scatter(test_df["strike"], test_df["dC/dK"])
plt.xlabel("Strike Price")
plt.ylabel("Cumulative Density Function")
plt.title("CDF")

plt.subplot(1, 3, 3)
plt.scatter(test_df["strike"], test_df["ddC/dKK"])
plt.xlabel("Strike Price")
plt.ylabel("Probability Density Function")
plt.title("PDF")

plt.tight_layout()
plt.show()
