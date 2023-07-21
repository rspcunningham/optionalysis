import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_mid_price(contract) -> float:
    return (contract["bid"] + contract["ask"]) / 2


def process_one_day_calls(chain: pd.DataFrame) -> pd.DataFrame:
    avg_window = 20

    temp = pd.DataFrame(
        {
            "strike": np.arange(
                chain["strike"].min(), chain["strike"].max(), resolution
            ),
            "price": np.nan,
        }
    )
    temp2 = pd.merge(temp, chain, on="strike", how="left").drop(columns="price_x")
    interp_chain = temp2.interpolate(method="cubicspline", order=5).rename(
        columns={"price_y": "price"}
    )

    interp_chain["price"] = (
        interp_chain["price"].rolling(window=avg_window, center=True).mean()
    )
    interp_chain["dC/dK"] = interp_chain["price"].diff() / interp_chain["strike"].diff()
    interp_chain["dC/dK"] = (
        interp_chain["dC/dK"].rolling(window=avg_window, center=True).mean()
    )
    interp_chain["ddC/dKK"] = (
        interp_chain["dC/dK"].diff() / interp_chain["strike"].diff()
    )
    interp_chain["ddC/dKK"] = (
        interp_chain["ddC/dKK"].rolling(window=avg_window, center=True).mean()
    )

    return interp_chain.dropna().reset_index(drop=True)


resolution = 1  # USD

options_df = pd.read_csv("data/TSLA_options.csv")
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

options_df["price"] = options_df.apply(get_mid_price, axis=1)
options_df = options_df.drop(columns=["bid", "ask"])
calls = options_df[options_df["type"] == 0].drop(columns="type")
calls = calls[calls["price"] != 0]

max_strike = calls["strike"].max()

master = pd.DataFrame()
master["strike"] = np.arange(0, max_strike + resolution, resolution)

for date in calls["expirationDate"].unique():
    this_date_calls = (
        calls[calls["expirationDate"] == date]
        .drop(columns="expirationDate")
        .reset_index(drop=True)
    )
    processed = process_one_day_calls(this_date_calls)
    master[date] = processed["ddC/dKK"]

master[master < 0] = np.nan
master.set_index("strike", inplace=True)
master_interpolated = master.interpolate(method="linear", order=1, axis=1)
# master = master.fillna(0)
"""
print(master_interpolated)

x_vals = master_interpolated.columns[1:]
y_vals = master_interpolated.iloc[280][1:]


plt.scatter(x_vals, y_vals)
plt.xlabel("Columns")
plt.ylabel("Values")
plt.title("Scatter Plot")
plt.show()
"""

low_cut_off = 150
high_cut_off = 300

temp = master_interpolated.loc[low_cut_off:high_cut_off].fillna(0)
print(temp)

"""# Create the figure and 3D axes
plt.imshow(
    temp,
    cmap="cividis",
    interpolation="gaussian",
    aspect="auto",
    extent=[0, len(temp.columns), len(temp.index), 0],
)

# Set custom tick labels for x and y axes
# plt.yticks(np.arange(len(temp.index), step=10), temp.index)
plt.yticks([])
# plt.yticks(np.arange(1, len(temp.index), step=10), np.arange(low_cut_off, high_cut_off, 10))

# Set labels and title
plt.xlabel("Weeks")
plt.ylabel("Price (USD)")
plt.title("$TSLA Price Prediction")

# Add a colorbar
plt.colorbar()

# Display the plot
plt.show()"""
