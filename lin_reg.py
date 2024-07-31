import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Check if the file exists
if os.path.exists('AmesHousing.csv'):
    # Load the dataset
    data = pd.read_csv('AmesHousing.csv')

    # Ensure the column names are correct
    if 'Gr Liv Area' in data.columns and 'SalePrice' in data.columns:
        # Calculate fit values
        n = len(data)
        x = data['Gr Liv Area'].to_numpy()
        y = data['SalePrice'].to_numpy()
        sum_xy = np.sum(x * y)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_x2 = np.sum(x * x)

        denominator = n * sum_x2 - sum_x * sum_x
        m = (n * sum_xy - sum_x * sum_y) / denominator
        b = (sum_y * sum_x2 - sum_x * sum_xy) / denominator
        print('y = %f * x + %f' % (m, b))

        # Define the plotting function
        def plot_data(x, y, m, b, plt=plt):
            plt.figure()
            # Plot data points with 'bo' = blue circles
            plt.plot(x, y, 'bo', label='Data')
            # Create the line based on our linear fit
            linear_x = np.linspace(x.min(), x.max(), 100)
            linear_y = linear_x * m + b
            # Plot the linear points using 'r-' = red line
            plt.plot(linear_x, linear_y, 'r-', label='Fit: y = %f * x + %f' % (m, b))
            plt.xlabel('Ground Living Area (square feet)')
            plt.ylabel('Sale Price ($)')
            plt.title('Sale Price vs Ground Living Area with Linear Fit')
            plt.legend()
            plt.show()

        # Plot the data with the fit
        plot_data(x, y, m, b)
    else:
        print("Columns 'Gr Liv Area' or 'SalePrice' not found in the dataset.")
else:
    print("AmesHousing.csv does not exist.")

