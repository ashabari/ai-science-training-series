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
        # Define the model, loss, and update functions
        def model(x, m, b):
            return m * x + b

        def loss(x, y, m, b):
            y_predicted = model(x, m, b)
            return np.power(y - y_predicted, 2)

        def updated_m(x, y, m, b, learning_rate):
            dL_dm = -2 * x * (y - model(x, m, b))
            dL_dm = np.mean(dL_dm)
            return m - learning_rate * dL_dm

        def updated_b(x, y, m, b, learning_rate):
            dL_db = -2 * (y - model(x, m, b))
            dL_db = np.mean(dL_db)
            return b - learning_rate * dL_db

        # Set initial slope and intercept
        m = 5.
        b = 1000.

        # Set learning rates for each parameter
        learning_rate_m = 1e-7
        learning_rate_b = 1e-1

        # Batch size
        batch_size = 512

        # Adjust loop_N to ensure the entire dataset is processed multiple times
        loop_N = 30 * len(data) // batch_size

        loss_history = []
        slopes = []

        # Open the file to save the output
        with open('sgd_test.txt', 'w') as file:
            for i in range(loop_N):
                # Sample a batch from the dataset
                data_batch = data.sample(batch_size)
                data_x = data_batch['Gr Liv Area'].to_numpy()
                data_y = data_batch['SalePrice'].to_numpy()

                # Update our slope and intercept based on the current values
                m = updated_m(data_x, data_y, m, b, learning_rate_m)
                b = updated_b(data_x, data_y, m, b, learning_rate_b)

                # Calculate the loss value
                loss_value = np.mean(loss(data_x, data_y, m, b))

                # Keep a history of our loss values and slopes
                loss_history.append(loss_value)
                slopes.append((m, b))

                # Save the output to the file
                file.write('[%03d]  y_i = %.2f * x + %.2f    loss: %f\n' % (i, m, b, loss_value))

        # Plot all the slopes from each iteration in a single plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        for i, (m, b) in enumerate(slopes):
            line_x = np.linspace(data['Gr Liv Area'].min(), data['Gr Liv Area'].max(), 100)
            line_y = line_x * m + b
            plt.plot(line_x, line_y)
            plt.text(line_x[-1], line_y[-1], str(i), fontsize=8, color='red')
        plt.scatter(data['Gr Liv Area'], data['SalePrice'], color='blue', label='Data')
        plt.xlabel('Ground Living Area (square feet)')
        plt.ylabel('Sale Price ($)')
        plt.title('Sale Price vs Ground Living Area with Slopes from Each Iteration')

        # Plot all the loss values in a single plot at the end
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(loop_N), loss_history, 'o-')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Values over Iterations')
        plt.tight_layout()
        plt.show()
else:
    print("AmesHousing.csv does not exist.")

