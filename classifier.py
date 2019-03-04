# © Anthony Krivonos 2019
# NBA ML
# classifier.py
# March 3rd, 2019

import csv
import numpy as np

##
#
#   Define Binary Classification Model
#
##

DEFAULT_TOLERANCE = 0.1

class BinaryModel():

    def __init__(self, featureCount):
        self.ε = DEFAULT_TOLERANCE
        self.w = np.zeros(featureCount + 1)
        self.λ = np.zeros(0)

    # Learn w weights, b intercept, and lambdas regularization parameters from data.
    def train(self, data, iter):

        # Update lambda
        if (len(self.λ) != len(data)):
            self.λ = np.zeros(len(data))

        # Define gradient descent w updater
        dw = lambda xij, wj, λi, yi: (2 * wj) - ((λi if yi == 1 else 0) * xij)

        # Define gradient ascent λ updater
        dλ = lambda w, xi, yi: 1 - np.dot(xi[:-1], w[:-1]) - w[-1] if yi == 1 else 1

        # Re-learn iter times
        for _ in range(iter):
            # Loop through data points
            for i in range(len(data)):
                # Get data row from list of data
                data_row = data[i]
                # List of features
                xi = data_row[:-1]
                # Append learning value of 1 to xi
                xi = np.append(xi, 1)
                # Classification label
                yi = data_row[-1]
                # Perform repeated gradient ascent for each feature
                for j in range(len(data_row)):
                    self.w[j] -= self.ε * dw(xi[j], self.w[j], self.λ[i], yi)
                # Perform gradient ascent to learn lambda
                self.λ[i] += self.ε * dλ(self.w, xi, yi)
                # Prevent negative λ
                self.λ[i] = self.λ[i] if self.λ[i] >= 0 else 0

    # Classify a vector of features
    def classify(self, features):
        # Record result
        result = np.dot(features, self.w[:-1]) + self.w[-1]
        # Return classification
        classification = 0 if result < DEFAULT_TOLERANCE else 1
        return classification

##
#
#   Train and Classify Data
#
##

# Number of times to iterate in training procedure
NUM_ITERATIONS = 1000

# Function to create a data row from parameters
# ->  Features:
# x0: Points (PTS)
# x1: Field Goal Percentage: (FG_PCT)
# x2: Three-Point Percentage: (FG3_PCT)
# x3: Free Throw Percentage: (FT_PCT)
# x4: Rebounds (REB)
# x5: Assists (AST)
# x6: Steals (STL)
# x7: Blocks (BLK)
# x8: Turnovers (TOV)
def create_data(pts, fg_pct, fg3_pct, ft_pct, reb, ast, stl, blk, tov):
    return [ pts, fg_pct, fg3_pct, ft_pct, reb, ast, stl, blk, tov ]

with open('data/trainingdata.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    # Store data rows
    rowCount = 0
    data = []
    for row in spamreader:
        if (rowCount > 0):
            row = [float(num) for num in row]
            row[-1] = int(row[-1])
            data.append(row)
        rowCount += 1
    
    # Prevent empty data
    if rowCount == 0:
        exit

    # Create model
    model = BinaryModel(len(data[0]) - 1)

    # Train model
    model.train(data, NUM_ITERATIONS)

    # Godly team => should win
    godly_team = create_data(
        pts= 132,
        fg_pct=0.8,
        fg3_pct=0.65,
        ft_pct=0.98,
        reb=82,
        ast=44,
        stl=25,
        blk=38,
        tov=3
    )

    # Decent team => could win
    decent_team = create_data(
        pts= 80,
        fg_pct=0.54,
        fg3_pct=0.43,
        ft_pct=0.81,
        reb=42,
        ast=34,
        stl=13,
        blk=19,
        tov=6
    )

    # Bad team => won't win
    bad_team = create_data(
        pts= 30,
        fg_pct=0.14,
        fg3_pct=0.03,
        ft_pct=0.09,
        reb=3,
        ast=1,
        stl=4,
        blk=2,
        tov=30
    )
    
    # Display if team won
    did_win = model.classify(godly_team) == 1
    # Print results
    print('Godly team ' + ('won!' if did_win == 1 else 'lost...'))
    # Display if team won
    did_win = model.classify(decent_team) == 1
    # Print results
    print('Decent team ' + ('won!' if did_win == 1 else 'lost...'))
    # Display if team won
    did_win = model.classify(bad_team) == 1
    # Print results
    print('Bad team ' + ('won!' if did_win == 1 else 'lost...'))
