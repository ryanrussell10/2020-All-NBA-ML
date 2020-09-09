import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from operator import itemgetter
import pickle


def main():

	past_data = pd.read_csv('past_nba_data.csv')
	current_data = pd.read_csv('2020_nba_data.csv')
	current_data_names = current_data.iloc[:, 0]

	feature_cols = ['G', 'Wins', 'Seed', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'WS', 'VORP', 'BPM', 'All-Star']
	y_col = ['All-NBA']

	x_past = past_data[feature_cols]
	y_past = past_data[y_col]

	x_current = current_data[feature_cols]

	x_train, x_test, y_train, y_test = train_test_split(x_past, y_past, test_size = 0.2)
	y_train = y_train.values.ravel()

	sum_probs = [0] * 26
	avg_probs = [0] * 26

	test_runs = 10

	for i in range(test_runs):

		print("--- Iteration " + str(i) + " ---")

		mlp = MLPClassifier(hidden_layer_sizes = (100, 100), max_iter = 2000, activation = 'relu')

		mlp.fit(x_train, y_train)
		y_pred = mlp.predict(x_test)

		accuracy = metrics.accuracy_score(y_test, y_pred)
		precision = metrics.precision_score(y_test, y_pred)

		print("Neural Network Accuracy: " + str(accuracy))
		print("Neural Network Precision: " + str(precision))
		print()

		probs = mlp.predict_proba(x_current)
		probs = probs[:, 1]

		for index, (sum_prob, prob) in enumerate(zip(sum_probs, probs)):
			sum_probs[index] = sum_prob + prob

	for index, sum_prob in enumerate(sum_probs):
		sum_probs[index] = sum_prob / test_runs

	avg_probs = sum_probs

	candidate_list = [[name, avg_prob] for name, avg_prob in zip(current_data_names, avg_probs)]
	candidate_list = sorted(candidate_list, key = itemgetter(1), reverse = True)

	for i in candidate_list:
		print(i)


	#filename = 'finalized_model.sav'
	#pickle.dump(mlp, open(filename, 'wb'))

main()