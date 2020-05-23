#include <iostream>
#include <iomanip>
#include <string>
#include "LinearRegression.h"

Model::Model()
{
	num_dimensions = 0;
	size = 0;
	inputs = new double* [0];
	outputs = new double[0];
	features = new double[0];
	learning_rate = 0.0001;
	threshold = 0.0001;
}

Model::Model(ifstream& training_data)
{
	learning_rate = 0.000001;
	threshold = 0.2;

	find_dimensions(training_data);

	inputs = new double* [num_dimensions];
	outputs = new double[size];
	features = new double[num_dimensions];

	for (int i = 0; i < num_dimensions; i++)
	{
		inputs[i] = new double[size];
		features[i] = 0;
	}
	features[num_dimensions - 1] = 0;

	for (int i = 0; i < size; i++)
	{
		for (int n = 0; n < num_dimensions; n++)
		{
			if (n == 0)
			{
				inputs[n][i] = 1;
			}
			else
			{
				training_data >> inputs[n][i];
			}
		}

		training_data >> outputs[i];
	}
}

void Model::find_dimensions(ifstream& training_data)
{
	int count = 0;
	string line;

	while (!training_data.eof())
	{
		getline(training_data, line);
		count++;
	}
	int dimensions = 0;
	int pos = 0, prev = 0;
	do
	{
		pos = line.find("	", prev);
		if (pos == string::npos) pos = line.length();
		if (!line.substr(prev, pos - prev).empty()) dimensions++;
		prev = pos + 1;
	}
	while (pos < line.length() && prev < line.length());

	num_dimensions = dimensions;
	size = count;

	training_data.clear();
	training_data.seekg(0, ios::beg);
}

/* main linear regression function */

double* Model::linear_regression()
{
	return linear_regression(false);
}

double* Model::linear_regression(bool will_print)
{
	double* predicted_outputs = new double[size];
	double* cost_gradient = new double[num_dimensions];
	int iteration = 0;

	do
	{
		predicted_outputs = fit();
		double cost = calculate_cost(predicted_outputs, cost_gradient);
		update_features(cost_gradient);

		if (will_print && iteration % 1000 == 0)
		{
			//cout << "Iterations: " << iteration << " Cost: " << cost << " Gradient: <"
			//	<< cost_gradient[0] << ", " << cost_gradient[1] << ", " << cost_gradient[2] << ">" << endl;
			print_iteration(iteration, cost, cost_gradient);
		}

		iteration++;
	} while (above_threshold(cost_gradient));

	return features;
}

/* utility functions */

double* Model::fit()
{
	double* predicted_outputs = new double[size];

	for (int i = 0; i < size; i++)
	{
		predicted_outputs[i] = 0;
		for (int n = 0; n < num_dimensions; n++)
		{
			predicted_outputs[i] += features[n] * inputs[n][i];
		}
	}

	return predicted_outputs;
}

double Model::calculate_cost(const double* predicted_outputs, double* cost_gradient)
{
	double cost = 0;

	for (int n = 0; n < num_dimensions; n++)
	{
		cost_gradient[n] = 0;
	}

	for (int i = 0; i < size; i++)
	{
		double difference = predicted_outputs[i] - outputs[i];
		cost += difference * difference;

		for (int n = 0; n < num_dimensions; n++)
		{
			cost_gradient[n] += difference * inputs[n][i] / size;
		}
	}

	return cost / (2.0 * size);
}

void Model::update_features(const double* cost_gradient)
{
	for (int n = 0; n < num_dimensions; n++)
	{
		features[n] -= learning_rate * cost_gradient[n];
	}
}

bool Model::above_threshold(const double* cost_gradient)
{
	for (int n = 0; n < num_dimensions; n++)
	{
		if (abs(cost_gradient[n]) > threshold)
		{
			return true;
		}
	}
	return false;
}



/* printing functions */

void Model::print_features()
{
	cout << "y = " << fixed;
	for (int n = 0; n < num_dimensions; n++)
	{
		if (n == 0)
		{
			cout << features[n];
		}
		else
		{
			cout << " + (" << features[n] << ")x" << n;
		}
	}
	cout << endl;
}

void Model::print_data()
{
	for (int i = 0; i < size; i++)
	{
		for (int n = 0; n < num_dimensions; n++)
		{
			cout << inputs[n][i] << " ";
		}
		cout << outputs[i] << endl;
	}
}

void Model::print_iteration(int iteration, double cost, double* cost_gradient)
{
	cout << "Iteration: " << iteration << " Cost: " << cost << " Gradient:<";

	for (int n = 0; n < num_dimensions; n++)
	{
		cout << scientific << setprecision(3) << cost_gradient[n];

		if (n != num_dimensions - 1) cout << ", ";
	}

	cout << ">" << endl;
}

/* getters and setters */

double Model::predict(double* values)
{
	double prediction = 0;
	for (int n = 1; n < num_dimensions; n++)
	{
		prediction += values[n - 1] * features[n];
	}
	return prediction;
}

double Model::get_threshold()
{
	return threshold;
}

void Model::set_threshold(double set_threshold)
{
	threshold = set_threshold;
}


