/*#include <iostream>

void fit(double *x_actual, double *weights, double *y_predicted, int size)
{
	for (int i = 0; i < size; i++)
	{
		y_predicted[i] = weights[1] * x_actual[i] + weights[0];
	}
}

void update_weights(double *weights, double *cost_gradient, double learning_rate)
{
	weights[0] -= learning_rate * cost_gradient[0];
	weights[1] -= learning_rate * cost_gradient[1];
}

void calculate_cost(double* x_actual, double* y_actual, double* weights, double* cost_gradient, int size)
{
	for (int i = 0; i < size; i++)
	{
		// partial derivative of cost function with respect to y-intercept (dC/dB)
		cost_gradient[0] += weights[0] + weights[1] * x_actual[i] - y_actual[i];

		// partial derivative of cost function with respect to slope (dC/dM)
		cost_gradient[1] += (weights[0] + weights[1] * x_actual[i] - y_actual[i]) * x_actual[i];
	}

	cost_gradient[0] *= (2 / size);
	cost_gradient[1] *= (2 / size);
}

double* linear_regression(double* x_actual, double* y_actual, int size)
{
	double learning_rate = 0.001;
	double threshold = 0.0001;

	double* y_predicted = new double[size];
	double* cost_gradient = new double[2];
	double* weights = new double[2];

	weights[0] = 0; // predicted y-intercept
	weights[1] = 0; // predicted slope

	do
	{
		calculate_cost(x_actual, y_actual, weights, cost_gradient, size);
		update_weights(weights, cost_gradient, learning_rate);
	}
	while (cost_gradient[0] > threshold || cost_gradient[1] > threshold);

	return weights;
}*/