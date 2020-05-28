 #include <iostream>
#include <fstream>
using namespace std;

void fit(double* x_actual, double* weights, double* y_predicted, int size)
{
	for (int i = 0; i < size; i++)
	{
		y_predicted[i] = weights[1] * x_actual[i] + weights[0];
	}
}

void update_weights(double* weights, double* cost_gradient, double learning_rate)
{
	weights[0] -= learning_rate * cost_gradient[0];
	weights[1] -= learning_rate * cost_gradient[1];
}

void calculate_cost(double* y_predicted, double* x_actual, double* y_actual, double* cost_gradient, int size)
{
	double gradient_intercept = 0;
	double gradient_slope = 0;

	for (int i = 0; i < size; i++)
	{
		// partial derivative of cost function with respect to y-intercept (dC/dB)
		gradient_intercept += (y_predicted[i] - y_actual[i]);
		// partial derivative of cost function with respect to slope (dC/dM)
		gradient_slope += ((y_predicted[i] - y_actual[i]) * x_actual[i]);
	}

	gradient_intercept *= (2.0 / size);
	gradient_slope *= (2.0 / size);

	cost_gradient[0] = gradient_intercept;
	cost_gradient[1] = gradient_slope;
}

double* linear_regression(double* x_actual, double* y_actual, int size)
{
	double learning_rate = 0.0001;
	double threshold = 0.0001;
	int iteration = 0;

	double* y_predicted = new double[size];
	double* cost_gradient = new double[2];
	double* weights = new double[2];

	weights[0] = 0; // will be the predicted y-intercept, however its not considered a weight, but rather a "bias"
	weights[1] = 1; // will be the predicted slope

	do
	{
		// predict y-values based off of the weights and given x-values
		fit(x_actual, weights, y_predicted, size);

		// calculate the cost gradient based off of the predicted and actual y-values
		calculate_cost(y_predicted, x_actual, y_actual, cost_gradient, size);

		// update the weights with the cost gradient
		update_weights(weights, cost_gradient, learning_rate);

		iteration++;
		if (iteration % 100000 == 0)
		{
			cout << "Iteration: " << iteration << " dC/dB: " << cost_gradient[0] << " dC/dM: " << cost_gradient[1] << endl;
		}
	}
	while (abs(cost_gradient[0]) > threshold || abs(cost_gradient[1]) > threshold);

	cout << "Iteration: " << iteration << " dC/dB: " << cost_gradient[0] << " dC/dM: " << cost_gradient[1] << endl;

	return weights;
}


int main()
{
	ifstream file("training_data.txt");
	int size;
	file >> size;

	double* x = new double[size];
	double* y = new double[size];
	

	for (int i = 0; i < size; i++)
	{
		file >> x[i];
		file >> y[i];
	}

	double* weights = new double[2];
	weights = linear_regression(x, y, size);

	cout << "y = (" << weights[1] << ")x + (" << weights[0] << ")" << endl;
	


	/*
	int size = 11;
	double* x = new double[size] { 71, 73, 64, 65, 61, 70, 65, 72, 63, 67, 64 };
	double* y = new double[size] { 160, 183, 154, 168, 159, 180, 145, 210, 132, 168, 141 };
	double* weights = new double[2];
	weights = linear_regression(x, y, size);

	cout << "y = (" << weights[1] << ")x + (" << weights[0] << ")" << endl;
	*/
}