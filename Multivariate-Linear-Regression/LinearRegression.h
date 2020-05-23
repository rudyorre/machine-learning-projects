#pragma once
#include <fstream>
using namespace std;

class Model
{
private:
	int num_dimensions;
	int size;
	double** inputs;
	double* outputs;
	double* features;
	double learning_rate;
	double threshold;

public:
	Model();
	Model(ifstream& training_data);
	void find_dimensions(ifstream& training_data);

	// linear regression and its utility functions
	double* linear_regression();
	double* linear_regression(bool will_print);
	double* fit();
	double calculate_cost(const double* predicted_outputs, double* cost_gradient);
	void update_features(const double* cost_gradient);
	bool above_threshold(const double* cost_gradient);

	// printers
	void print_features();
	void print_data();
	void print_iteration(int iteration, double cost, double* cost_gradient);

	// getters and setters
	double predict(double* values);
	double get_threshold();
	void set_threshold(double set_threshold);
};