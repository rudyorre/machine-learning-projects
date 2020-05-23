#include <iostream>
#include <fstream>
#include "LinearRegression.h"


int main()
{
	ifstream training_data("training_data_5.txt");

	Model model(training_data);

	model.linear_regression(true);
	model.print_features();

	double* values = new double[4]{ 125,	925,	10.19999981,	91 };

	cout << "Predicted Death Rate: " << model.predict(values) << endl;
}