#include <iostream>
#include <random>
#include <cmath>
#include <cstring>
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(const py::tuple& shape, const py::dict& config) {
	this->layers = py::len(shape);
	this->shape = new unsigned[layers];
	this->activations = new VectorXd[layers];
	this->weights = new MatrixXd[layers - 1];
	this->weighted_sums = new VectorXd[layers - 1];
	this->biases = new VectorXd[layers - 1];
	this->deltas = new VectorXd[layers - 1];
	this->actfuncs = new function_[layers - 1];
	this->actfunc_ders = new function_[layers - 1];
	this->func_params = new double[layers - 1];

	for (unsigned l = 0; l < layers; l++) {
		this->shape[l] = py::int_(shape[l]);
		activations[l] = VectorXd::Zero(this->shape[l]);
	}

	for (unsigned l = 0; l < layers - 1; l++) {
		weights[l] = MatrixXd::Random(this->shape[l + 1], this->shape[l]);
		biases[l] = VectorXd::Zero(this->shape[l + 1]);
	}

	const py::dict function_names = config[py::cast("functions")];
	if (py::len(function_names) != layers - 1) throw std::runtime_error("Network depth does not correspond to the function list");
	std::string function_name;
	int i = 0;
	for (auto item : function_names) {
		function_name = py::str(item.second);
		actfuncs[i] = get_function_by_name(function_name);
		actfunc_ders[i] = get_function_der_by_name(function_name);
		(i < layers - 1) && i++;
	}

	const py::dict function_parameters = config[py::cast("parameters")];
	if (py::len(function_parameters) != layers - 1) throw std::runtime_error("Network depth does not correspond to the parameter list");
	double parameter;
	i = 0;
	for (auto item : function_parameters) {
		parameter = py::float_(py::str(item.second));
		func_params[i] = parameter;
		(i < layers - 1) && i++;
	}
}


void NeuralNetwork::inspect() const {
	using namespace std;
	cout << "=============NeuralNetwork===============\n";
	unsigned l;

	cout << "\nShape:";
	for (l = 0; l < layers; l++) cout << " " << shape[l];

	cout << "\n\nWeights:\n";
	cout << "----------------------------------\n";
	for (l = 0; l < layers - 1; l++) {
		cout << weights[l];
		cout << "\n----------------------------------\n";
	}

	cout << "\nActivations:\n";
	cout << "----------------------------------\n";
	for (l = 0; l < layers; l++) {
		cout << activations[l].transpose() << endl;
	}
	cout << "----------------------------------" << endl;
}


const VectorXd& NeuralNetwork::forwardprop(const VectorXd& X) {
	// activations[0](0, X.size()) = X(0, shape[0]);
	assert(X.size() == activations[0].size());

	activations[0] = X;
	for (unsigned l = 0; l < layers - 1; l++) {
		// Eigen matrix multiplication requires optimization
		weighted_sums[l] = weights[l] * activations[l] + biases[l];
		activations[l + 1] = actfuncs[l](weighted_sums[l], func_params[l]);
	}
	return activations[layers - 1];
}


void NeuralNetwork::backprop(const VectorXd& Y, const double lr) {
	deltas[layers - 2] = (Y - activations[layers - 1]).cwiseProduct(actfunc_ders[layers - 2](weighted_sums[layers - 2], func_params[layers - 2]));
	for (size_t l = layers - 2; l > 0; l--) {
		deltas[l - 1] = (weights[l].transpose() * deltas[l]).cwiseProduct(actfunc_ders[l - 1](weighted_sums[l - 1], func_params[l - 1]));
	}
	for (unsigned l = 0; l < layers - 1; l++) {
		for (unsigned j = 0; j < shape[l + 1]; j++)
			weights[l].row(j) += lr * activations[l] * deltas[l](j);
		biases[l] += lr * deltas[l];
	}
}


void NeuralNetwork::train(const MatrixXd& input, const MatrixXd& target, const double lr, const unsigned epochs, const unsigned test_freq) {
	float accuracy = 0, average_accuracy = 0, best_accuracy = 0;
	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_int_distribution<unsigned> uni(0, input.rows() - 1);

	unsigned tests = 0;
	for (unsigned epoch = 0; epoch < epochs; epoch++) {
		if (!(epoch % test_freq)) {
			accuracy = test(input, target);
			average_accuracy = (average_accuracy * tests + accuracy) / (tests + 1);
			average_accuracy = std::round(average_accuracy * 100) / 100;
			(accuracy > best_accuracy) && (best_accuracy = accuracy);
			std::cout << "\rEpoch " << epoch << " | "
			<< "Accuracy: " << accuracy << "% | "
			<< "Average accuracy: " << average_accuracy << "% | "
			<< "Best accuracy: " << best_accuracy << "%\t";
			tests++;
		}
		unsigned i = uni(rng);
		VectorXd X = input.row(i);
		VectorXd Y = target.row(i);
		VectorXd rv = forwardprop(X);
		backprop(Y, lr);
	}
}


const float NeuralNetwork::test(const MatrixXd& input, const MatrixXd& target) {
	float correct_predictions = 0;
	float accuracy;
	VectorXd network_output;
	VectorXd::Index real_max_index, expected_max_index;

	for (unsigned i = 0; i < input.rows(); i++) {
		network_output = forwardprop(input.row(i));
		network_output.maxCoeff(&real_max_index);
		((VectorXd)(target.row(i))).maxCoeff(&expected_max_index);
		(real_max_index == expected_max_index) && correct_predictions++;
	}

	accuracy = correct_predictions / input.rows() * 100;
	accuracy = std::round(accuracy * 100) / 100;
	return accuracy;
}


NeuralNetwork::~NeuralNetwork() {
	delete[] shape;
	delete[] activations;
	delete[] weights;
	delete[] weighted_sums;
	delete[] biases;
	delete[] deltas;
	delete[] actfuncs;
	delete[] actfunc_ders;
	delete[] func_params;
}


const VectorXd sigmoid(const VectorXd& X, double width) {
	VectorXd result(X.size());
	for (unsigned i = 0; i < result.size(); i++)
		result(i) = 1 / (1 + exp(-X(i) * width));
	return result;
}

const VectorXd sigmoid_der(const VectorXd& X, double width) {
	VectorXd result(X.size());
	result = sigmoid(X, width).cwiseProduct((-sigmoid(X, width).array() + 1).matrix());
	return result;
}

const VectorXd ReLU(const VectorXd& X, double angle) {
	VectorXd result(X.size());
	for (unsigned i = 0; i < result.size(); i++)
		result(i) = fmax(0, X(i) * angle);
	return result;
}

const VectorXd ReLU_der(const VectorXd& X, double angle) {
	VectorXd result(X.size());
	for (unsigned i = 0; i < result.size(); i++)
		result(i) = 0 ? X(i) < 0 : angle;
	return result;
}