#include <iostream>
#include <random>
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(const py::dict& config) {
	const py::list shape = config["shape_modifiers"];
	this->layers = py::len(shape);
	this->allocate_memory();

	for (unsigned l = 0; l < layers; l++) {
		this->shape[l] = py::int_(shape[l]);
		activations[l] = VectorXd::Zero(this->shape[l]);
	}

	for (unsigned l = 0; l < layers - 1; l++) {
		weights[l] = MatrixXd::Random(this->shape[l + 1], this->shape[l]);
		biases[l] = VectorXd::Zero(this->shape[l + 1]);
	}	

	function_names = config["activation_functions"];
	this->set_activation_functions();

	const py::list function_parameters = config["activation_function_parameters"];
	if (py::len(function_parameters) != layers - 1)
		throw std::runtime_error("Network depth does not correspond to the parameter list");
	double parameter;
	for (int i = 0; i < layers - 1; i++) {
		parameter = py::float_(function_parameters[i]);
		func_params[i] = parameter;
	}
}


NeuralNetwork::NeuralNetwork(const std::string& dumpfile) {
	this->load(dumpfile);
	this->set_activation_functions();
}


NeuralNetwork::NeuralNetwork(const NeuralNetwork* src) {
	this->layers = src->layers;
	allocate_memory();
	copy_state(src);
}


void NeuralNetwork::allocate_memory() {
	this->shape = new unsigned[layers];
	this->activations = new VectorXd[layers];
	this->weights = new MatrixXd[layers - 1];
	this->weighted_sums = new VectorXd[layers - 1];
	this->biases = new VectorXd[layers - 1];
	this->deltas = new VectorXd[layers - 1];
	this->actfuncs = new function_[layers - 1];
	this->actfunc_ders = new function_[layers - 1];
	this->func_params = new double[layers - 1];
}


void NeuralNetwork::set_activation_functions() {
	if (py::len(function_names) != layers - 1)
		throw std::runtime_error("Network depth does not correspond to the function list");
	std::string function_name;
	for (int i = 0; i < layers - 1; i++) {
		function_name = py::str(function_names[i]);
		actfuncs[i] = get_function_by_name(function_name);
		actfunc_ders[i] = get_function_der_by_name(function_name);
		if (!actfuncs[i] || !actfunc_ders[i]) throw std::runtime_error("Invalid function name has been passed");
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
		biases[l] += deltas[l];
	}
}


void NeuralNetwork::init_train(const MatrixXd *input, const MatrixXd *target, const py::dict& config) {
	this->input = input;
	this->target = target;
	this->epochs = py::int_(py::float_(config["epochs"]));
	this->test_freq = py::int_(py::float_(config["test_frequency"]));
	this->lr = py::float_(config["rate"]);
	rng = Rnd<Index>(0, input->rows() - 1);

	this->train_in_parallel_with_averaging(8);
}


void NeuralNetwork::init_train(const MatrixXd *input, const MatrixXd *target, const unsigned epochs, const double lr) {
	this->input = input;
	this->target = target;
	this->epochs = epochs;
	this->test_freq = epochs;
	this->lr = lr;
	rng = Rnd<Index>(0, input->rows() - 1);
}


void NeuralNetwork::train(void) {	
	for (unsigned epoch = 0; epoch < epochs; epoch++) {
		// if (!(epoch % test_freq)) run_test(epoch);
		i = rng();
		X = input->row(i);
		Y = target->row(i);
		forwardprop(X);
		backprop(Y, lr);
	}
}


const float NeuralNetwork::test(const MatrixXd *input, const MatrixXd *target) {
	Index expected_predictions = 0, correct_predictions = 0, ones;
	VectorXd network_output;
	VectorXi real_output_indices, expected_output_indices;
	float accuracy;

	for (Index i = 0; i < input->rows(); i++) {
		network_output = forwardprop(input->row(i));
		ones = target->row(i).count();
		expected_predictions += ones;
		real_output_indices = argsort(network_output)(seqN(0, ones));
		expected_output_indices = argsort(target->row(i))(seqN(0, ones));
		if (ones) correct_predictions += intersect1d_len(real_output_indices, expected_output_indices);
	}

	accuracy = (float)correct_predictions / (float)expected_predictions * 100;
	accuracy = std::round(accuracy * 100) / 100;
	return accuracy;
}


void NeuralNetwork::run_test(const unsigned epoch) {
	static float accuracy, best_accuracy = 0;
	accuracy = test(input, target);
	(accuracy > best_accuracy) && (best_accuracy = accuracy);
	std::cout << "\rEpoch " << epoch << " | "
		<< "Accuracy: " << accuracy << "% | "
		<< "Best accuracy: " << best_accuracy << "%\t";
}


NeuralNetwork::~NeuralNetwork() {
	// this->dump("dump.bin");
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