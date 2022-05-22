#include <iostream>
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(const py::dict& config) {
	const py::list shape = config["shape_modifiers"];
	layers = py::len(shape);
	allocate_memory();

	for (unsigned l = 0; l < layers; l++) {
		this->shape[l] = py::int_(shape[l]);
		activations[l] = VectorXd::Zero(this->shape[l]);
	}

	for (unsigned l = 0; l < layers - 1; l++) {
		weights[l] = MatrixXd::Random(this->shape[l + 1], this->shape[l]);
		biases[l] = VectorXd::Zero(this->shape[l + 1]);
	}	

	function_names = config["activation_functions"];
	set_activation_functions();

	const py::list function_parameters = config["activation_function_parameters"];
	if (py::len(function_parameters) != layers - 1)
		throw std::runtime_error("Network depth does not correspond to the parameter list");
	double parameter;
	for (int i = 0; i < layers - 1; i++) {
		parameter = py::float_(function_parameters[i]);
		func_params[i] = parameter;
	}
}


void NeuralNetwork::allocate_memory() {
	shape = new unsigned[layers];
	activations = new VectorXd[layers];
	weights = new MatrixXd[layers - 1];
	weighted_sums = new VectorXd[layers - 1];
	biases = new VectorXd[layers - 1];
	deltas = new VectorXd[layers - 1];
	actfuncs = new function_[layers - 1];
	actfunc_ders = new function_[layers - 1];
	func_params = new double[layers - 1];
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


void NeuralNetwork::inspect(void) const noexcept {
	using namespace std;
	cout << "=============NeuralNetwork===============\n";
	size_t l;

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
		cout << activations[l].transpose() << '\n';
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


void NeuralNetwork::backprop(const VectorXd& Y, const double lr) noexcept {
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


MatrixXd* NeuralNetwork::input = nullptr; MatrixXd* NeuralNetwork::target = nullptr;
double NeuralNetwork::lr = 0;
double NeuralNetwork::delta_lr = 1;
unsigned NeuralNetwork::accuracy_stuck_limit = UINT_MAX;
unsigned NeuralNetwork::delta_accuracy_stuck_limit = 1;
unsigned NeuralNetwork::total_epochs = 0;

void NeuralNetwork::init_train(MatrixXd *input, MatrixXd *target, const py::dict& config) {
	this->input = input;
	this->target = target;

	epochs = py::int_(py::float_(config["epochs"]));
	test_freq = py::int_(py::float_(config["test_frequency"]));
	lr = py::float_(config["rate"]);

	if (py::bool_(config["dynamic_rate"])){
		delta_lr = py::float_(config["rate_delta"]);
		accuracy_stuck_limit = py::int_(py::float_(config["accuracy_stuck_limit"]));
		delta_accuracy_stuck_limit = py::int_(config["accuracy_stuck_limit_delta"]);
	}

	if (!py::bool_(config["parallel_training"]))
		rng = Rnd<Index>(0, input->rows() - 1);
}


void NeuralNetwork::train(void) noexcept {	
	for (unsigned epoch = 1; epoch < epochs + 1; epoch++) {
		if (!(epoch % test_freq)) [[unlikely]] monitor(epoch);
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


void NeuralNetwork::monitor(const unsigned epoch) noexcept {
	static float accuracy, best_accuracy = 0;
	static unsigned accuracy_not_increased_for = 0;

	accuracy = test(input, target);

	if (accuracy > best_accuracy) {
		best_accuracy = accuracy;
		accuracy_not_increased_for = 0;
	}
	else accuracy_not_increased_for += test_freq;
	if (accuracy_not_increased_for > accuracy_stuck_limit) [[unlikely]] {
		lr /= delta_lr;
		accuracy_stuck_limit *= delta_accuracy_stuck_limit;
		accuracy_not_increased_for = 0;
	}

	std::cout << "\rEpoch " << epoch + total_epochs << " | "
		<< "Accuracy: " << accuracy << "% | "
		<< "Best accuracy: " << best_accuracy << "%\33[0K";
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