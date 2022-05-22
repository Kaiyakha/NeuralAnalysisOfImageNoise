#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/Eigen.h>
#include "Rnd.h"

namespace py = pybind11;
using namespace Eigen;

typedef const VectorXd(*function_)(const VectorXd&, double);

class NeuralNetwork
{
private:
//	network configuration
	size_t layers;
	unsigned* shape;
	VectorXd* activations;
	MatrixXd* weights;
	VectorXd* weighted_sums;
	VectorXd* biases;
	VectorXd* deltas;
	py::list function_names;
	function_* actfuncs;
	function_* actfunc_ders;
	double* func_params;

//	train parameters
	Rnd<Index> rng;
	static MatrixXd *input, *target;
	VectorXd X, Y;
	Index i;
	unsigned epochs, test_freq;
	static double lr;
	static unsigned total_epochs;

//	dynamic learing rate
	static double delta_lr;
	static unsigned accuracy_stuck_limit;
	static unsigned delta_accuracy_stuck_limit;

//	private methods
	NeuralNetwork(const NeuralNetwork* src);
	void allocate_memory();
	void set_activation_functions();
	void backprop(const VectorXd& Y, const double lr) noexcept;
	void init_train(MatrixXd* input, MatrixXd* target, const py::dict& config);
	void init_train(const int threads, const int thread_num, const unsigned epochs);
	void train(void) noexcept;
	void monitor(const unsigned epoch) noexcept;
	void load(const std::string& filename);
	void copy_state(const NeuralNetwork* src);
	void average_state(NeuralNetwork* family[], const unsigned count);

	void train_in_parallel_with_averaging(int threads);

public:
	NeuralNetwork(const py::dict& config);
	NeuralNetwork(const std::string& dumpfile);
	void inspect(void) const noexcept;
	const VectorXd& forwardprop(const VectorXd& X);
	const float test(const MatrixXd *input, const MatrixXd *target);
	void dump(const std::string& filename) const;
	~NeuralNetwork();

	friend void launch_train(NeuralNetwork& network, MatrixXd* input, MatrixXd* target, const py::dict& config);
};

const VectorXd sigmoid(const VectorXd& X, const double width);
const VectorXd sigmoid_der(const VectorXd& X, const double width);
const VectorXd ReLU(const VectorXd& X, const double angle);
const VectorXd ReLU_der(const VectorXd& X, const double angle);

function_ get_function_by_name(const std::string& name);
function_ get_function_der_by_name(const std::string& name);

const VectorXi argsort(VectorXd X);
const Index intersect1d_len(VectorXi X, VectorXi Y);

const std::string get_terminator(void) noexcept;