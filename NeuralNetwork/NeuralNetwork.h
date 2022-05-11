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
	double lr;

	NeuralNetwork(const NeuralNetwork* src); // init a copy of a network
	void allocate_memory();
	void set_activation_functions();
	void backprop(const VectorXd& Y, const double lr);
	void init_train(const int threads, const int thread_num, const unsigned epochs, const double lr);
	void run_test(const unsigned epoch);
	void dump(const std::string& filename) const;
	void load(const std::string& filename);
	void copy_state(const NeuralNetwork* src);
	void average_state(NeuralNetwork* family[], const unsigned count);

	void train_in_parallel_with_averaging(const int threads);

public:
	NeuralNetwork(const py::dict& config);
	NeuralNetwork(const std::string& dumpfile);
	void inspect() const;
	const VectorXd& forwardprop(const VectorXd& X);
	void init_train(MatrixXd *input, MatrixXd *target, const py::dict& config);
	void train(void);
	const float test(const MatrixXd *input, const MatrixXd *target);
	~NeuralNetwork();
};

const VectorXd sigmoid(const VectorXd& X, const double width);
const VectorXd sigmoid_der(const VectorXd& X, const double width);
const VectorXd ReLU(const VectorXd& X, const double angle);
const VectorXd ReLU_der(const VectorXd& X, const double angle);

function_ get_function_by_name(const std::string& name);
function_ get_function_der_by_name(const std::string& name);

const VectorXi argsort(VectorXd X);
const Index intersect1d_len(VectorXi X, VectorXi Y);