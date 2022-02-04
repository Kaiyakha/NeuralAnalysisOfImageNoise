#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/Eigen.h>

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
	function_* actfuncs;
	function_* actfunc_ders;
	double* func_params;
	void backprop(const VectorXd& Y, const double lr);

public:
	NeuralNetwork(const py::tuple& shape, const py::dict& config);
	void inspect() const;
	VectorXd forwardprop(const VectorXd& X);
	void train(const MatrixXd& input, const MatrixXd& target, const double lr, const unsigned epochs, const unsigned test_freq);
	const float test(const MatrixXd& X, const MatrixXd& Y);
	~NeuralNetwork();
};

const VectorXd sigmoid(const VectorXd& X, double width);
const VectorXd sigmoid_der(const VectorXd& X, double width);
const VectorXd ReLU(const VectorXd& X, double angle);
const VectorXd ReLU_der(const VectorXd& X, double angle);

function_ get_function_by_name(const std::string& name);
function_ get_function_der_by_name(const std::string& name);

const VectorXi argsort(VectorXd X);
const Index intersect1d_len(VectorXi X, VectorXi Y);