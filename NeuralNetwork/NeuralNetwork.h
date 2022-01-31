#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/Eigen.h>

namespace py = pybind11;
using namespace Eigen;

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
	const VectorXd(**actfuncs)(const VectorXd&);
	const VectorXd(**actfunc_ders)(const VectorXd&);
	void backprop(const VectorXd& Y, const double lr);

public:
	NeuralNetwork(py::tuple& shape);
	void inspect() const;
	const VectorXd& forwardprop(const VectorXd& X);
	void train(const MatrixXd& input, const MatrixXd& target, const double lr, const unsigned epochs, const unsigned test_freq);
	const float test(const MatrixXd& X, const MatrixXd& Y);
	~NeuralNetwork();
};

static const VectorXd sigmoid(const VectorXd& X);
static const VectorXd sigmoid_der(const VectorXd& X);