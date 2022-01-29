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
	VectorXd(**actfuncs)(VectorXd&);
	VectorXd(**actfunc_ders)(VectorXd&);
	void backprop(VectorXd& Y, double lr);

public:
	NeuralNetwork(py::tuple& shape);
	void inspect() const;
	const VectorXd& forwardprop(VectorXd& X);
	void train(MatrixXd& input, MatrixXd& target, double lr, unsigned epochs);
	void test(MatrixXd& X, MatrixXd& Y);
	~NeuralNetwork();
};

static VectorXd sigmoid(VectorXd& X);
static VectorXd sigmoid_der(VectorXd& X);