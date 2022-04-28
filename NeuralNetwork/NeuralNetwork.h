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
	py::list function_names;
	function_* actfuncs;
	function_* actfunc_ders;
	double* func_params;

	void allocate_memory();
	void set_activation_functions();
	void backprop(const VectorXd& Y, const double lr);
	void run_test(const unsigned epoch, const MatrixXd& input, const MatrixXd& target);
	void dump(const std::string& filename) const;
	void load(const std::string& filename);

public:
	NeuralNetwork(const py::dict& config);
	NeuralNetwork(const std::string& dumpfile);
	void inspect() const;
	const VectorXd& forwardprop(const VectorXd& X);
	void train(const MatrixXd& input, const MatrixXd& target, const py::dict& config);
	const float test(const MatrixXd& X, const MatrixXd& Y);
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