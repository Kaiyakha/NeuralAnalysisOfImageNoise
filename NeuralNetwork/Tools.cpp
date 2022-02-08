#include "NeuralNetwork.h"

function_ get_function_by_name(const std::string& name) {
	if (name == "sigmoid") return &sigmoid;
	else if (name == "ReLU") return &ReLU;
	else return nullptr;
}

function_ get_function_der_by_name(const std::string& name) {
	if (name == "sigmoid") return &sigmoid_der;
	else if (name == "ReLU") return &ReLU_der;
	else return nullptr;
}

const VectorXd sigmoid(const VectorXd& X, const double width) {
	VectorXd result(X.size());
	for (unsigned i = 0; i < result.size(); i++)
		result(i) = 1 / (1 + exp(-X(i) * width));
	return result;
}

const VectorXd sigmoid_der(const VectorXd& X, const double width) {
	VectorXd result(X.size());
	result = sigmoid(X, width).cwiseProduct((-sigmoid(X, width).array() + 1).matrix());
	return result;
}

const VectorXd ReLU(const VectorXd& X, const double angle) {
	VectorXd result(X.size());
	for (unsigned i = 0; i < result.size(); i++)
		result(i) = fmax(0, X(i) * angle);
	return result;
}

const VectorXd ReLU_der(const VectorXd& X, const double angle) {
	VectorXd result(X.size());
	for (unsigned i = 0; i < result.size(); i++)
		result(i) = 0 ? X(i) < 0 : angle;
	return result;
}

const VectorXi argsort(VectorXd X) {
	VectorXi indices = VectorXi::Zero(X.size());
	int max_ind;
	for (int i = 0; i < X.size(); i++) {
		X.maxCoeff(&max_ind);
		X(max_ind) = -DBL_MAX;
		indices(i) = max_ind;
	}
	return indices;
}

const Index intersect1d_len(VectorXi X, VectorXi Y) {
	int intersection_len = 0;
	std::vector<int> intersection;
	X.reverseInPlace(); Y.reverseInPlace();
	std::set_intersection(X.data(), X.data() + X.size(), Y.data(), Y.data() + Y.size(), std::back_inserter(intersection));
	return intersection.size();
}