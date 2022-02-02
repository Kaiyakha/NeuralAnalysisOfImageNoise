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