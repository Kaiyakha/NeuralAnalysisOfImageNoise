#include "NeuralNetwork.h"
#include <fstream>

template <class Container> static void writeVector(const Container& V, std::ofstream& file);
static void writeMatrix(const MatrixXd& M, std::ofstream& file);
template <class Container> static const Container readVector(std::ifstream& file);
static const MatrixXd readMatrix(std::ifstream& file);


NeuralNetwork::NeuralNetwork(const std::string& dumpfile) {
	load(dumpfile);
	set_activation_functions();
}

void NeuralNetwork::dump(const std::string& filename) const {
	std::ofstream file(filename, std::ios::binary);
	if (!file.is_open()) throw std::runtime_error("failed to open a dump file");

	file.write((const char*)&layers, sizeof layers);
	file.write((const char*)shape, sizeof *shape * layers);
	for (unsigned i = 0; i < layers; i++) writeVector(activations[i], file);
	for (unsigned i = 0; i < layers - 1; i++) {
		writeMatrix(weights[i], file);
		writeVector(weighted_sums[i], file);
		writeVector(biases[i], file);
		writeVector(deltas[i], file);
		writeVector((const std::string)py::str(function_names[i]), file);
	}	
	file.write((const char*)func_params, sizeof *func_params * (layers - 1));
	file.write((const char*)&total_epochs, sizeof total_epochs);

	file.close();
}


void NeuralNetwork::load(const std::string& filename) {
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open()) throw std::runtime_error("failed to open a dump file");

	file.read((char*)&layers, sizeof layers);
	this->allocate_memory();

	file.read((char*)shape, sizeof *shape * layers);
	for (unsigned i = 0; i < layers; i++) activations[i] = readVector<VectorXd>(file);
	for (unsigned i = 0; i < layers - 1; i++) {
		weights[i] = readMatrix(file);
		weighted_sums[i] = readVector<VectorXd>(file);
		biases[i] = readVector<VectorXd>(file);
		deltas[i] = readVector<VectorXd>(file);
		function_names.append(readVector<std::string>(file));
	}
	file.read((char*)func_params, sizeof *func_params * (layers - 1));
	file.read((char*)&total_epochs, sizeof total_epochs);

	file.close();
}


template <class Container>
static void writeVector(const Container& V, std::ofstream& file) {
	size_t size = V.size();
	file.write((const char*)&size, sizeof size);
	file.write((const char*)V.data(), sizeof *V.data() * size);
}

static void writeMatrix(const MatrixXd& M, std::ofstream& file) {
	Index size[2] = { M.rows(), M.cols() };
	file.write((const char*)&size[0], sizeof size[0]);
	file.write((const char*)&size[1], sizeof size[1]);
	file.write((const char*)M.data(), sizeof *M.data() * M.size());
}

template <class Container>
static const Container readVector(std::ifstream& file) {
	size_t size;
	file.read((char*)&size, sizeof size);
	Container V; V.resize(size);
	file.read((char*)V.data(), sizeof *V.data() * size);
	return V;
}

static const MatrixXd readMatrix(std::ifstream& file) {
	Index size[2];
	file.read((char*)&size[0], sizeof size[0]);
	file.read((char*)&size[1], sizeof size[1]);
	MatrixXd M(size[0], size[1]);
	file.read((char*)M.data(), sizeof *M.data() * M.size());
	return M;
}