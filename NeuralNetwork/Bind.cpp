#include "NeuralNetwork.h"

PYBIND11_MODULE(NeuralNetwork, mod) {
	py::class_<NeuralNetwork>(mod, "NeuralNetwork")
		.def(py::init<const py::dict&>())
		.def(py::init<const std::string&>())
		.def("inspect", &NeuralNetwork::inspect)
		.def("__call__", [](NeuralNetwork& network, const VectorXd& X) { return network.forwardprop(X); })
		.def("train", [](NeuralNetwork& network, const MatrixXd* input, const MatrixXd* target, const py::dict& config) {
				network.init_train(input, target, config);
				network.train();
			})
		.def("test", &NeuralNetwork::test);
}