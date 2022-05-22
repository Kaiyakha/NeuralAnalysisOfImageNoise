#include "NeuralNetwork.h"


static void launch_train(NeuralNetwork& network, MatrixXd* input, MatrixXd* target, const py::dict& config) {
	network.init_train(input, target, config);
	if (py::bool_(config["parallel_training"])) {
		const int threads = py::int_(config["threads"]);
		network.train_in_parallel_with_averaging(threads);
	}
	else network.train();
	network.total_epochs += py::int_(py::float_(config["epochs"]));
}


PYBIND11_MODULE(NeuralNetwork, mod) {
	py::class_<NeuralNetwork>(mod, "NeuralNetwork")
		.def(py::init<const py::dict&>())
		.def(py::init<const std::string&>())
		.def("inspect", &NeuralNetwork::inspect)
		.def("__call__", [](NeuralNetwork& network, const VectorXd& X) { return network.forwardprop(X); })
		.def("train", &launch_train)
		.def("test", &NeuralNetwork::test)
		.def("dump", &NeuralNetwork::dump);
}