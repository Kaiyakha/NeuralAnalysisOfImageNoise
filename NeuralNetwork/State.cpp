#include "NeuralNetwork.h"


void NeuralNetwork::copy_state(const NeuralNetwork* src) {
	std::copy(src->shape, src->shape + layers, this->shape);
	std::copy(src->weights, src->weights + layers - 1, this->weights);
	std::copy(src->biases, src->biases + layers - 1, this->biases);
	std::copy(src->actfuncs, src->actfuncs + layers - 1, this->actfuncs);
	std::copy(src->actfunc_ders, src->actfunc_ders + layers - 1, this->actfunc_ders);
	std::copy(src->func_params, src->func_params + layers - 1, this->func_params);
}


void NeuralNetwork::average_state(NeuralNetwork* family[], const unsigned count) {
	size_t l;

	for (unsigned id = 0; id < count - 1; id++) {
		for (l = 0; l < layers - 1; l++) {
			this->weights[l] += family[id]->weights[l];
			this->biases[l] += family[id]->biases[l];
		}
	}

	for (l = 0; l < layers - 1; l++) {
		this->weights[l] /= (double)count;
		this->biases[l] /= (double)count;
	}
}
