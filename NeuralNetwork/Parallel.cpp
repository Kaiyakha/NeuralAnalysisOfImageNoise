#include <omp.h>
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(const NeuralNetwork* src) {
	this->layers = src->layers;
	allocate_memory();

	std::copy(src->shape, src->shape + layers, this->shape);
	std::copy(src->actfuncs, src->actfuncs + layers - 1, this->actfuncs);
	std::copy(src->actfunc_ders, src->actfunc_ders + layers - 1, this->actfunc_ders);
	std::copy(src->func_params, src->func_params + layers - 1, this->func_params);

	for (unsigned l = 0; l < layers - 1; l++) {
		weights[l] = MatrixXd::Random(this->shape[l + 1], this->shape[l]);
		biases[l] = VectorXd::Zero(this->shape[l + 1]);
	}
}


void NeuralNetwork::init_train(const int threads, const int thread_num, const unsigned epochs) {
	this->epochs = epochs;
	this->test_freq = epochs + 1;

	static const Index block_size = input->rows() / static_cast<Index>(threads);
	const Index block_start = block_size * static_cast<Index>(thread_num);
	const Index block_end = block_start + block_size - 1;
	rng = Rnd<Index>(block_start, block_end);
}


void NeuralNetwork::train_in_parallel_with_averaging(int threads) {
	NeuralNetwork** family = new NeuralNetwork*[threads - 1];
	for (int i = 0; i < threads - 1; i++) family[i] = new NeuralNetwork(this);
	const unsigned global_epochs = epochs;

#	pragma omp parallel num_threads(threads)
	{
		const int id = omp_get_thread_num();
#		pragma omp single
		threads = omp_get_num_threads();

		if (id) family[id - 1]->init_train(threads, id, test_freq);
		else this->init_train(threads, id, test_freq);

#		pragma omp barrier
		for (unsigned epoch = 1; epoch <= global_epochs / epochs; epoch++) {
			if (id) family[id - 1]->train();
			else this->train();
#			pragma omp barrier
#			pragma omp master
			{
				this->average_state(family, threads);
				this->monitor(epoch * epochs);
			}
#			pragma omp barrier
			if (id) family[id - 1]->copy_state(this);
		}
	}

	for (int i = 0; i < threads - 1; i++) delete family[i];
	delete[] family;
}