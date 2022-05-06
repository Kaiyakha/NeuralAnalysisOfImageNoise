#include <omp.h>
#include <iostream>
#include "NeuralNetwork.h"


void NeuralNetwork::train_in_parallel_with_averaging(const int threads) {
	NeuralNetwork** family = new NeuralNetwork*[threads - 1];
	for (int i = 0; i < threads - 1; i++) family[i] = new NeuralNetwork(this);
	const unsigned global_epochs = epochs;

	omp_set_num_threads(threads);
#	pragma omp parallel
	{
		const int id = omp_get_thread_num();
		if (id) family[id - 1]->init_train(input, target, test_freq, lr);
		else this->init_train(input, target, test_freq, lr);

#		pragma omp barrier
		for (unsigned epoch = 0; epoch < global_epochs; epoch++) {
			if (id) family[id - 1]->train();
			else this->train();
#			pragma omp barrier
#			pragma omp master
			{
				this->average_state(family, threads);
				this->run_test(epoch * epochs);
			}
#			pragma omp barrier
			if (id) family[id - 1]->copy_state(this);
		}
	}

	for (int i = 0; i < threads - 1; i++) delete family[i];
	delete[] family;
}