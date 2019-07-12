//Copyright[2019][Gaganov Ilya]
//Licensed under the Apache License, Version 2.0

#pragma once
#include "layer.h"
#include "train_data.h"
#include "settings.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <random> 
#include <ctime>
#include <string>
#include <cmath>
#include <omp.h>
#include <iterator>
#include <algorithm>
#include <set>
#include <numeric>
#include <ostream>
#include <memory>
#include <iomanip>

using namespace std;

class neural_network
{
public:
	neural_network(void) {}

	neural_network(const string &name_file, const bool& only_scale = false)
	{
		ifstream file(name_file);

		if (only_scale == false)
			settings = Settings(file);

		size_t N_layers;

		file >> N_layers;
		layers.reserve(N_layers);

		for (size_t i = 0; i < N_layers; ++i)
			layers.push_back(make_shared <layer> (file, only_scale));

		file.close();
	}

	neural_network(const neural_network &a)
	{
		for (size_t i = 0; i < a.layers.size(); i++)
			layers.push_back(make_shared<layer> (*(a.layers[i])));
	}

	//the basic method of creating a network
	neural_network(const vector<int> &parameters)
	{
		bool error = false;
		for (int i : parameters)
			if (i < 1)
			{
				cout << "the number of input and output features must be positive" << endl;
				error = true;
			}
		if (error == false)
			for (size_t i = 0; i < parameters.size() - 1; ++i)
				layers.push_back(make_shared <layer>(parameters[i], parameters[i + 1], get_activation_function("sigmoid")));
	}

	//adding a layer to the end
	void next_layer(const layer &new_layer)
	{
		layers.push_back(make_shared<layer>(new_layer));
		return;
	}

	//to give the value of the network from the input
	vector<double> get_out(const vector<double> &first_in) const
	{

		vector<double> enter;
		vector<double> out;

		layers[0]->get_out(first_in, out);

		for (size_t i = 1; i < layers.size(); ++i)
		{
			enter = move(out);
			layers[i]->get_out(enter, out, settings.correct_summation);
		}
		correction_out(out);
		return  out;
	}

	vector<double> get_out(const one_train_data &first_in) const
	{
		return get_out(first_in.input);
	}

	void train_on_file(const string &name_file, const double &speed, const size_t &max_iteration, const size_t & size_train_batch = 1)
	{
		train_data test(name_file);
		train(test, speed, max_iteration, size_train_batch);
	}
   
	void train(train_data &data_for_train, const double& speed, const size_t& max_iteration, const size_t& size_train_batch = 1)
	{	
		double start_time;
		const size_t size_batch = get_batch_size(data_for_train.size(), size_train_batch);
		init_memory_for_train(size_batch);

		const size_t size_test = data_for_train.size() * settings.part_for_test;
		const train_data test = data_for_train.get_part_for_test(size_test);

		settings.settings_optimization.adam.step = 0;
		
		size_t n_data_for_only_train = data_for_train.size() * (1 - settings.part_for_test);
		if (n_data_for_only_train == 0 || n_data_for_only_train < size_batch)
			n_data_for_only_train = data_for_train.size();
		train_data data_for_only_train = data_for_train.get_first_n(n_data_for_only_train);


		for (size_t iteration = 1; iteration <= max_iteration; ++iteration)
		{
			start_train_progressbar(iteration, max_iteration, start_time);
			train_data batch(data_for_only_train.get_part(size_batch));

			train_nn(batch, speed);
			
			print_info_iteration(iteration, max_iteration, size_test, test, start_time);
			auto_save(iteration);
		}

		delete_memory_after_train();
	}

	void save(const string &name_file,  const bool &only_scale = false) const
	{
		ofstream file(name_file);
		
		if (only_scale == false)
			settings.save(file);

		file << layers.size() << endl;

		for (size_t i = 0; i < layers.size(); ++i)
			layers[i]->save(file, only_scale);

		file.close();
		return;
	}

	void random_mutation(const double &speed)
	{
#pragma omp parallel for  num_threads(settings.n_threads)
		for (size_t i = 0; i < layers.size(); ++i)
			layers[i]->random_mutation(speed);
	}

	void smart_mutation(const double &speed)
	{
#pragma omp parallel for  num_threads(settings.n_threads)
		for (size_t i = 0; i < layers.size(); ++i)
			layers[i]->smart_mutation(speed);
	}

	void print_info(void)
	{
		for (size_t i = 0; i < layers.size(); ++i)
			layers[i]->print(i);

		settings.print_settings();
	}

	double testing(const train_data& test) const
	{
		size_t n_true;
		const double er = get_error_for_test(test, n_true);
		cout << "error = " << scientific << setprecision(15) << er << " n_true = " << n_true << "/" << test.size() << endl;
		return er;
	}

	layer& operator[] (const size_t &i)
	{
		return *(layers[i]);
	}

	layer& get_layer (const size_t& i) //for Python
	{
		return *(layers[i]);
	}

	Settings settings;

private:

	void delete_memory_after_train()
	{
		for (auto i : layers)
			i->delete_memory_after_train();
	}

	void auto_save(const size_t &iteration) const
	{
		if (settings.auto_save_iteration != 0)
			if (iteration % settings.auto_save_iteration == 0)
				save(settings.auto_save_name_file);
	}

	void print_info_iteration(const size_t &iteration, const size_t &max_iteration, const size_t &size_test, const train_data &test, const double &start_time) const
	{
		train_progressbar(iteration, max_iteration, start_time);

		if (settings.n_print != 0 && size_test != 0)
			if (iteration == max_iteration || (iteration % settings.n_print == 0))
			{
				testing(test);
				cout << "iteration = " << iteration << endl << endl;
			}
	}

	void start_train_progressbar(const size_t &i, const size_t &max_iteration, double &start_time) const
	{
		if (settings.n_print == 0)
			return;
		if ((i - 1) % settings.n_print == 0)
		{
			start_time = omp_get_wtime();
			cout << "train progressbar: ";
			const size_t i_last_print = i - 1;
			const size_t n_iteration_between_print = (max_iteration >= i_last_print + settings.n_print) ? settings.n_print : max_iteration - i_last_print;
			start_progressbar(n_iteration_between_print);
		}
	}

	void start_progressbar(const size_t& max_iteration) const
	{
		const size_t len_str_max_iteration = std::to_string(max_iteration).size();
		cout << setw(len_str_max_iteration) << 0 << "/" << max_iteration << " (" << setw(3) << 0 << "%)";
	}

	void progressbar(const size_t& now, const size_t& max_iteration) const
	{
		const size_t len_str_max_iteration = std::to_string(max_iteration).size();
		const size_t proc = (now * 100) / max_iteration;
		string deleter;
		for (int i = 0; i < 2 * len_str_max_iteration + 8; ++i)
			deleter.push_back('\b');
		cout << deleter;
		cout << setw(len_str_max_iteration) << now << "/" << max_iteration << " (" << setw(3) << proc << "%)";
	}

	void train_progressbar(const size_t & i, const size_t & max_iteration, const double& start_time) const
	{
		if (settings.n_print == 0)
			return;
		const size_t i_after_print = i - ((i - 1) / settings.n_print) * settings.n_print;

		const size_t i_last_print = ((i - 1) / settings.n_print) * settings.n_print;

		const size_t n_iteration_between_print = (max_iteration < i_last_print + settings.n_print) ? max_iteration - i_last_print : settings.n_print;

		progressbar(i_after_print, n_iteration_between_print);

		if (i_after_print == n_iteration_between_print)
		{
			const double train_time = omp_get_wtime() - start_time;
			cout << " train_time = " << fixed << setprecision(3) << train_time << endl;
		}
	}

	size_t get_batch_size(const size_t &size_file_test, const size_t & size_train_batch) const
	{
		if (size_train_batch == 0 || size_train_batch >= size_file_test)
			return size_file_test;
		else
			return size_train_batch;
		return 1;
	}

	double get_error_for_test(const train_data &test, size_t &n_true) const
	{
		const double start_test = omp_get_wtime();
		double error = 0.0;
		size_t n_true_answer = 0;
		cout << "test progressbar: ";
		start_progressbar(test.size());
		size_t iteration_done = 0;
		omp_lock_t lock;
		omp_init_lock(&lock);
#pragma omp parallel for  num_threads(settings.n_threads)  reduction (+: error) reduction (+: n_true_answer) shared(test)
		for (size_t i = 0; i < test.size(); ++i) 
		{
			size_t need_max = 0;
			const vector <double> out = get_out(test[i]->input);
			for (size_t j = 0; j < out.size(); ++j)
			{
				const double delta = fabs(out[j] - test[i]->out[j]);
				error += delta;
				if (delta < settings.min_error)
					need_max++;
			}
			if (need_max == out.size())
				n_true_answer++;
			omp_set_lock(&lock);
			iteration_done++;
			progressbar(iteration_done, test.size());
			omp_unset_lock(&lock);
		}
		n_true = n_true_answer;
		omp_destroy_lock(&lock);

		const double time_test = omp_get_wtime() - start_test;
		cout << " time_test = " << fixed << setprecision(3) << time_test << endl;
		return error / test.size();
	}

	void error_last_layer(const train_data &batch, vector <vector <double>> &error) const
	{
		error.resize(batch.size());
		for (size_t i = 0; i < batch.size(); ++i)
			error[i].resize(batch[i]->out.size());

		for (size_t i = 0; i < batch.size(); ++i)
			for (size_t j = 0; j < batch[i]->out.size(); ++j)
				error[i][j] = layers.back()->output[i][j] - batch[i]->out[j];
		return;
	}

	void forward_stroke(const train_data &batch)
	{
		vector <shared_ptr<layer>>& layers2 = layers;
#pragma omp parallel for num_threads(settings.n_threads) shared(batch, layers2)
		for (size_t j = 0; j < batch.size(); ++j)
		{
			layers2[0]->get_out(batch[j]->input, layers2[0]->output[j]);

			for (size_t i = 1; i < layers2.size(); ++i)
				layers2[i]->get_out(layers2[i - 1]->output[j], layers2[i]->output[j]);
			correction_out(layers2.back()->output[j]);
		}
	}

	void correction_out(vector<double> &out) const
	{
		if (settings.max_on_last_layer == 1)
		{
			const size_t max_n = distance(out.begin(), max_element(out.begin(), out.end()));
			fill(out.begin(), out.end(), 0.0);
			out[max_n] = 1;
			return;
		}
		if (settings.one_if_value_greater_intermediate_value == 1)
		{
			for_each(out.begin(), out.end(), [&](double& num)
				{
					if (num >= settings.intermediate_value)
					{
						num = 1.0;
					}
					else
					{
						num = 0.0;
					}
				}
			);
			return;
		}
	}

	void init_memory_for_train(const size_t & size_batch)
	{
		for (size_t i = 0; i < layers.size(); ++i)
			layers[i]->init_memory_for_train(size_batch, settings);
	}

	void train_nn(const train_data & batch, const double &speed)
	{

		forward_stroke(batch);

		vector <vector <double>> error;
		vector <shared_ptr<layer>>& layers2 = layers;
		error_last_layer(batch, error);

#pragma omp parallel for  num_threads(settings.n_threads) shared(error, batch, layers2)
		for (size_t i = 0; i < batch.size(); ++i)
		{
			for (size_t j = layers2.size() - 1; j >= 1; --j)
				layers2[j]->back_running(error[i], layers2[j - 1]->output[i], settings.correct_summation);

			layers2[0]->back_running(error[i], batch[i]->input, settings.correct_summation);
		}

#pragma omp parallel for  num_threads(settings.n_threads) shared(layers2)
		for (int i = 0; i < layers2.size(); ++i)
			layers2[i]->correction_of_scales(speed, settings);

		settings.settings_optimization.adam.next_step();
		return;
	}

	vector <shared_ptr<layer>> layers;
};