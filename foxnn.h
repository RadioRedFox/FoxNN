#pragma once
#include "layer.h"
#include "additional_memory.h"
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

using namespace std;

class neural_network
{
public:
	neural_network(void) {}

	neural_network(const string &name_file)
	{
		ifstream file(name_file);
		size_t N_layers, N_in, N_n;
		vector <vector <double>> w;
		file >> N_layers;
		string name_function;
		vector<activation_function> activation_functions(N_layers, nullptr);
		for (size_t i = 0; i < N_layers; i++)
		{
			file >> name_function;
			activation_functions[i] = get_activation_function_from_file(name_function, file);
		}
		for (size_t i = 0; i < N_layers; i++)
		{
			file >> N_in;
			file >> N_n;
			w.resize(N_n);
			for (size_t j = 0; j < N_n; ++j)
			{
				w[j].resize(N_in + 1);
				for (size_t k = 0; k < N_in + 1; ++k)
					file >> w[j][k];
			}
			layers.push_back(shared_ptr <layer> (new layer(w, activation_functions[i])));
		}

		settings = Settings(file);

		file.close();
	}

	neural_network(const neural_network &a)
	{
		for (size_t i = 0; i < a.layers.size(); i++)
			layers.push_back(shared_ptr<layer> (new layer(*(a.layers[i]))));
	}

	neural_network(const vector<int> &parameters)
	{
		for (int i : parameters)
			if (i < 1)
				throw exception("the number of input and output features must be positive");
		for (size_t i = 0; i < parameters.size() - 1; i++)
			layers.push_back(shared_ptr <layer> (new layer(parameters[i], parameters[i + 1], get_activation_function("sigmoid"))));
	}

	void next_layer(const layer &new_layer)
	{
		shared_ptr<layer> copy_new_layer (new layer(new_layer));
		layers.push_back(move(copy_new_layer));
		return;
	}

	int get_out(const vector<double> &first_in, vector <double> &out) const
	{

		vector<double> enter;
		layers[0]->get_out(first_in, out);

		for (size_t i = 1; i < layers.size(); ++i)
		{
			enter = move(out);
			layers[i]->get_out(enter, out);
		}

		return  distance(out.begin(), max_element(out.begin(), out.end()));;
	}

	double get_out(const vector<double> &first_in) const
	{
		vector<double> enter;
		vector<double> out;

		layers[0]->get_out(first_in, out);

		enter = move(out);

		for_each(layers.begin() + 1, layers.end(), [&](const auto next_layer) {
			next_layer->get_out(enter, out);
			enter = move(out);
		});

		out = move(enter);
		return out[0];
	}

	void train_on_file(const string &name_file, const double &speed, const size_t &max_iteration, const double &min_error, const size_t &N_print, const size_t &size_part = 1)
	{
		vector <memory_layer> memory_layers;
		vector <derivative> derivativs;

		train_data test(name_file);

		size_t size_part_test = get_part_size(test.size(), size_part);

		create_memory_layer_and_derivativs(size_part_test, memory_layers, derivativs);

		int n_true;
		double er;

		for (int i = 1; i <= max_iteration; ++i)
		{
			train_data train_test(test.get_part(size_part_test));

			init_memory_layers_zero(train_test, memory_layers);
			for (size_t i = 0; i < derivativs.size(); ++i)
				derivativs[i].zero();
			forward_stroke(size_part_test, memory_layers);
			train(train_test, speed, memory_layers, derivativs);
			er = get_error_for_test(test, n_true, min_error);
			if (er < min_error || i == max_iteration)
			{
				cout << "iteration = " << i << " error = " << er << " n_true = " << n_true << endl;
				return;
			}
			if (i %  N_print == 0)
			{
				cout << "iteration = " << i << " error = " << er << " n_true = " << n_true << endl;
			}
		}

	}

	void save(const string &name_file) const
	{
		ofstream file(name_file);
		vector <vector <double>> w;
		file << layers.size() << endl;
		for (size_t i = 0; i < layers.size(); ++i)
			file << layers[i]->get_name_activation_function() << endl;
		for (size_t i = 0; i < layers.size(); ++i)
		{
			file << layers[i]->get_N_w() << " " << layers[i]->get_N_n() << endl;
			layers[i]->get_all_w(w);
			for (size_t j = 0; j < w.size(); j++)
				for (size_t k = 0; k < w[j].size(); k++)
					file << w[j][k] << endl;
		}
		settings.save(file);
		file.close();
		return;
	}

	void random_mutation(const double &speed)
	{
#pragma omp parallel for  num_threads(settings.n_threads)
		for (int i = 0; i < layers.size(); i++)
			layers[i]->random_mutation(speed);
	}

	void smart_mutation(const double &speed)
	{
#pragma omp parallel for  num_threads(settings.n_threads)
		for (int i = 0; i < layers.size(); i++)
			layers[i]->smart_mutation(speed);
	}

	void print_info(void)
	{
		for (size_t i = 0; i < layers.size(); ++i)
			cout << "layer " << i << " n_in = " << layers[i]->get_N_w() << " n_out = " << layers[i]->get_N_n() << " activation_function_base_class = " << layers[i]->get_name_activation_function() << endl;
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

	size_t get_part_size(const size_t &size_file_test, const size_t &size_part)
	{
		if (size_part == 0 || size_part >= size_file_test)
			return size_file_test;
		else
			return size_part;
		return 1;
	}

	double error_function(const vector <memory_layer> &memory_layers, const vector <vector <int>> &y) const
	{
		vector <double> for_sum;

		for_sum.reserve(y.size() * layers.back()->get_N_n());

		const size_t y_size = y.size();
		const size_t N_n_in_last_layer = layers.back()->get_N_n();

		for (size_t i = 0; i < y_size; i++)
			for (size_t j = 0; j < N_n_in_last_layer; j++)
				for_sum.push_back((memory_layers.back().enter[i][j] - y[i][j]) * (memory_layers.back().enter[i][j] - y[i][j]));

		sort(for_sum.begin(), for_sum.end(), f_abs_sort);

		const double res = accumulate(for_sum.cbegin(), for_sum.cend(), 0.0);

		return res / 2.0;
	}

	double get_error_for_test(const train_data &test, int &n_true, const double &error) const
	{

		double res = 0;
		n_true = 0;
		vector <double> out;
		for (size_t i = 0; i < test.size(); ++i)
		{
			int need_max = 0;
			get_out(test[i]->in, out);
			for (size_t j = 0; j < out.size(); ++j)
			{
				res += fabs(out[j] - test[i]->out[j]);
				if (fabs(out[j] - test[i]->out[j]) < error)
					need_max++;
			}
			if (need_max == out.size())
				n_true++;
		}
		return res / test.size();
	}

	void error_last_layer(const memory_layer &last_layer, const train_data &test, vector <vector <double>> &error) const
	{
		error.resize(test.size());
		for (size_t i = 0; i < test.size(); ++i)
			error[i].resize(test[i]->out.size());

		for (size_t i = 0; i < test.size(); ++i)
			for (size_t j = 0; j < test[i]->out.size(); ++j)
				error[i][j] = last_layer.enter[i][j] - test[i]->out[j];
		return;
	}

	void init_memory_layers_zero(const train_data &test, vector <memory_layer> &memory_layers) const
	{	
		for (size_t i = 0; i < test.size(); ++i)
		{
			memory_layers[0].enter[i].resize(test[i]->in.size());
			copy(test[i]->in.cbegin(), test[i]->in.cend(), memory_layers[0].enter[i].begin());
		}
	}

	void forward_stroke(const int &test_size, vector <memory_layer> &memory_layers) const
	{
#pragma omp parallel for num_threads(settings.n_threads)
		for (int j = 0; j < test_size; j++)
			for (size_t i = 0; i < layers.size(); i++)
				layers[i]->get_out(memory_layers[i].enter[j], memory_layers[i + 1].enter[j]);
	}

	void create_memory_layer_and_derivativs(const size_t &test_size, vector <memory_layer> &memory_layers, vector <derivative> &derivativs) const
	{
		memory_layers.resize(layers.size() + 1);
		for (size_t i = 0; i < memory_layers.size(); ++i)
			memory_layers[i].enter.resize(test_size);

		derivativs.resize(layers.size());

		for (size_t i = 0; i < derivativs.size(); ++i)
			derivativs[i].init(layers[i]->get_N_n(), layers[i]->get_N_w());
	}

	void train(const train_data &test, const double &speed, vector <memory_layer> &memory_layers, vector <derivative> &derivativs)
	{
		vector <vector <double>> error;

		error_last_layer(memory_layers.back(), test, error);
		const int layers_size_minus_1 = layers.size() - 1;
#pragma omp parallel for  num_threads(settings.n_threads)
		for (int i = 0; i < test.size(); i++)
			for (int j = layers_size_minus_1; j >= 0; --j)
				layers[j]->back_running(error[i], memory_layers[j].enter[i], derivativs[j].d);

#pragma omp parallel for  num_threads(settings.n_threads)	
		for (int i = 0; i < layers.size(); i++)
			for (size_t j = 0; j < derivativs[i].d.size(); j++)
				for (size_t k = 0; k < derivativs[i].d[j].size(); k++)
					derivativs[i].d[j][k] *= speed;

#pragma omp parallel for  num_threads(settings.n_threads)
		for (int i = 0; i < layers.size(); i++)
			layers[i]->correction_of_scales(derivativs[i].d);

		return;
	}

	vector <shared_ptr<layer>> layers;
};