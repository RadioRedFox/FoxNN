//Copyright[2019][Gaganov Ilya]
//Licensed under the Apache License, Version 2.0

#pragma once

#include "neuron.h"
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


using namespace std;

class layer
{
public:

	//creating layer
	layer(const size_t &N_in, const size_t &N_neurons, const activation_function activ_f = nullptr) : neurons()
	{
		neurons.reserve(N_neurons);
		for (size_t i = 0; i < N_neurons; i++)
			neurons.push_back(make_shared<neuron> (N_in));

		if (activ_f == nullptr)
			res_function = get_activation_function("sigmoid"); //default function
		else
			res_function = get_activation_function(activ_f); 
	}

	//creating layer
	layer(const vector <vector <double>> &v_w, activation_function activ_f = nullptr)
	{
		neurons.reserve(v_w.size());
		for (vector <vector <double>>::const_iterator w = v_w.cbegin(); w < v_w.end(); ++w)
			neurons.push_back(make_shared<neuron>(*w));

		if (activ_f == nullptr)
			res_function = get_activation_function("sigmoid"); //default function
		else
			res_function = get_activation_function(activ_f);
	}

	//copy constructor
	layer(const layer &a)
	{
		for (size_t i = 0; i < a.neurons.size(); ++i)
			neurons.push_back(make_shared<neuron> (*(a.neurons[i])));

		res_function = get_activation_function(a.res_function);
	}
	
	//creating layer from file
	layer(ifstream& open_file, const bool& only_scale = false)
	{
		if (only_scale == false)
			res_function = get_activation_function_from_file(open_file);
		else
			res_function = get_activation_function("sigmoid"); //default function
		size_t N_neuron;
		open_file >> N_neuron;
		neurons.reserve(N_neuron);
		for (size_t i = 0; i < N_neuron; ++i)
			neurons.push_back(make_shared<neuron>(open_file));
	}

	//move constructor
	layer (layer &&a) noexcept
	{
		for (size_t i = 0; i < a.neurons.size(); ++i)
		{
			neurons.push_back(move(a.neurons[i]));
			a.neurons[i] = nullptr;
		}
		a.neurons.clear();
		res_function = get_activation_function(a.res_function);
		a.res_function = nullptr;
	}

	//error count for the previous layer
	void get_error(vector <double> &out_error, const vector <double> &error, const vector <double> enter, const bool& correct_summation) const
	{
		const size_t N_neurons = neurons.size();
		const size_t N_enter = enter.size();
		vector <double> d_out(N_neurons);
		

		out_error.resize(N_enter, 0.0);
		//calculation of derivatives
		transform(neurons.cbegin(), neurons.cend(), d_out.begin(), [&](const shared_ptr<neuron> n) 
			{return n->get_d_out(enter, res_function, correct_summation); }); //	d_out[i] = neurons[i]->get_d_out(enter)

		if (correct_summation == false)
		{
			for (size_t i = 0; i < N_enter; ++i)
				for (size_t j = 0; j < N_neurons; ++j)
					out_error[i] += error[j] * d_out[j] * neurons[j]->w[i];
		}
		else //before summing a large array of small numbers for accuracy they are sorted
		{
			vector <vector <double>> for_sum_error(N_enter);

			for (vector<vector <double>>::iterator v = for_sum_error.begin(); v < for_sum_error.end(); ++v)
				v->resize(N_neurons);

			for (size_t i = 0; i < N_enter; ++i)
				for (size_t j = 0; j < N_neurons; ++j)
					for_sum_error[i][j] = error[j] * d_out[j] * neurons[j]->w[i];

			for (vector<vector <double>>::iterator v = for_sum_error.begin(); v < for_sum_error.end(); ++v)
				sort(v->begin(), v->end(), f_abs_sort);

			transform(for_sum_error.cbegin(), for_sum_error.cend(), out_error.begin(), [](const vector<double>& sum_error)
				{
					return accumulate(sum_error.cbegin(), sum_error.cend(), 0.0);
				}); // out_error[i] = sum(for_sum_error[i])
		}
	}


	void back_running(vector <double> &error, const vector <double> &enter, const bool& correct_summation)
	{
		const size_t N_neurons = neurons.size();
		vector <double> out_error;
		//error count for the previous layer
		get_error(out_error, error, enter, correct_summation);

		//The calculation of the derivative(momentum) for the weights
		for (size_t i = 0; i < N_neurons; ++i)
			neurons[i]->derivate_w(error[i], enter, res_function, correct_summation);
		error = move(out_error);
	}

	//calculate the value of the layer
	void get_out(const vector <double> &enter, vector <double> &out, const bool& correct_summation = false) const
	{
		out.resize(neurons.size());
		transform(neurons.cbegin(), neurons.cend(), out.begin(), [&](const shared_ptr<neuron> n) 
			{return n->get_out(enter, res_function, correct_summation); }); //out[i] = neurot[i].get_out(entr)
	}

	//change the weights after calculating the momentum
	void correction_of_scales(const double& speed, const Settings &setting)
	{
		for (size_t i = 0; i < neurons.size(); ++i)
			neurons[i]->correction_of_scales(speed, setting);
		return;
	}

	//get N value in enter
	size_t get_N_w(void) const
	{
		return neurons[0]->get_N();
	}

	//get N value in out
	size_t get_N_n(void) const
	{
		return neurons.size();
	}

	//randomly change the weights to a random value
	void random_mutation(const double &speed)
	{
		const size_t N_neurons = neurons.size();
		for (size_t i = 0; i < N_neurons; ++i)
			neurons[i]->random_mutation(speed);
	}

	//change of weights by a value commensurate with the value of weights
	void smart_mutation(const double &speed)
	{
		const size_t N_neurons = neurons.size();
		for (size_t i = 0; i < N_neurons; ++i)
			neurons[i]->smart_mutation(speed);
	}

	~layer()
	{
	}

	string get_name_activation_function() const
	{
		return res_function->name;
	}

	layer& operator= (const layer &a)
	{
		if (this != &a)
		{
			for (size_t i = 0; i < a.neurons.size(); ++i)
				neurons.push_back(make_shared<neuron>(*(a.neurons[i])));

			res_function = get_activation_function(a.res_function);
		}
		return *this;
	}

	layer& operator= (layer &&a) noexcept
	{
		if (this != &a)
		{
			for (size_t i = 0; i < a.neurons.size(); ++i)
			{
				neurons.push_back(move(a.neurons[i]));
				a.neurons[i] = nullptr;
			}
			a.neurons.clear();
			res_function = get_activation_function(a.res_function);
			a.res_function = nullptr;
		}
		return *this;
	}

	layer& operator= (activation_function activ_f)
	{
		if (activ_f != nullptr)
			res_function = get_activation_function(activ_f);
		return *this;
	}

	layer& operator= (const string &name)
	{
		auto new_activation_function = get_activation_function(name);
		if (new_activation_function != nullptr)
			res_function = new_activation_function;
		return *this;
	}

	void set_activation_function (const string &name, const vector<double>& parameters = {}) //for Python
	{
		auto new_activation_function = get_activation_function(name, parameters);
		if (new_activation_function != nullptr)
			res_function = new_activation_function;
		return;
	}

	void set_activation_function (activation_function activ_f) //for Python
	{
		if (activ_f != nullptr)
			res_function = get_activation_function(activ_f);
		return;
	}
	friend class neural_network;

	void print(const size_t& num_lauer = 0)
	{
		cout << "layer " << num_lauer << " n_in = " << neurons[0]->get_N() << " n_out = " << neurons.size() << " activation_function = " << res_function->name << endl;
	}
private:

	void init_memory_for_train(const size_t & size_batch, const Settings &settings)
	{
		for (size_t i = 0; i < neurons.size(); ++i)
			neurons[i]->init_memory_for_train(settings);

		output.resize(size_batch);
	}

	void delete_memory_after_train()
	{
		for (size_t i = 0; i < neurons.size(); ++i)
			neurons[i]->delete_memory_after_train();

		for (size_t i = 0; i < output.size(); ++i)
		{
			output[i].clear();
			output[i].shrink_to_fit();
		}
		output.clear();
		output.shrink_to_fit();
	}

	void save(ofstream& open_file, const bool& only_scale = false) const
	{
		if (only_scale == false)
			res_function->save(open_file);
		open_file << neurons.size() << endl;
		for (size_t i = 0; i < neurons.size(); ++i)
			neurons[i]->save(open_file);
	}


	vector <vector<double>> output;
	vector <shared_ptr<neuron>> neurons;
	activation_function res_function;
};
