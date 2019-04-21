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
	layer(const size_t &N_in, const size_t &N_neurons,  activation_function_base_class const * activ_f = nullptr) : neurons()
	{
		neurons.reserve(N_neurons);
		for (size_t i = 0; i < N_neurons; i++)
			neurons.push_back(shared_ptr<neuron> (new neuron(N_in)));

		if (activ_f == nullptr)
			res_function = activation_function(new sigmoid());
		else
			res_function = get_activation_function(activ_f);
	}

	layer(const vector <vector <double>> &v_w, activation_function_base_class const * activ_f = nullptr)
	{
		neurons.reserve(v_w.size());
		for (vector <vector <double>>::const_iterator w = v_w.cbegin(); w < v_w.end(); ++w)
			neurons.push_back(shared_ptr<neuron> (new neuron(*w)));

		if (activ_f == nullptr)
			res_function = activation_function(new sigmoid());
		else
			res_function = get_activation_function(activ_f);
	}

	layer(const size_t &N_in, const size_t &N_neurons, const activation_function activ_f = nullptr) : neurons()
	{
		neurons.reserve(N_neurons);
		for (size_t i = 0; i < N_neurons; i++)
			neurons.push_back(shared_ptr<neuron> (new neuron(N_in)));

		if (activ_f == nullptr)
			res_function = activation_function(new sigmoid());
		else
			res_function = get_activation_function(activ_f);
	}

	layer(const vector <vector <double>> &v_w, activation_function activ_f = nullptr)
	{
		neurons.reserve(v_w.size());
		for (vector <vector <double>>::const_iterator w = v_w.cbegin(); w < v_w.end(); ++w)
			neurons.push_back(shared_ptr<neuron>(new neuron(*w)));

		if (activ_f == nullptr)
			res_function = activation_function(new sigmoid());
		else
			res_function = get_activation_function(activ_f);
	}

	layer(const layer &a)
	{
		for (size_t i = 0; i < a.neurons.size(); ++i)
			neurons.push_back(shared_ptr<neuron> (new neuron(*(a.neurons[i]))));

		res_function = get_activation_function(a.res_function);
	}
	
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

	void get_error(vector <double> &out_error, const vector <double> &error, const vector <double> enter) const
	{
		const size_t N_neurons = neurons.size();
		const size_t N_enter = enter.size();
		vector <double> d_out(N_neurons);
		vector <vector <double>> for_sum_error(N_enter);


		for (vector<vector <double>>::iterator v = for_sum_error.begin(); v < for_sum_error.end(); ++v)
			v->resize(N_neurons);

		transform(neurons.cbegin(), neurons.cend(), d_out.begin(), [&](const shared_ptr<neuron> n) {return n->get_d_out(enter, res_function); }); //	d_out[i] = neurons[i]->get_d_out(enter)

		for (size_t i = 0; i < N_enter; ++i)
			for (size_t j = 0; j < N_neurons; ++j)
				for_sum_error[i][j] = error[j] * d_out[j] * neurons[j]->get_w(i);

		for (vector<vector <double>>::iterator v = for_sum_error.begin(); v < for_sum_error.end(); ++v)
			sort(v->begin(), v->end(), f_abs_sort);

		out_error.resize(N_enter);
		transform(for_sum_error.cbegin(), for_sum_error.cend(), out_error.begin(), [](const vector<double> &sum_error)
		{
			return accumulate(sum_error.cbegin(), sum_error.cend(), 0.0);
		}); // out_error[i] = sum(for_sum_error[i])
	}

	void back_running(vector <double> &error, const vector <double> &enter, vector <vector <double>> &d) const
	{
		const size_t N_neurons = neurons.size();
		vector <double> out_error;
		get_error(out_error, error, enter);

		for (size_t i = 0; i < N_neurons; ++i)
			neurons[i]->derivate_w(error[i], enter, d[i], res_function);
		error = move(out_error);
	}

	void get_out(const vector <double> &enter, vector <double> &out) const
	{
		out.resize(neurons.size());
		transform(neurons.cbegin(), neurons.cend(), out.begin(), [&](const shared_ptr<neuron> n) {return n->get_out(enter, res_function); }); //out[i] = neurot[i].get_out(entr)
	}

	void correction_of_scales(const vector <vector <double>> &d)
	{
		const size_t N_neurons = neurons.size();
		for (size_t i = 0; i < N_neurons; ++i)
			neurons[i]->correction_of_scales(d[i]);
		return;
	}

	void get_all_w(vector <vector <double>> &w) const
	{
		const size_t N_neurons = neurons.size();
		w.resize(N_neurons);
		for (size_t i = 0; i < N_neurons; i++)
			neurons[i]->get_w(w[i]);
		return;
	}

	size_t get_N_w(void) const
	{
		return neurons[0]->get_N();
	}

	size_t get_N_n(void) const
	{
		return neurons.size();
	}

	void random_mutation(const double &speed)
	{
		const size_t N_neurons = neurons.size();
		for (size_t i = 0; i < N_neurons; ++i)
			neurons[i]->random_mutation(speed);
	}

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
				neurons.push_back(shared_ptr<neuron>(new neuron(*(a.neurons[i]))));

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

	void set_activation_function (const string& name) //for Python
	{
		auto new_activation_function = get_activation_function(name);
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

private:
	vector <shared_ptr<neuron>> neurons;
	activation_function res_function;
};
