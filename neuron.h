#pragma once

#include "activation_function.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random> 
#include <string>
#include <ctime>
#include <cmath>
#include <omp.h>
#include <iterator>
#include <algorithm>
#include <set>
#include <numeric>
#include <memory>


using namespace std;

bool f_abs_sort(const double &a, const double &b)
{
	return (fabs(a) <= fabs(b));
}

class neuron
{
public:
	neuron(const size_t &N)
	{
		random_device rd;
		std::mt19937 gen(rd());
		uniform_real_distribution<> urd(-1, 1);
		w.resize(N + 1);
		generate(w.begin(), w.end(), [&]() {return urd(gen); });
	}

	neuron(const vector <double> &new_w)
	{
		w.resize(new_w.size());
		copy(new_w.cbegin(), new_w.cend(), w.begin());
	}

	neuron(const neuron &a)
	{
		w.resize(a.w.size());
		copy(a.w.cbegin(), a.w.cend(), w.begin());
	}

	neuron(const neuron * const a)
	{
		w.resize(a->w.size());
		copy(a->w.cbegin(), a->w.cend(), w.begin());
	}

	neuron(neuron &&a) noexcept
	{
		w = move(a.w);
		a.w.clear();
	}

	double get_d_out(const vector <double> &enter, const activation_function &func) const
	{
		vector <double> for_sum(w.size());

		transform(w.cbegin(), w.cend() - 1, enter.cbegin(), for_sum.begin(), [](const double &a, const double &b) {return a * b; });

		for_sum.back() = -w.back();

		sort(for_sum.begin(), for_sum.end(), f_abs_sort);

		const double sum = accumulate(for_sum.cbegin(), for_sum.cend(), 0.0);

		return func->get_derivative_out(sum);
	}

	void derivate_w(const double &error, const vector <double> &enter, vector <double> &d, const activation_function &func) const
	{
		const size_t N_w = w.size();
		vector <double> for_sum(N_w);

		transform(w.cbegin(), w.cend() - 1, enter.cbegin(), for_sum.begin(), [](const double &a, const double &b) {return a * b; }); //for_sum[i] = enter[i] * w[i]
		for_sum.back() = -w.back(); // for_sum[N_w - 1] =  - 1 * w[N_w - 1]

		sort(for_sum.begin(), for_sum.end(), f_abs_sort);
		const double sum = accumulate(for_sum.cbegin(), for_sum.cend(), 0.0);

		const double d_res_function_multiplied_error = func->get_derivative_out(sum) * error;

		for (size_t i = 0; i < N_w - 1; ++i)
		{
#pragma omp atomic
			d[i] += enter[i] * d_res_function_multiplied_error; // d[i] = d[i] + enter[i] * error * derivative_res_function(sum)
		}

#pragma omp atomic
		d.back() += -d_res_function_multiplied_error;

		return;
	}

	void correction_of_scales(const vector <double> &d)
	{
		transform(w.cbegin(), w.cend(), d.cbegin(), w.begin(), [](const double &a, const double &b) { return a - b; }); //w[i] = w[i] - d[i]
	}

	double get_out(const vector <double> &enter, const activation_function &func) const
	{
		vector <double> for_sum(w.size());

		transform(w.cbegin(), w.cend() - 1, enter.cbegin(), for_sum.begin(), [](const double &a, const double &b) {return a * b; });

		for_sum.back() = -w.back();

		sort(for_sum.begin(), for_sum.end(), f_abs_sort);

		const double sum = accumulate(for_sum.cbegin(), for_sum.cend(), 0.0);

		return  func->get_out(sum);
	}

	double get_w(const size_t &i) const
	{
		return w[i];
	}

	void get_w(vector <double> &ww) const
	{
		ww.resize(w.size());
		copy(w.begin(), w.end(), ww.begin());
		return;
	}

	size_t get_N(void) const
	{
		return w.size() - 1;
	}

	void random_mutation(const double &speed)
	{
		const size_t N_w = w.size();
		random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> urd(-1, 1);
		for (size_t i = 0; i < N_w; ++i)
			w[i] += urd(gen) * speed;
	}

	void smart_mutation(const double &speed)
	{
		if (speed >= 0 && speed <= 0)
			return;
		const size_t N_w = w.size();
		random_device rd;
		std::mt19937 gen(rd());
		for (size_t i = 0; i < N_w; ++i)
		{
			std::uniform_real_distribution<> urd(w[i] - fabs(w[i] / speed), w[i] + fabs(w[i] / speed));
			w[i] += urd(gen) * speed;
		}
	}

	neuron& operator= (const neuron &a)
	{
		if (this != &a)
		{
			w.resize(a.w.size());
			copy(a.w.cbegin(), a.w.cend(), w.begin());
		}
		return *this;
	}

	neuron& operator= (neuron &&a) noexcept
	{
		if (this != &a)
		{
			w = move(a.w);
			a.w.clear();
		}
		return *this;
	}
private:
	vector <double> w; //scales
};