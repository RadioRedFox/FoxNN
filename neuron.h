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
#include <iomanip>
#include "optimization.h"

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

	neuron(ifstream& open_file)
	{
		size_t size_w;
		open_file >> size_w;
		w.resize(size_w);
		w.shrink_to_fit();
		for (size_t i = 0; i < size_w; ++i)
			open_file >> w[i];
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

	neuron(neuron &&a) noexcept
	{
		w = move(a.w);
		a.w.clear();
	}

	double get_d_out(const vector <double> &enter, const activation_function &func, const bool &correct_summation = false) const
	{
		const double sum = scalar_product(enter, correct_summation);
		return func->get_derivative_out(sum);
	}

	void derivate_w(const double &error, const vector <double> &enter, const activation_function &func, const bool& correct_summation = false)
	{
		const size_t N_w = w.size();
		const double sum = scalar_product(enter, correct_summation);
		const double d_res_function_multiplied_error = func->get_derivative_out(sum) * error;

		for (size_t i = 0; i < N_w - 1; ++i)
		{
#pragma omp atomic
			optimization[i]->derivative += enter[i] * d_res_function_multiplied_error; // d[i] = d[i] + enter[i] * error * derivative_res_function(sum)
		}

#pragma omp atomic
		optimization.back()->derivative -= d_res_function_multiplied_error;

		return;
	}

	void correction_of_scales(const double& speed, const Settings &settings)
	{
		for (size_t i = 0; i < w.size(); ++i)
		{
			//w[i] -= speed * derivative[i]; //w[i] = w[i] - d[i]
			optimization[i]->correction_of_scales(w[i], speed ,settings);
			optimization[i]->derivative = 0.0;
		}
	}

	double get_out(const vector <double> &enter, const activation_function &func, const bool& correct_summation = false) const
	{
		const double sum = scalar_product(enter, correct_summation);
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
			std::uniform_real_distribution<> urd(w[i] - fabs(w[i] * speed), w[i] + fabs(w[i] * speed));
			w[i] = urd(gen);
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

	friend class layer;
private:

	double scalar_product(const vector <double>& enter, const bool& correct_summation) const
	{
		double sum;

		if (correct_summation == false)
		{
			sum = inner_product(enter.cbegin(), enter.cend(), w.cbegin(), 0.0);
			sum += -w.back();
		}
		else
		{
			vector <double> for_sum(w.size());
			transform(enter.cbegin(), enter.cend(), w.cbegin(), for_sum.begin(), [](const double& a, const double& b) {return a * b; }); //for_sum[i] = enter[i] * w[i]
			for_sum.back() = -w.back(); // for_sum[N_w - 1] =  - 1 * w[N_w - 1]
			sort(for_sum.begin(), for_sum.end(), f_abs_sort);
			sum = accumulate(for_sum.cbegin(), for_sum.cend(), 0.0);
		}
		return sum;
	}

	void save(ofstream& open_file) const
	{
		open_file << w.size() << endl;
		for (size_t i = 0; i < w.size(); ++i)
			open_file << scientific << setprecision(15) << w[i] << endl;
	}

	void init_memory_for_train(const Settings &settings)
	{
		optimization.reserve(w.size());
		for (size_t i = 0; i < w.size(); ++i)
			optimization.push_back(get_optimization(settings.settings_optimization.mode));
	}

	void delete_memory_after_train()
	{
		optimization.clear();
		optimization.shrink_to_fit();
	}
	vector <double> w; //scales
	vector <shared_ptr<base_class_optimization>> optimization;
};