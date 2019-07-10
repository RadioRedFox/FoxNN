#pragma once

#include "activation_function.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random> 
#include <omp.h>
#include <iterator>
#include <algorithm>
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
	//Creation of a neuron with random filling of scales
	neuron(const size_t &N)
	{
		random_device rd;
		std::mt19937 gen(rd());
		uniform_real_distribution<> urd(-1, 1);
		w.resize(N + 1);
		generate(w.begin(), w.end(), [&]() {return urd(gen); });
	}

	//Creating a neuron by reading weights from a file
	neuron(ifstream& open_file)
	{
		size_t size_w;
		open_file >> size_w;
		w.resize(size_w);
		w.shrink_to_fit();
		for (size_t i = 0; i < size_w; ++i)
			open_file >> w[i];
	}

	//copy weights constructor
	neuron(const vector <double> &new_w)
	{
		w.resize(new_w.size());
		copy(new_w.cbegin(), new_w.cend(), w.begin());
	}

	//copy constructor
	neuron(const neuron &a)
	{
		w.resize(a.w.size());
		copy(a.w.cbegin(), a.w.cend(), w.begin());
	}

	//move constructor
	neuron(neuron &&a) noexcept
	{
		w = move(a.w);
		a.w.clear();
	}

	//derivative at the point
	double get_d_out(const vector <double> &enter, const activation_function &func, const bool &correct_summation = false) const
	{
		const double sum = scalar_product(enter, correct_summation); //w[0]*enter[0] + w[1]*enter[1] + ...
		return func->get_derivative_out(sum);
	}

	//The calculation of the derivative(momentum) for the weights
	void derivate_w(const double &error, const vector <double> &enter, const activation_function &func, const bool& correct_summation = false)
	{
		const size_t N_w = w.size();
		const double sum = scalar_product(enter, correct_summation);  //w[0]*enter[0] + w[1]*enter[1] + ...
		const double d_res_function_multiplied_error = func->get_derivative_out(sum) * error; // f'(sum) * error

		for (size_t i = 0; i < N_w - 1; ++i)
		{
#pragma omp atomic
			optimization[i]->derivative += enter[i] * d_res_function_multiplied_error; // d[i] = d[i] + enter[i] * error * f'(sum)
		}

#pragma omp atomic
		optimization.back()->derivative -= d_res_function_multiplied_error;

		return;
	}

	//change the weights after calculating the momentum
	void correction_of_scales(const double& speed, const Settings &settings)
	{
		for (size_t i = 0; i < w.size(); ++i)
		{
			//w[i] -= speed * derivative[i]; //w[i] = w[i] - d[i]
			optimization[i]->correction_of_scales(w[i], speed ,settings);
			optimization[i]->derivative = 0.0; //zeroing to count the momentum at the next iteration
		}
	}

	//calculate the value of the neuron
	double get_out(const vector <double> &enter, const activation_function &func, const bool& correct_summation = false) const
	{
		const double sum = scalar_product(enter, correct_summation); //w[0]*enter[0] + w[1]*enter[1] + ...
		return  func->get_out(sum); //f(sum)
	}

	//outputs the number of weights (excluding the last)
	size_t get_N(void) const
	{
		return w.size() - 1;
	}

	//randomly change the weights to a random value
	void random_mutation(const double &speed)
	{
		const size_t N_w = w.size();
		random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> urd(-1, 1);
		for (size_t i = 0; i < N_w; ++i)
			w[i] += urd(gen) * speed;
	}

	//change of weights by a value commensurate with the value of weights
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

	//operator = copy
	neuron& operator= (const neuron &a)
	{
		if (this != &a)
		{
			w.resize(a.w.size());
			copy(a.w.cbegin(), a.w.cend(), w.begin());
		}
		return *this;
	}

	////operator = move
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

	//w[0]*enter[0] + w[1]*enter[1] + ...
	double scalar_product(const vector <double>& enter, const bool& correct_summation) const
	{
		double sum;

		if (correct_summation == false)
		{
			sum = inner_product(enter.cbegin(), enter.cend(), w.cbegin(), 0.0); ////w[0]*enter[0] + w[1]*enter[1] + ...
			sum += -w.back();
		}
		else //before summing a large array of small numbers for accuracy they are sorted
		{
			vector <double> for_sum(w.size());
			transform(enter.cbegin(), enter.cend(), w.cbegin(), for_sum.begin(), [](const double& a, const double& b) {return a * b; }); //for_sum[i] = enter[i] * w[i]
			for_sum.back() = -w.back(); // for_sum[N_w - 1] =  - 1 * w[N_w - 1]
			sort(for_sum.begin(), for_sum.end(), f_abs_sort);
			sum = accumulate(for_sum.cbegin(), for_sum.cend(), 0.0);
		}
		return sum;
	}

	//saving a neuron to a file
	void save(ofstream& open_file) const
	{
		open_file << w.size() << endl;
		for (size_t i = 0; i < w.size(); ++i)
			open_file << scientific << setprecision(15) << w[i] << endl;
	}

	//the memory allocation for the counting of the pulse
	void init_memory_for_train(const Settings &settings)
	{
		optimization.reserve(w.size());
		for (size_t i = 0; i < w.size(); ++i)
			optimization.push_back(get_optimization(settings.settings_optimization.mode));
	}

	//free memory after training
	void delete_memory_after_train()
	{
		optimization.clear();
		optimization.shrink_to_fit();
	}

	vector <double> w; //scales
	vector <shared_ptr<base_class_optimization>> optimization;
};