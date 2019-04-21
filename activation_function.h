#pragma once

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

class activation_function_base_class
{
public:
	activation_function_base_class(const string &name_activation_function) :name(name_activation_function) {}

	virtual double get_out(const double &x) const = 0;

	virtual double get_derivative_out(const double &x) const = 0;

	void save_name(ofstream &file)
	{
		file << name << endl;
	}

	virtual void save(ofstream &file) const = 0;

	const string name;
};

using activation_function = shared_ptr<activation_function_base_class const>;

class sigmoid : public activation_function_base_class
{
public:
	sigmoid() : activation_function_base_class("sigmoid") {}

	double get_out(const double &x) const
	{
		double res = 1.0 / (1.0 + exp(-x));
		if (res != res)
		{
			if (x > 0)
				return 1;
			else
				return 0.000000001;
		}
		return res;
	}

	double get_derivative_out(const double &x) const
	{
		double d_res = exp(-x) / ((1.0 + exp(-x))*(1.0 + exp(-x)));
		if (d_res != d_res)
			return 0.0000000001;
		return d_res;
	}

	void save(ofstream &file) const {}
};

class sinusoid : public activation_function_base_class
{
public:
	sinusoid() : activation_function_base_class("sinusoid") {}

	double get_out(const double &x) const
	{
		return sin(x);
	}

	double get_derivative_out(const double &x) const
	{
		return cos(x);
	}

	void save(ofstream &file) const	{}
};

class gaussian : public activation_function_base_class
{
public:
	gaussian() : activation_function_base_class("gaussian") {}

	double get_out(const double &x) const
	{
		double res = exp(-x * x);
		if (res != res)
				return 0.000000001;
		return res;
	}

	double get_derivative_out(const double &x) const
	{
		double d_res = - 2 * x * exp(-x * x);
		if (d_res != d_res)
		{
			if (x < 0)
				return 0.0000000001;
			else
				return -0.0000000001;
		}
		return d_res;
	}

	void save(ofstream &file) const {}
};

class relu : public activation_function_base_class
{
public:
	relu() : activation_function_base_class("relu") {}

	double get_out(const double &x) const
	{
		if (x < 0)
			return 0;
		else
			return x;
	}

	double get_derivative_out(const double &x) const
	{
		if (x < 0)
			return 0;
		else
			return 1;
	}

	void save(ofstream &file) const {}
};

activation_function need_name_only(const string &name)
{
	activation_function activ_f = nullptr;
	if (name == "sigmoid")
		activ_f = activation_function(new sigmoid());
	if (name == "sinusoid")
		activ_f = activation_function(new sinusoid());
	if (name == "gaussian")
		activ_f = activation_function(new gaussian());
	if (name == "relu")
		activ_f = activation_function(new relu());
	return activ_f;
}

activation_function get_activation_function_from_file(const string &name, ifstream &file)
{
	activation_function activ_f = need_name_only(name);
	return activ_f;
}

activation_function get_activation_function(const string &name, const vector<double> &parameters = {})
{
	activation_function activ_f = need_name_only(name);
	return activ_f;
}

activation_function get_activation_function(activation_function_base_class const * copy_function)
{
	activation_function activ_f = need_name_only(copy_function->name);
	return activ_f;
}

activation_function get_activation_function(const activation_function copy_function)
{
	activation_function activ_f = need_name_only(copy_function->name);
	return activ_f;
}