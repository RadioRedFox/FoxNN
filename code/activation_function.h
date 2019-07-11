#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <random> 
#include <ctime>
#include <string>

#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>
#include <iterator>
#include <algorithm>
#include <set>
#include <numeric>
#include <memory>
#include <iomanip>
#include <map>

using namespace std;

class activation_function_base_class
{
public:
	activation_function_base_class(const string &name_activation_function, const vector <double> &new_parameters = {}) :name(name_activation_function)
	{
		parameters.resize(new_parameters.size());
		parameters.shrink_to_fit();
		copy(new_parameters.cbegin(), new_parameters.cend(), parameters.begin());
	}

	static void read_from_file(ifstream& open_file, vector <double>& new_parameters, string& name_activation_function)
	{
		open_file >> name_activation_function;
		size_t n_parameters;
		open_file >> n_parameters;
		new_parameters.resize(n_parameters);
		new_parameters.shrink_to_fit();
		for (size_t i = 0; i < n_parameters; ++i)
			open_file >> new_parameters[i];
	}

	virtual double get_out(const double &x) const = 0;

	virtual double get_derivative_out(const double &x) const = 0;

	void save(ofstream &open_file) const
	{
		open_file << name << endl;
		open_file << parameters.size() << endl;
		for (size_t i = 0; i < parameters.size(); ++i)
			open_file << setprecision(15) << parameters[i] << endl;
	}

	string name;
	vector <double> parameters;
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
				return 0.0000000000001;
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

};

class identity_x : public activation_function_base_class
{
public:
	identity_x() : activation_function_base_class("identity_x") {}

	double get_out(const double& x) const
	{
			return x;
	}

	double get_derivative_out(const double& x) const
	{
			return 1;
	}

};

class tan_h : public activation_function_base_class
{
public:
	tan_h() : activation_function_base_class("tan_h") {}

	double get_out(const double& x) const
	{
		double res = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
		if (res != res)
		{
			if (x > 0)
				return 1;
			else
				return 0.0000000000001;
		}
		return res;
	}

	double get_derivative_out(const double& x) const
	{
		double res = 1 - get_out(x) * get_out(x); // 1 - f(x)^2
		if (res >= 0.0 && res <= 0.0)
			return 0.0000000000001;
		return res;
	}
};

class arctan : public activation_function_base_class
{
public:
	arctan() : activation_function_base_class("arctan") {}

	double get_out(const double& x) const
	{
		double res = atan(x);
		if (res != res)
		{
			if (x > 0)
				return M_PI_2;
			else
				return -M_PI_2;
		}
		return res;
	}

	double get_derivative_out(const double& x) const
	{
		return 1 / (x * x + 1);
	}
};

class elu : public activation_function_base_class
{
public:
	elu(const vector<double>& new_parameters) : activation_function_base_class("elu", new_parameters)
	{
		if (new_parameters.size() == 0)
		{
			parameters.reserve(1);
			parameters.push_back(1.0);
		}
	}

	double get_out(const double& x) const
	{
		if (x >= 0.0)
			return x;
		double res = parameters[0] * (exp(x) - 1);
		if (res != res)
		{
			return -parameters[0];
		}
		return res;
	}

	double get_derivative_out(const double& x) const
	{
		double res = get_out(x) + parameters[0];
		if (res >= 0.0 && res <= 0.0)
			return 0.0000000000001;
		return res;
	}
};

enum name_functions {Sigmoid = 1, Sinusoid, Gaussian, Relu, Identity_x, Tan_h, Arctan, Elu};

map <string, size_t> get_map()
{
	map <string, size_t> map_functions;
	map_functions["sigmoid"] = Sigmoid;
	map_functions["sinusoid"] = Sinusoid;
	map_functions["gaussian"] = Gaussian;
	map_functions["relu"] = Relu;
	map_functions["identity_x"] = Identity_x;
	map_functions["tan_h"] = Tan_h;
	map_functions["arctan"] = Arctan;
	map_functions["elu"] = Elu;
	return map_functions;
}

void print_names_functions()
{
	map <string, size_t> map_functions = get_map();
	for (auto i = map_functions.cbegin(); i != map_functions.cend(); ++i)
		cout << i->first << endl;
}


activation_function get_activation_function(const string &name, const vector<double> &parameters = {})
{
	activation_function activ_f = nullptr;
	static map <string, size_t> map_functions = get_map();
	size_t num_function = 0;
	try
	{
		num_function = map_functions[name];
	}
	catch(...)
	{
		return nullptr;
	}
	switch (num_function)
	{
	case(Sigmoid):
		activ_f = activation_function(new sigmoid());
		break;
	case(Sinusoid):
		activ_f = activation_function(new sinusoid());
		break;
	case(Gaussian):
		activ_f = activation_function(new gaussian());
		break;
	case(Relu):
		activ_f = activation_function(new relu());
		break;
	case(Identity_x):
		activ_f = activation_function(new identity_x());
		break;
	case(Tan_h):
		activ_f = activation_function(new tan_h());
		break;
	case(Arctan):
		activ_f = activation_function(new arctan());
		break;
	case(Elu):
		activ_f = activation_function(new elu(parameters));
		break;
	default:
		return nullptr;
	}
	return activ_f;
}

activation_function get_activation_function_from_file(ifstream& open_file)
{
	string name;
	vector<double> parameters;
	activation_function_base_class::read_from_file(open_file, parameters, name);
	return get_activation_function(name, parameters);
}

activation_function get_activation_function(const activation_function copy_function)
{
	return get_activation_function(copy_function->name, copy_function->parameters);
}