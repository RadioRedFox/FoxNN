//Copyright[2019][Gaganov Ilya]
//Licensed under the Apache License, Version 2.0

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

template <typename T>
void write_line(const T &arg, ofstream& open_file)
{
	if (typeid(arg) == typeid(double))
		open_file << scientific << setprecision(15) << arg << endl;
	else
		open_file << arg << endl;
}

class Nesterov
{
public:
	Nesterov()
	{
		gamma = 0.9;
	}

	Nesterov(ifstream& open_file)
	{
		open_file >> gamma;
		set_gamma(gamma);
	}

	void save(ofstream& open_file) const
	{
		write_line(gamma, open_file);
	}

	void print_settings() const
	{
		cout << "settings.settings_optimization.Nesterov.gamma = " << gamma << endl;
	}

	void set_gamma(const double &new_gamma)
	{
		if (new_gamma <= 0)
		{
			gamma = 0.01;
			return;
		}
		if (new_gamma >= 1)
		{
			gamma = 0.99;
			return;
		}
		gamma = new_gamma;
	}
private:
	friend class Nesterov_optimization;
	double gamma;
};

class Adam
{
public:
	Adam()
	{
		betta_1 = 0.9;
		betta_2 = 0.999;
		epsilon = 0.00000001;
		step = 0;
		step_to_zero_after_n = 1000;
	}

	Adam(ifstream& open_file)
	{
		open_file >> betta_1;
		open_file >> betta_2;
		open_file >> epsilon;
		open_file >> step_to_zero_after_n;
		set_betta_1(betta_1);
		set_betta_2(betta_2);
		set_epsilon(epsilon);
		step = 0;
	}

	void next_step()
	{
		step++;
		if (step_to_zero_after_n == 0)
			return;
		if (step > step_to_zero_after_n)
			step = 0;
	}

	void save(ofstream& open_file) const
	{
		write_line(betta_1, open_file);
		write_line(betta_2, open_file);
		write_line(epsilon, open_file);
		write_line(step_to_zero_after_n, open_file);
	}

	void print_settings() const
	{
		cout << "settings.settings_optimization.Adam.betta_1 = " << betta_1 << endl;
		cout << "settings.settings_optimization.Adam.betta_2 = " << betta_2 << endl;
		cout << "settings.settings_optimization.Adam.epsilon = " << epsilon << endl;
	}

	void set_betta_1(const double& new_betta_1)
	{
		if (new_betta_1 <= 0)
		{
			betta_1 = 0.01;
			return;
		}
		if (new_betta_1 >= 1)
		{
			betta_1 = 0.99;
			return;
		}
		betta_1 = new_betta_1;
	}
	void set_betta_2(const double& new_betta_2)
	{
		if (new_betta_2 <= 0)
		{
			betta_2 = 0.01;
			return;
		}
		if (new_betta_2 >= 1)
		{
			betta_2 = 0.99;
			return;
		}
		betta_2 = new_betta_2;
	}
	void set_epsilon(const double& new_epsilon)
	{
		if (new_epsilon <= 0)
		{
			epsilon = 0.01;
			return;
		}
		epsilon = new_epsilon;
	}

	size_t step_to_zero_after_n;

private:
	friend class Adam_optimization;
	friend class neural_network;
	double betta_1;
	double betta_2;
	double epsilon;
	size_t step; //t
	
};

class Settings_optimization
{
public:

	Settings_optimization() : mode("SGD") {}

	Adam adam;
	Nesterov nesterov;

	void save(ofstream& open_file) const
	{
		adam.save(open_file);
		nesterov.save(open_file);
		write_line(mode, open_file);
	}

	Settings_optimization(ifstream& open_file)
	{
		adam = Adam(open_file);
		nesterov = Nesterov(open_file);
		open_file >> mode;
		set_mode(mode);
	}

	void print_settings() const
	{
		cout << "mode optimization = " << mode << endl;
		adam.print_settings();
		nesterov.print_settings();
	}
 
	void set_mode(const string& next_mode)
	{
		if (next_mode != "Adam" and next_mode != "Nesterov" and next_mode != "SGD")
			mode = "SGD";
		else
			mode = next_mode;
	}
private:
	friend class Optimization;
	friend class neuron;
	string mode;
};

class Settings
{
public:
	Settings()
	{
		n_threads = 1;
		n_print = 0;
		min_error = 0.01;
		max_on_last_layer = 0;
		one_if_value_greater_intermediate_value = 0;
		intermediate_value = 0.5;
		part_for_test = 0.1;
		auto_save_name_file = "auto_save.txt";
		auto_save_iteration = 0;
		correct_summation = 0;
	}

	Settings(ifstream& open_file)
	{
		open_file >> n_threads;
		open_file >> n_print;
		open_file >> min_error;
		open_file >> max_on_last_layer;
		open_file >> one_if_value_greater_intermediate_value;
		open_file >> intermediate_value;
		open_file >> part_for_test;
		open_file >> auto_save_name_file;
		open_file >> auto_save_iteration;
		open_file >> correct_summation;
		set_part_for_test(part_for_test);
		settings_optimization = Settings_optimization(open_file);
	}

	void save(ofstream& open_file) const
	{
		write_line(n_threads, open_file);
		write_line(n_print, open_file);
		write_line(min_error, open_file);
		write_line(max_on_last_layer, open_file);
		write_line(one_if_value_greater_intermediate_value, open_file);
		write_line(intermediate_value, open_file);
		write_line(part_for_test, open_file);
		write_line(auto_save_name_file, open_file);
		write_line(auto_save_iteration, open_file);
		write_line(correct_summation, open_file);
		settings_optimization.save(open_file);
	}

	void set_mode(const string& next_mode)
	{
		settings_optimization.set_mode(next_mode);
	}

	void set_part_for_test(const double value_for_part_for_test)
	{
		if (value_for_part_for_test <= 0)
		{
			part_for_test = 0;
			return;
		}
		if (value_for_part_for_test >= 1)
		{
			part_for_test = 1;
			return;
		}
		part_for_test = value_for_part_for_test;
	}

	void print_settings() const
	{
		cout << "settings: " << endl;
		cout << "n_threads = " << n_threads << endl;
		cout << "max_on_last_layer = " << max_on_last_layer << endl;
		cout << "one_if_value_greater_intermediate_value = " << one_if_value_greater_intermediate_value << endl;
		cout << "intermediate_value = " << intermediate_value << endl;
		cout << "n_print = " << n_print << endl;
		cout << "min_error = " << min_error << endl;
		cout << "part_for_test = " << part_for_test << endl;
		cout << "auto_save_name_file = " << auto_save_name_file << endl;
		cout << "auto_save_iteration = " << auto_save_iteration << endl;
		cout << "correct_summation = " << correct_summation << endl;
		settings_optimization.print_settings();
	}


	size_t n_threads;
	size_t n_print;
	double min_error;
	bool max_on_last_layer;
	bool one_if_value_greater_intermediate_value;
	double intermediate_value;
	string auto_save_name_file;
	size_t auto_save_iteration;
	bool correct_summation;
	Settings_optimization settings_optimization;
private:
	friend class neural_network;
	double part_for_test;
};