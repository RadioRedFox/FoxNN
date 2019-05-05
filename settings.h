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

class Settings
{
public:
	Settings()
	{
		n_threads = 1;
		max_on_last_layer = 0;
		one_if_value_greater_intermediate_value = 0;
		intermediate_value = 0.5;
	}

	Settings(ifstream& open_file)
	{
		open_file >> n_threads;
		open_file >> max_on_last_layer;
		open_file >> one_if_value_greater_intermediate_value;
		open_file >> intermediate_value;
	}

	void save(ofstream& open_file) const
	{
		open_file << n_threads << endl;
		open_file << max_on_last_layer << endl;
		open_file << one_if_value_greater_intermediate_value << endl;
		open_file << intermediate_value << endl;
	}

	size_t n_threads;
	bool max_on_last_layer;
	bool one_if_value_greater_intermediate_value;
	double intermediate_value;
};