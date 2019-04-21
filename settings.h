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

class Settings
{
public:
	Settings()
	{
		n_threads = 1;
	}

	Settings(ifstream& open_file)
	{
		open_file >> n_threads;
	}

	void save(ofstream& open_file) const
	{
		open_file << n_threads << endl;
	}

	size_t n_threads;
};