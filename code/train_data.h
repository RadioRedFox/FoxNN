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
#include <memory>
#include <iomanip>

using namespace std;

class one_train_data
{
public:
	one_train_data() {}

	one_train_data(ifstream& open_file, const size_t& N_input, const size_t& N_out)
	{
		input.resize(N_input);
		input.shrink_to_fit();
		out.resize(N_out);
		out.shrink_to_fit();
		for (size_t j = 0; j < N_input; ++j)
			open_file >> input[j];

		for (size_t j = 0; j < N_out; ++j)
			open_file >> out[j];
	}

	one_train_data(const one_train_data &a)
	{
		input.resize(a.input.size());
		copy(a.input.cbegin(), a.input.cend(), input.begin());
		out.resize(a.out.size());
		copy(a.out.cbegin(), a.out.cend(), out.begin());
		input.shrink_to_fit();
		out.shrink_to_fit();
	}

	one_train_data(one_train_data &&a) noexcept
	{
		input = move(a.input);
		out = move(a.out);
		a.input.clear();
		a.out.clear();
		input.shrink_to_fit();
		out.shrink_to_fit();
	}

	one_train_data& operator= (const one_train_data &a)
	{
		if (this != &a)
		{
			input.resize(a.input.size());
			copy(a.input.cbegin(), a.input.cend(), input.begin());
			out.resize(a.out.size());
			copy(a.out.cbegin(), a.out.cend(), out.begin());
			input.shrink_to_fit();
			out.shrink_to_fit();
		}
		return *this;
	}

	one_train_data& operator= (one_train_data &&a) noexcept
	{
		if (this != &a)
		{
			input = move(a.input);
			out = move(a.out);
			a.input.clear();
			a.out.clear();
			input.shrink_to_fit();
			out.shrink_to_fit();
		}
		return *this;
	}

	one_train_data(const vector<double> &new_input, const vector<double> &new_out)
	{
		input.resize(new_input.size());
		copy(new_input.cbegin(), new_input.cend(), input.begin());
		out.resize(new_out.size());
		copy(new_out.cbegin(), new_out.cend(), out.begin());
		input.shrink_to_fit();
		out.shrink_to_fit();
	}

	vector<double> input;
	vector<double> out;

	friend class train_data;

private:


	void save(ofstream  &open_file) const
	{
		for (size_t j = 0; j < input.size(); ++j)
			open_file << scientific << setprecision(15) << input[j] << endl;

		for (size_t j = 0; j < out.size(); ++j)
			open_file << scientific << setprecision(15) << out[j] << endl;
	}

	
};

class train_data
{

public:
	train_data() {}

	train_data(const string &name_file)
	{
		ifstream file;
		file.exceptions(ifstream::badbit | ifstream::failbit);

		file.open(name_file);

		size_t N_test, N_input, N_out;

		file >> N_test;
		file >> N_input;
		file >> N_out;

		data.reserve(N_test);

		for (size_t i = 0; i < N_test; ++i)
			data.push_back(make_shared <const one_train_data> (file, N_input, N_out));
		file.close();
	}

	train_data(const train_data &a) 
	{
		data.resize(a.data.size());
		copy(a.data.cbegin(), a.data.cend(), data.begin());
	}

	train_data(train_data &&a) noexcept
	{
		data = move(a.data);
		a.data.clear();
	}

	train_data(vector<shared_ptr<const one_train_data>> &new_data)
	{
		data = move(new_data);
	}

	train_data& operator= (const train_data &a)
	{
		if (&a != this)
		{
			data.resize(a.data.size());
			copy(a.data.cbegin(), a.data.cend(), data.begin());
		}
		return *this;
	}

	train_data& operator= (train_data &&a) noexcept
	{
		if (&a != this)
		{
			data = move(a.data);
			a.data.clear();
		}
		return *this;
	}

	train_data get_part(const size_t &size) 
	{
		if (size >= data.size() || size == 0)
			return train_data(*this);

		if (size == 1)
			return get_one();

		return get_data_n(size);
	}

	train_data get_first_n(const size_t& size)
	{
		if (size >= data.size())
			return train_data(*this);

		vector<shared_ptr<const one_train_data>> for_new_train_data;
		for_new_train_data.resize(size);
		copy(data.begin(), data.begin() + size, for_new_train_data.begin());
		return train_data(for_new_train_data);
	}

	train_data get_part_for_test(const size_t& size)
	{
		if (size == 0)
			return train_data();

		if (size >= data.size())
			return train_data(*this);

		return get_data_n(size);
	}

	size_t size() const
	{
		return data.size();
	}

	void save(const string &name_file) const
	{
		ofstream file(name_file);
		
		file << data.size() << endl;
		file << data[0]->input.size() << endl;
		file << data[0]->out.size() << endl;

		for (size_t i = 0; i < data.size(); ++i)
			data[i]->save(file);

		file.close();
	}

	const shared_ptr<const one_train_data> operator[] (const size_t &i) const
	{
		return data[i];
	}
	
	const one_train_data get(const size_t& i) const //for python
	{
		return *(data[i]);
	}

	void add_data(const train_data &a)
	{
		data.resize(data.size() + a.size());
		copy(a.data.cbegin(), a.data.cend(), data.begin() + data.size() - a.data.size());
	}

	void add_data(const one_train_data &a)
	{
		data.push_back(make_shared <const one_train_data>(a));
	}

	void add_data(const std::vector<double> &input, const std::vector<double> &out)
	{
		data.push_back(make_shared <const one_train_data>(input, out));
	}

	void add_data(const std::vector<std::vector<double>> &input, const std::vector<std::vector<double>> &out)
	{
		data.reserve(data.size() + input.size());
		for (size_t i = 0; i < input.size(); ++i)
			data.push_back(make_shared <const one_train_data>(input[i], out[i]));
	}

	void shrink_to_fit()
	{
		data.shrink_to_fit();
	}

	void reserve(size_t size)
	{
		data.reserve(size);
	}

private:

	train_data(const shared_ptr<const one_train_data>& one)
	{
		data.push_back(one);
	}

	train_data get_one() const
	{
		random_device rd;
		std::mt19937 gen(rd());
		uniform_int_distribution<> urd(0, data.size() - 1);
		return train_data(data[urd(gen)]);
	}

	train_data get_data_n(const size_t &n)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		shuffle(data.begin(), data.end(), gen);
		//random_shuffle(data.begin(), data.end());
		vector<shared_ptr<const one_train_data>> for_new_train_data;
		for_new_train_data.resize(n);
		copy(data.end() - n, data.end(), for_new_train_data.begin());
		return train_data(for_new_train_data);
	}

	vector<shared_ptr<const one_train_data>> data;

};