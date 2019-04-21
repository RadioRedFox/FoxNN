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

class memory_layer
{
public:
	memory_layer() {}

	memory_layer(const size_t &N_in)
	{
		enter.resize(N_in);
	}

	vector <vector <double>> enter;
};

class derivative
{
public:
	derivative() {}

	derivative(const size_t &N_n, const size_t &N_w)
	{
		d.resize(N_n);
		for (size_t i = 0; i < N_n; i++)
			d[i].resize(N_w + 1, 0);
	}

	void init(const size_t &N_n, const size_t &N_w)
	{
		d.resize(N_n);
		for_each(d.begin(), d.end(), [&](vector<double> &v) {v.resize(N_w + 1); });
	}

	void zero()
	{
		for_each(d.begin(), d.end(), [](vector<double> &v) {fill(v.begin(), v.end(), 0.0); }); //all = 0
	}
	vector <vector <double>> d;
};

class one_train_data
{
public:
	one_train_data() {}

	one_train_data(const one_train_data &a)
	{
		in.resize(a.in.size());
		copy(a.in.cbegin(), a.in.cend(), in.begin());
		out.resize(a.out.size());
		copy(a.out.cbegin(), a.out.cend(), out.begin());
		in.shrink_to_fit();
		out.shrink_to_fit();
	}

	one_train_data(one_train_data &&a) noexcept
	{
		in = move(a.in);
		out = move(a.out);
		a.in.clear();
		a.out.clear();
		in.shrink_to_fit();
		out.shrink_to_fit();
	}

	one_train_data& operator= (const one_train_data &a)
	{
		if (this != &a)
		{
			in.resize(a.in.size());
			copy(a.in.cbegin(), a.in.cend(), in.begin());
			out.resize(a.out.size());
			copy(a.out.cbegin(), a.out.cend(), out.begin());
			in.shrink_to_fit();
			out.shrink_to_fit();
		}
		return *this;
	}

	one_train_data& operator= (one_train_data &&a) noexcept
	{
		if (this != &a)
		{
			in = move(a.in);
			out = move(a.out);
			a.in.clear();
			a.out.clear();
			in.shrink_to_fit();
			out.shrink_to_fit();
		}
		return *this;
	}

	one_train_data(const vector<double> &in_, const vector<double> &out_)
	{
		in.resize(in_.size());
		copy(in_.cbegin(), in_.cend(), in.begin());
		out.resize(out_.size());
		copy(out_.cbegin(), out_.cend(), out.begin());
		in.shrink_to_fit();
		out.shrink_to_fit();
	}

	vector<double> in;
	vector<double> out;

	friend class train_data;

private:
	one_train_data(ifstream &open_file, const size_t &N_in, const size_t &N_out)
	{
		in.resize(N_in);
		in.shrink_to_fit();
		out.resize(N_out);
		out.shrink_to_fit();
		for (size_t j = 0; j < N_in; ++j)
			open_file >> in[j];

		for (size_t j = 0; j < N_out; ++j)
			open_file >> out[j];
	}

	void save(ofstream  &open_file) const
	{
		for (size_t j = 0; j < in.size(); ++j)
			open_file << in[j] << endl;

		for (size_t j = 0; j < out.size(); ++j)
			open_file << out[j] << endl;
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

		size_t N_test, N_in, N_out;

		file >> N_test;
		file >> N_in;
		file >> N_out;

		data.reserve(N_test);

		for (size_t i = 0; i < N_test; ++i)
			data.push_back(shared_ptr <const one_train_data> (new one_train_data(file, N_in, N_out)));
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

	train_data(const train_data &a, const int &n)
	{
		data.resize(n);
		copy(a.data.cbegin(), a.data.cbegin() + n, data.begin());
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

	size_t size() const
	{
		return data.size();
	}

	void save(const string &name_file) const
	{
		ofstream file(name_file);
		
		file << data.size() << endl;
		file << data[0]->in.size() << endl;
		file << data[0]->out.size() << endl;

		for (size_t i = 0; i < data.size(); ++i)
			data[i]->save(file);

		file.close();
	}

	const shared_ptr<const one_train_data> operator[] (const size_t &i) const
	{
		return data[i];
	}
	
	void add_data(const train_data &a)
	{
		data.resize(data.size() + a.size());
		copy(a.data.cbegin(), a.data.cend(), data.begin() + data.size() - a.data.size());
	}

	void add_data(const one_train_data &a)
	{
		data.push_back(shared_ptr <const one_train_data>(new one_train_data(a)));
	}

	void add_data(const vector<double>& in_, const vector<double>& out_)
	{
		data.push_back(shared_ptr <const one_train_data>(new one_train_data(in_, out_)));
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
		random_shuffle(data.begin(), data.end());
		return train_data(*this, n);
	}

	vector<shared_ptr<const one_train_data>> data;

};