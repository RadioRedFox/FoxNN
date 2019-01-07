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

using namespace std;

bool f_abs_sort(const double &a, const double &b)
{
	return (fabs(a) <= fabs(b));
}

class neuron
{
public:
	neuron(const int &N)
	{
		mt19937 gen(clock());
		uniform_real_distribution<> urd(-1, 1);
		w.resize(N + 1);
		for (int i = 0; i < w.size(); i++)
			w[i] = urd(gen);
	}

	neuron(vector <double> &new_w)
	{
		w = move(new_w);
	}

	neuron(const neuron &a)
	{
		w.resize(a.w.size());
		copy(a.w.begin(), a.w.end(), w.begin());
	}

	double res_function(const double &x) const
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

	double derivate_res_function(const double &x) const
	{
		double d_res = exp(-x) / ((1.0 + exp(-x))*(1.0 + exp(-x)));
		if (d_res != d_res)
			return 0.0000000001;
		return d_res;
	}

	double get_d_out(const vector <double> &enter) const
	{
		double sum = 0;
		vector <double> for_sum;
		for_sum.resize(w.size());
		for (int i = 0; i < w.size() - 1; ++i)
			for_sum[i] = w[i] * enter[i];

		for_sum[w.size() - 1] = -w[w.size() - 1];

		sort(for_sum.begin(), for_sum.end(), f_abs_sort);

		for (int i = 0; i < for_sum.size(); ++i)
			sum += for_sum[i];

		double derivate = derivate_res_function(sum);
		return derivate;
	}

	void derivate_w(const double &error, const vector <double> &enter, vector <double> &d) const
	{
		double sum = 0;
		double d_res_function;
		vector <double> for_sum;

		for_sum.resize(w.size());
		for (int i = 0; i < w.size() - 1; ++i)
			for_sum[i] = w[i] * enter[i];
		for_sum[w.size() - 1] = -w[w.size() - 1];

		sort(for_sum.begin(), for_sum.end(), f_abs_sort);
		for (int i = 0; i < for_sum.size(); ++i)
			sum += for_sum[i];

		d_res_function = derivate_res_function(sum);

		for_sum.~vector();
		for (size_t i = 0; i < w.size() - 1; ++i)
		{
#pragma omp atomic
			d[i] += enter[i] * d_res_function * error;
		}

#pragma omp atomic
		d[d.size() - 1] += -d_res_function * error;

		return;
	}

	void correction_of_scales(const vector <double> &d)
	{
		for (int i = 0; i < w.size(); i++)
			w[i] -= d[i];
		return;
	}

	double get_out(const vector <double> &enter) const
	{
		double sum = 0;
		vector <double> for_sum;
		for_sum.resize(w.size());
		for (int i = 0; i < w.size() - 1; i++)
			for_sum[i] = w[i] * enter[i];

		for_sum[w.size() - 1] = -w[w.size() - 1];

		sort(for_sum.begin(), for_sum.end(), f_abs_sort);

		for (int i = 0; i < for_sum.size(); i++)
			sum += for_sum[i];

		double res = res_function(sum);
		return res;
	}

	double get_w(const int &i) const
	{
		return w[i];
	}

	void get_w(vector <double> &ww) const
	{
		ww.resize(w.size());
		copy(w.begin(), w.end(), ww.begin());
		return;
	}

	int get_N(void) const
	{
		return w.size() - 1;
	}

	void mutation(const double &speed)
	{
		for (int i = 0; i < w.size(); i++)
		{
			std::mt19937 gen(clock());
			std::uniform_real_distribution<> urd(-1, 1);
			w[i] += urd(gen) * speed;
		}
	}

private:
	vector <double> w; //scales
};

class layer
{
public:
	layer(const int &N_in, const int &N_neurons) : neurons()
	{

		for (int i = 0; i < N_neurons; i++)
		{
			neuron *new_neuron = new neuron(N_in);
			neurons.push_back(move(new_neuron));
		}
	}

	layer(vector <vector <double>> &w)
	{
		for (int i = 0; i < w.size(); i++)
		{
			neuron *new_neuron = new neuron(w[i]);
			neurons.push_back(move(new_neuron));
		}
	}

	layer(const layer &a)
	{
		for (int i = 0; i < a.neurons.size(); i++)
		{
			neuron *new_neuron = new neuron(*(a.neurons[i]));
			neurons.push_back(new_neuron);
		}
	}

	void get_error(vector <double> &out_error, const vector <double> &error, const vector <double> enter) const
	{
		vector <double> d_out;
		vector <vector <double>> for_sum_error;
		d_out.resize(neurons.size());

		for_sum_error.resize(enter.size());
		for (int i = 0; i < for_sum_error.size(); i++)
			for_sum_error[i].resize(neurons.size());



		for (int i = 0; i < neurons.size(); i++)
			d_out[i] = neurons[i]->get_d_out(enter);

		out_error.resize(enter.size(), 0);

		for (int i = 0; i < enter.size(); i++)
			for (int j = 0; j < neurons.size(); j++)
				for_sum_error[i][j] = error[j] * d_out[j] * neurons[j]->get_w(i);

		for (int i = 0; i < enter.size(); i++)
			sort(for_sum_error[i].begin(), for_sum_error[i].end(), f_abs_sort);

		for (int i = 0; i < enter.size(); i++)
			for (int j = 0; j < neurons.size(); j++)
				out_error[i] += for_sum_error[i][j];

		return;
	}

	void back_running(vector <double> &error, const vector <double> &enter, vector <vector <double>> &d) const
	{
		vector <double> out_error;
		get_error(out_error, error, enter);

		for (int i = 0; i < neurons.size(); i++)
			neurons[i]->derivate_w(error[i], enter, d[i]);
		error = move(out_error);
	}

	void get_out(const vector <double> &enter, vector <double> &out) const
	{
		out.resize(neurons.size());
#pragma omp parallel for
		for (int i = 0; i < neurons.size(); i++)
			out[i] = neurons[i]->get_out(enter);
		return;
	}

	void correction_of_scales(const vector <vector <double>> &d)
	{
		for (int i = 0; i < neurons.size(); i++)
			neurons[i]->correction_of_scales(d[i]);
		return;
	}

	void get_all_w(vector <vector <double>> &w) const
	{
		w.resize(neurons.size());
		for (int i = 0; i < neurons.size(); i++)
			neurons[i]->get_w(w[i]);
		return;
	}

	int get_N_w(void) const
	{
		return neurons[0]->get_N();
	}

	int get_N_n(void) const
	{
		return neurons.size();
	}

	void mutation(const double &speed)
	{
		for (int i = 0; i < neurons.size(); i++)
			neurons[i]->mutation(speed);
		//  cout << neurons[0].get_w(0) << endl;
	}

	~layer()
	{
		for (int i = 0; i < neurons.size(); ++i)
			delete neurons[i];
	}

private:
	vector <neuron*> neurons;
};

class memory_layer
{
public:
	memory_layer() {}

	memory_layer(const int &N_in)
	{
		enter.resize(N_in);
	}

	vector <vector <double>> enter;
};

class derivative
{
public:
	derivative() {}

	derivative(const int &N_n, const int &N_w)
	{
		d.resize(N_n);
		for (int i = 0; i < N_n; i++)
			d[i].resize(N_w + 1, 0);
	}

	void init(const int &N_n, const int &N_w)
	{
		d.resize(N_n);
		for (int i = 0; i < N_n; i++)
			d[i].resize(N_w + 1, 0);
	}

	void zero()
	{
		for (int i = 0; i < d.size(); ++i)
			for (int j = 0; j < d[i].size(); ++j)
				d[i][j] = 0;
	}
	vector <vector <double>> d;
};

class neural_network
{
public:
	neural_network(void) {}

	neural_network(const string &name_file)
	{
		ifstream file(name_file);
		int N_layers, N_in, N_n;
		double next_w;
		vector <vector <double>> w;
		file >> N_layers;
		for (int i = 0; i < N_layers; i++)
		{
			file >> N_in;
			file >> N_n;
			w.resize(N_n);
			for (int j = 0; j < N_n; j++)
				for (int k = 0; k < N_in + 1; k++)
				{
					file >> next_w;
					w[j].push_back(next_w);
				}
			layer* next_layer = new layer(w);
			layers.push_back(move(next_layer));
		}
		file.close();
	}

	neural_network(const neural_network &a)
	{
		for (int i = 0; i < a.layers.size(); i++)
		{
			layer* next = new layer(*(a.layers[i]));
			layers.push_back(move(next));
		}
	}

	neural_network(const vector<int> &parametrs)
	{
		for (int i = 0; i < parametrs.size() - 1; i++)
		{
			layer *Layer = new layer(parametrs[i], parametrs[i + 1]);
			layers.push_back(move(Layer));
		}
	}

	void next_layers(layer &new_layer)
	{
		layers.push_back(move(&new_layer));
		return;
	}

	int get_out(const vector<double> &first_in, vector <double> &out) const
	{

		vector<double> enter;

		layers[0]->get_out(first_in, out);

		enter = move(out);

		for (int i = 1; i < layers.size(); i++)
		{
			layers[i]->get_out(enter, out);
			enter = move(out);
		}

		out = move(enter);

		return  distance(out.begin(), max_element(out.begin(), out.end()));;
	}

	double get_out(const vector<double> &first_in) const
	{
		vector<double> enter;
		vector<double> out;

		layers[0]->get_out(first_in, out);

		enter = move(out);

		for (int i = 1; i < layers.size(); i++)
		{
			layers[i]->get_out(enter, out);
			enter = move(out);
		}

		out = move(enter);
		return out[0];
	}

	double error_function(const vector <memory_layer> &memory_layers, const vector <vector <int>> &y) const
	{
		double res = 0;
		vector <double> for_sum;

		for_sum.reserve(y.size() * layers[layers.size() - 1]->get_N_n());

		for (int i = 0; i < y.size(); i++)
			for (int j = 0; j < layers[layers.size() - 1]->get_N_n(); j++)
			{
				for_sum.push_back((memory_layers[layers.size()].enter[i][j] - y[i][j]) * (memory_layers[layers.size()].enter[i][j] - y[i][j]));
			}

		sort(for_sum.begin(), for_sum.end(), f_abs_sort);


		for (int i = 0; i < for_sum.size(); i++)
			res += for_sum[i];

		res /= 2;
		return res;
	}

	double get_error_for_test(const vector <vector <double>> &test, const vector <vector <double>> &y, int &n_true, const double &error) const
	{

		double res = 0;
		n_true = 0;
		vector <double> out;
		for (int i = 0; i < y.size(); i++)
		{
			int need_max = 0;
			get_out(test[i], out);
			for (int j = 0; j < out.size(); j++)
			{
				res += fabs(out[j] - y[i][j]);
				if (fabs(out[j] - y[i][j]) < error)
					need_max++;
			}
			if (need_max == out.size())
				n_true++;
		}
		return res / y.size();
	}

	void error_last_layer(const memory_layer &last_layer, const vector <vector <double>> &y, vector <vector <double>> &error) const
	{
		error.resize(y.size());
		for (int i = 0; i < y.size(); i++)
			error[i].resize(y[0].size());

		for (int i = 0; i < y.size(); i++)
			for (int j = 0; j < y[0].size(); j++)
				error[i][j] = last_layer.enter[i][j] - y[i][j];
		return;
	}

	void init_memory_layers_zero(const vector <vector <double>> &test, vector <memory_layer> &memory_layers) const
	{
		for (int i = 0; i < test.size(); i++)
		{
			memory_layers[0].enter[i].resize(test[i].size());
			copy(test[i].begin(), test[i].end(), memory_layers[0].enter[i].begin());
		}
	}

	void forward_stroke(const int &test_size)
	{
#pragma omp parallel for
		for (int j = 0; j < test_size; j++)
			for (int i = 0; i < layers.size(); i++)
				layers[i]->get_out(memory_layers[i].enter[j], memory_layers[i + 1].enter[j]);
	}

	void create_memory_layer_and_derivativs(const int &test_size)
	{
		memory_layers.resize(layers.size() + 1);
		for (int i = 0; i < memory_layers.size(); ++i)
			memory_layers[i].enter.resize(test_size);

		derivativs.resize(layers.size());

		for (int i = 0; i < derivativs.size(); i++)
			derivativs[i].init(layers[i]->get_N_n(), layers[i]->get_N_w());
	}

	void init_memory_for_train(const vector <vector <double>> &test)
	{

		create_memory_layer_and_derivativs(test.size());

		init_memory_layers_zero(test, memory_layers);

		forward_stroke(test.size());

		return;
	}

	void parse_vector_y(vector <vector <int>> &yy, const vector <int> &y) const
	{
		const int N_in_last_layer = layers[layers.size() - 1]->get_N_n();

		yy.resize(y.size());
		for (int i = 0; i < yy.size(); i++)
		{
			yy[i].resize(N_in_last_layer, 0);
			yy[i][y[i]] = 1;
		}

		return;
	}

	void train(const vector <vector <double>> &test, const vector <vector <double>> &y, const double &speed)
	{

		vector <vector <double>> error;
		if (memory_layers.size() == 0)
		{
			init_memory_for_train(test);
		}
		else
		{
			for (int i = 0; i < derivativs.size(); ++i)
				derivativs[i].zero();
			forward_stroke(test.size());
		}

		error_last_layer(memory_layers[memory_layers.size() - 1], y, error);

#pragma omp parallel for 
		for (int i = 0; i < test.size(); i++)
			for (int j = layers.size() - 1; j >= 0; j--)
				layers[j]->back_running(error[i], memory_layers[j].enter[i], derivativs[j].d);

#pragma omp parallel for	
		for (int i = 0; i < layers.size(); i++)
			for (int j = 0; j < derivativs[i].d.size(); j++)
				for (int k = 0; k < derivativs[i].d[j].size(); k++)
					derivativs[i].d[j][k] *= speed;

#pragma omp parallel for
		for (int i = 0; i < layers.size(); i++)
			layers[i]->correction_of_scales(derivativs[i].d);

		return;
	}

	void read_data_train_file(const string &name_file, vector <vector <double>> &test, vector <vector <double>> &y)
	{
		ifstream file;
		file.exceptions(ifstream::badbit | ifstream::failbit);

		file.open(name_file);


		int N_test, N_in, N_out;

		file >> N_test;
		file >> N_in;
		file >> N_out;

		y.resize(N_test);

		test.resize(N_test);
		for (int i = 0; i < N_test; i++)
			test[i].resize(N_in);

		y.resize(N_test);
		for (int i = 0; i < N_test; i++)
			y[i].resize(N_out);

		for (int i = 0; i < N_test; i++)
		{
			for (int j = 0; j < N_in; j++)
				file >> test[i][j];

			for (int j = 0; j < N_out; j++)
				file >> y[i][j];
		}
		file.close();
	}

	void train_on_file(const string &name_file, const double &speed, const int &max_iteration, const double &min_error, const int &N_print)
	{

		vector <vector <double>> test;
		vector <vector <double>> y;

		try
		{
			read_data_train_file(name_file, test, y);
		}
		catch (const ifstream::failure &ex)
		{
			cout << ex.what() << endl;
			cout << "can't open file '" << name_file << "'" << endl;
			return;
		}
		catch (const exception &ex)
		{
			cout << ex.what() << endl;
			return;
		}


		int n_true;
		double er;

		for (int i = 1; i <= max_iteration; ++i)
		{
			train(test, y, speed);
			er = get_error_for_test(test, y, n_true, min_error);
			if (er < min_error || i == max_iteration)
			{
				cout << "iteration = " << i << " error = " << er << " n_true = " << n_true << endl;
				return;
			}
			if (i %  N_print == 0)
			{
				cout << "iteration = " << i << " error = " << er << " n_true = " << n_true << endl;
			}
		}

	}

	void save(const string &name_file) const
	{
		ofstream file(name_file);
		vector <vector <double>> w;
		file << layers.size() << endl;
		for (int i = 0; i < layers.size(); i++)
		{
			file << layers[i]->get_N_w() << " " << layers[i]->get_N_n() << endl;
			layers[i]->get_all_w(w);
			for (int j = 0; j < w.size(); j++)
				for (int k = 0; k < w[j].size(); k++)
					file << w[j][k] << endl;
		}
		file.close();
		return;
	}

	void mutation(const double &speed)
	{
#pragma omp parallel for
		for (int i = 0; i < layers.size(); i++)
			layers[i]->mutation(speed);
	}

	~neural_network()
	{
		for (int i = 0; i < layers.size(); ++i)
			delete layers[i];
	}

private:
	vector <memory_layer> memory_layers;
	vector <derivative> derivativs;
	vector <layer *> layers;
};




