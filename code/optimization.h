#pragma once

#include <math.h> 
#include "settings.h"

using namespace std;

class base_class_optimization
{
public:
	base_class_optimization() : derivative(0) {}

	virtual void correction_of_scales(double &w, const double& speed, const Settings& settings) = 0;

	double derivative;
};

class Adam_optimization : public base_class_optimization
{
public:
	Adam_optimization() : base_class_optimization(), m(0), v(0) {}

	void correction_of_scales(double& w, const double& speed, const Settings& settings)
	{
		const double& betta_1 = settings.settings_optimization.adam.betta_1;
		const double& betta_2 = settings.settings_optimization.adam.betta_2;
		const double& epsilon = settings.settings_optimization.adam.epsilon;
		const double& t = settings.settings_optimization.adam.step;

		m = betta_1 * m + (1 - betta_1) * derivative;
		v = betta_2 * v + (1 - betta_2) * derivative * derivative;

		const double mh = m / (1 - pow(betta_1, t));
		const double vh = v / (1 - pow(betta_2, t));

		double res = speed * mh / sqrt(epsilon + vh);

		if (t == settings.settings_optimization.adam.step_to_zero_after_n)
		{
			v = 0;
			m = 0;
		}

		if (res != res)
			w -= speed * derivative;
		else
			w -= res;
	}

private:
	double m;
	double v;
};

class Nesterov_optimization : public base_class_optimization
{
public:
	Nesterov_optimization() : base_class_optimization(), v(0) {}

	void correction_of_scales(double& w, const double& speed, const Settings& settings)
	{
		v = settings.settings_optimization.nesterov.gamma * v + speed * derivative;
		w -= v;
	}

	double v;
};

class SGD : public base_class_optimization
{
public:
	SGD() : base_class_optimization() {}
	void correction_of_scales(double& w, const double& speed, const Settings& settings)
	{
		w -= speed * derivative;
	}
};

shared_ptr<base_class_optimization> get_optimization(const string &mode)
{
	if (mode == "Adam")
		return make_shared<Adam_optimization>();
	if (mode == "Nesterov")
		return make_shared<Adam_optimization>();

	return make_shared<SGD>();
}
