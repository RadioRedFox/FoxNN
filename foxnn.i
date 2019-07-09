%module foxnn

%{
#include "foxnn.h"
#include "layer.h"
#include "train_data.h"
#include "settings.h"
%}

%include "std_string.i"
%include "std_vector.i"

namespace std {
typedef unsigned int size_t;
%template(IntVector) vector<int>;
%template(DoubleVector) vector<double>;
%template(DoubleVVector) vector<vector<double>>;
}
%include foxnn.h
%include layer.h
%include train_data.h
%include settings.h















