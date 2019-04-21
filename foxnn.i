%module foxnn

%{
#include "foxnn.h"
#include "additional_memory.h"
%}

%include "std_string.i"
%include "std_vector.i"

using namespace std;
typedef unsigned int size_t;
%template(IntVector) vector<int>;
%template(DoubleVector) vector<double>;
%include foxnn.h
%include additional_memory.h
%include settings.h















