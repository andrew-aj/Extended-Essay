#include "activationFunctions.h"
//#include <iostream>
#include <cmath>
using namespace std;

double alpha = 0.1;

//Activation Functions
double activationFunctions::iden(double input)
{

    return input;

}

double activationFunctions::binstep(double input)
{

    double a = 0.0;
    (input < 0) ? a = 0 : a = 1;
    return a;

}

double activationFunctions::logistic(double input)
{

    double a = (1 / (1 + exp(-input)));
    return a;

}

double activationFunctions::tanh(double input)
{

    double a = (2 / (1 + exp(-2 * (input))));
    return a - 1;

}

double activationFunctions::arctan(double input)
{

    return atan(input);

}

double activationFunctions::rectified(double input)
{

    double a = 0.0;
    (input < 0) ? a = 0 : a = input;
    return a;

}

double activationFunctions::parametric(double input)
{

    double a = 0.0;
    (input < 0) ? a = alpha * input : a = input;
    return a;

}

double activationFunctions::exponential(double input)
{

    double a = alpha * (exp(input) - 1);
    if (input < 0) return a;
    else return input;

}

double activationFunctions::softplus(double input)
{

    return (log1p(exp(input)));

}

//Derivatives of the activation functions

double activationFunctions::idender(double input)
{

    return 1;

}

double activationFunctions::binder(double input)
{

    if (input != 0) return 0;

}

double activationFunctions::logder(double input)
{

    double a = (logistic(input) * (1 - logistic(input)));
    return a;

}

double activationFunctions::tanhder(double input)
{

    return (1 - pow(tanh(input), 2.0));

}

double activationFunctions::atander(double input)
{

    return (1 / (pow(input, 2) + 1));

}

double activationFunctions::rectder(double input)
{

    if (input < 0) return 0;
    else return 1;

}

double activationFunctions::parader(double input)
{

    if (input < 0) return alpha;
    else return 1;

}

double activationFunctions::exponder(double input)
{

    if (input < 0) return (exponential(input) + alpha);
    else return 1;

}

double activationFunctions::softder(double input)
{

    return (1 / (1 + exp(-input)));

}