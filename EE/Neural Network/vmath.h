#pragma once
#include <vector>
using namespace std;

class vmath
{
public:
    vector<double> vadd(vector<double> v1, vector<double> v2);
    vector<double> vsub(vector<double> v1, vector<double> v2);
    vector<double> vmul(vector<double> v1, vector<double> v2);
    vector<double> vdiv(vector<double> v1, vector<double> v2);
    vector<double> vpow(vector<double> v1, vector<double> v2);
    double vsum(vector<double> vector);
    double vdot(vector<double> v1, vector<double> v2);
};