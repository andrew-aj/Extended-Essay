using namespace std;

class activationFunctions
{
public:

    //Activation Functions
    double iden(double input);
    double binstep(double input);
    double logistic(double input);
    double tanh(double input);
    double arctan(double input);
    double rectified(double input);
    double parametric(double input);
    double exponential(double input);
    double softplus(double inptu);

    //Derivatives of the activation functions
    double idender(double input);
    double binder(double input);
    double logder(double input);
    double tanhder(double input);
    double atander(double input);
    double rectder(double input);
    double parader(double input);
    double exponder(double input);
    double softder(double input);
};
