#include <stdexcept>

class lapack_exception: public std::runtime_error
{
    public:
        lapack_exception(std::string const& msg):
            std::runtime_error(msg)
        {}
};

void eliminate(double * matrix, double * rhs, int n, int m);