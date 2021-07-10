

#include "h_globvar.hpp"
#include "h_simulate_PSV.hpp"
#include <chrono>
using namespace std::chrono;

int main()
{

    auto start = high_resolution_clock::now();
    simulate_PSV();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    // To get the value of duration use the count()
    // member function on the duration object
    std::cout << "TOTAL TIME == "<<duration.count()/1000 << "mili secs \n";
}
