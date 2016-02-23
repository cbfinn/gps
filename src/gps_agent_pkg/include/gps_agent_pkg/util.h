#include <sstream>
#include <string>
#include <vector>

namespace util
{
void split(const std::string &s, char delim, std::vector<std::string> &elems);
}

template <typename T>
std::string to_string(T value)
{
    //create an output string stream
    std::ostringstream os ;

    //throw the value into the string stream
    os << value ;

    //convert the string stream into a string and return
    return os.str() ;
}
