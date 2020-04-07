#include "io_util.h"

namespace IO_UTIL
{
bool string_to_struct< bool >::operator()(std::string const & str) const
{
    if (str == "true")
        return true;
    if (str == "false")
        return false;
    bool erg;
    std::stringstream ss(str);
    if (!(ss >> erg))
    {
        std::stringstream out;
        out << "\"" + str + "\" not castable to " << typeid(bool).name();
        throw std::invalid_argument(out.str());
    }
    return erg;
}
}


//print_as_struct::print_as_struct(){}

print_struct::print_struct(){}

printer_struct::printer_struct(){}
