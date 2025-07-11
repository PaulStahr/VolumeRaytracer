#ifndef IO_UTIL_H
#define IO_UTIL_H

#include <ostream>
#include <string>
#include "util.h"

std::string get_next_free_filenumber(std::string const & filename, std::string const & suffix);

std::string get_next_free_filename(std::string const & folder, std::string const & filename, std::string const & suffix);

template <class InputIt>
std::ostream & print_elements(std::ostream & out, InputIt begin, InputIt end)
{
    while (begin != end)
    {
        out << *begin;
        ++begin;
    }
    return out;
}

template <typename V>
struct print_as_struct
{
    print_as_struct(){}

    template <typename T>
    std::ostream & operator()(std::ostream & out, T const & elem) const
    {
        return out << static_cast<V>(elem);
    }
};

template <typename V>
static const print_as_struct<V> print_as;

template <typename F>
struct print_convert_struct
{
    F _func;
    print_convert_struct(F func_) : _func(func_){};
    
    template <typename T>
    std::ostream & operator()(std::ostream & out, T const & elem) const
    {
        return out << _func(elem);
    }
};

template <typename F>
print_convert_struct<F> print_convert(F func)
{
    return print_convert_struct<F>(func);
}

struct print_struct
{
    print_struct();
    
    template <typename T>
    std::ostream & operator()(std::ostream & out, T const & elem) const
    {
        return out << elem;
    }
};

static const print_struct print;

struct printer_struct
{
    printer_struct();
    template <typename T>
    std::ostream & operator()(std::ostream & out, T const & elem) const
    {
        return out << elem;
    }
};

static const printer_struct printer;

struct print_dummy_t
{
    print_dummy_t(){}
    template <typename T>
    std::ostream & operator()(std::ostream & out, T const &)
    {
        return out;
    }
};
//inline print_dummy_t::print_dummy_t() = default;

static const print_dummy_t print_dummy;

template <typename InputIter, typename PrintFunction>
struct print_iter_element
{
    InputIter _iter;
    PrintFunction _print;
    print_iter_element(InputIter iter_, PrintFunction print_) : _iter(iter_), _print(print_){}
    std::ostream & operator()(std::ostream & out, size_t index){return _print(out, _iter[index]);}
};

template <typename InputIter>
print_iter_element<InputIter, print_struct> get_print_element_func(InputIter iter)
{
    return print_iter_element<InputIter, print_struct>(iter, print);
}

template <typename InputIter, typename PrintFunction>
print_iter_element<InputIter, PrintFunction> get_print_element_func(InputIter iter, PrintFunction print)
{
    return print_iter_element<InputIter, PrintFunction>(iter, print);
}

template <class InputIt, class Seperator, class PrintFunction>
std::ostream & print_elements(std::ostream & out, InputIt begin, InputIt end, Seperator const & seperator, PrintFunction func)
{
    if (begin != end)
    {
        func(out,*begin);
        while (++begin != end)
        {
            func(out << seperator,*begin);
        }
    }
    return out;
}

template <class Container, class Seperator, class PrintFunction>
std::ostream & print_elements(std::ostream & out, Container const & cont, Seperator const & seperator, PrintFunction func)
{
    return print_elements(out, cont.cbegin(), cont.cend(), seperator, func);
}

template <class InputIt, class Seperator>
std::ostream & print_elements(std::ostream & out, InputIt begin, InputIt end, Seperator const & seperator)
{
    typedef typename std::iterator_traits<InputIt>::value_type value_t;
    return print_elements(out, begin, end, seperator, [](std::ostream & outl, value_t const & val) -> std::ostream &{return outl << val;});
}

template <class Container, class Seperator>
std::ostream & print_elements(std::ostream & out, Container const & cont, Seperator const & seperator)
{
    return print_elements(out, cont.cbegin(), cont.cend(), seperator);
}

template <typename Container>
std::ostream & print_matrix(std::ostream & out, Container const & vec)
{
    for (size_t i = 0; i < vec.size(); ++i)
    {
        print_elements(out, vec[i].begin(), vec[i].end(), ' ')<<std::endl;
    }
    return out;
}

template <typename InputIter>
std::ostream & print_matrix(std::ostream & out, pair_id_injection const & pair_id, InputIter iter)
{
    size_t num_elems = pair_id._num_elements;
    for (size_t i = 0; i < num_elems; ++i)
    {
        auto row = pair_id.get_row(i);
        for (size_t j = 0; j < num_elems; ++j)
        {
            if (j != 0)
            {
                out << ' ';
            }
            if (i == j)
            {
                out << '0';
            }
            else
            {
                out << iter[row[j]];
            }
        }
        out << std::endl;
    }
    return out;
}

template <typename InputIter>
std::ostream & print_matrix(std::ostream & out, size_t width, size_t height, InputIter iter)
{
    for (size_t i = 0; i < height; ++i)
    {
        for (size_t j = 0; j < width; ++j)
        {
            if (j != 0)
            {
                out << ' ';
            }
            out << *iter;
            ++iter;
        }
        out << std::endl;
    }
    return out;
}

template <uint64_t divisor>
struct print_div_struct
{
    print_div_struct(){}
    
    template <typename T>
    std::ostream & operator ()(std::ostream & out, T elem) const
    {
        return out << static_cast<double>(elem) / divisor;
    }
};

template <uint64_t divisor>
static const print_div_struct<divisor> print_div;

namespace IO_UTIL
{
template <typename T>
struct string_to_struct
{
    using argument_type = std::string;
    using result_type = T;

    string_to_struct() {}

    T operator()(std::string const& str) const;
};

template<typename T>
T string_to_struct<T>::operator()(const std::string& str) const
{
    T erg;
    std::stringstream ss(str);
    if (!(ss >> erg))
    {
        std::stringstream out;
        out << "\"" + str + "\" not castable to " << typeid(T).name();
        throw std::invalid_argument(out.str());
    }
    return erg;
}

template <>
struct string_to_struct<bool>
{
    using argument_type = std::string;
    using result_type = bool;

    string_to_struct() {}

    bool operator()(std::string const& str) const;
};

template <typename T>
extern const string_to_struct<T> string_to = string_to_struct<T>();


}

#endif
