#include "./vertical.hpp"

template <typename T>
inline VerticalList<T>::~VerticalList()
{
    for (SFBase*& el : elements) delete el;
}

template <typename T>
inline VerticalList<T> VerticalList<T>::addElement(T *element)
{
    elements.push_back(element);
    return this;
}
