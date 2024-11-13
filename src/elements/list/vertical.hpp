#pragma once

#include "elements/base.hpp"

template <typename T>
class VerticalList : SFBase {
    static_assert(
        std::enable_if<std::is_base_of<SFBase, T>::value>::value,
        "T must be derived from SFBase"
    );
    std::vector<T*> elements;

public:
    ~VerticalList();
    VerticalList<T> addElement(T*);
};
