#ifndef VECTOR_TO_POINTERS
#define VECTOR_TO_POINTERS

#include <vector>

namespace scran {

template<typename T>
inline std::vector<T*> vector_to_pointers(std::vector<std::vector<T> >& input) {
    std::vector<T*> output(input.size());
    auto oIt = output.begin();
    for (auto& i : input) {
        *oIt = i.data();
        ++oIt;
    }
    return output;
}

template<typename T>
inline std::vector<const T*> vector_to_pointers(const std::vector<std::vector<T> >& input) {
    std::vector<const T*> output(input.size());
    auto oIt = output.begin();
    for (auto& i : input) {
        *oIt = i.data();
        ++oIt;
    }
    return output;
}

}

#endif
