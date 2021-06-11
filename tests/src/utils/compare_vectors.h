#ifndef COMPARE_VECTORS_H
#define COMPARE_VECTORS_H

#include <gtest/gtest.h>
#include <vector>

template<typename T>
void compare_vectors(size_t n, const T* ptr1, const T* ptr2) {
    EXPECT_EQ(std::vector<T>(ptr1, ptr1+n), std::vector<T>(ptr2, ptr2+n));
}

template<typename T>
void compare_vectors(size_t n, const T* ptr1, const std::vector<T>& ref) {
    EXPECT_EQ(std::vector<T>(ptr1, ptr1+n), ref);
}

template<typename T>
void compare_vectors(const std::vector<T>& ref, size_t n, const T* ptr1) {
    EXPECT_EQ(std::vector<T>(ptr1, ptr1+n), ref);
}

#endif
