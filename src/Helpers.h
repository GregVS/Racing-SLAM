#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <string>

template <typename T> static T time_it(const std::string& name, std::function<T()> func)
{
    auto start = std::chrono::high_resolution_clock::now();
    T result = func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << name << " took " << duration.count() * 1000 << " ms" << std::endl;
    return result;
}

static void time_it(const std::string& name, std::function<void()> func)
{
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << name << " took " << duration.count() * 1000 << " ms" << std::endl;
}
