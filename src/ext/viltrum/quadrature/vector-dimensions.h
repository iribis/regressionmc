#pragma once

#include <vector>
#include <array>

// T: Type
// DIMBINS: Not sure what it is
template<typename T, std::size_t DIMBINS>
class vector_dimensions {
    std::vector<T> data; // to store the data
    std::array<std::size_t, DIMBINS> res; // resolution of each dimenions?
public:
    // Ok, there is a resolution here
    const std::array<std::size_t, DIMBINS>& resolution() const { return res; }
    std::size_t resolution(std::size_t i) const { return resolution()[i]; }

private:
    // p: might be the position in term of ... something?
    std::size_t position(const std::array<std::size_t,DIMBINS>& p) const {
        std::size_t pos = 0; // 
        std::size_t prod = 1; // Accumulate the resolutions 
        for (std::size_t d = 0;d<DIMBINS; ++d) {
            pos += p[d]*prod; 
            prod*=resolution(d);
        }
        return pos;
    }
public:
    // Allocate the resolution
    vector_dimensions(const std::array<std::size_t, DIMBINS>& r, const T& t = T()) : res(r) {
        std::size_t elements(1);
        for (auto r : res) elements*=r;
        data.resize(elements, t);
    }    
    vector_dimensions(const std::array<std::size_t, DIMBINS>& r, const std::vector<T>& e) : res(r),data(e) {
        std::size_t elements(1);
        for (auto r : res) elements*=r;
        data.resize(elements);
    }
    
    const std::vector<T>& raw_data() const { return data; } 

    T& operator[](const std::array<std::size_t,DIMBINS>& p) {
        return data[position(p)];
    }

    T& operator()(const std::array<std::size_t,DIMBINS>& p) {
        return (*this)[p];
    }

    const T& operator[](const std::array<std::size_t,DIMBINS>& p) const {
        return data[position(p)];
    }

    const T& operator()(const std::array<std::size_t,DIMBINS>& p) const {
        return (*this)[p];
    }


    std::size_t size() const { return raw_data().size(); }
};

