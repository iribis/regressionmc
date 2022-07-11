#pragma once

#include <cmath>

class ErrorSingleDimensionStandard {
    float norm(float f) const { return std::abs(f); }
    double norm(double f) const { return std::abs(f); }
    template<typename V>
    auto norm(const V& v) const {
        auto i = v.begin();
        auto s = norm(*i); ++i;
        while (i!=v.end()) {
            s += norm(*i); ++i;
        }
        return s;
    }

public:
    template<typename R>
    auto operator()(const R& region) const {
        auto max_err = norm(region.error(0)); std::size_t max_dim = 0; 
        decltype(max_err) err;
        for (std::size_t d = 1; d<R::dimensions; ++d) {
            err = norm(region.error(d));
            if (err>max_err) {
                max_err = err; max_dim = d;
            }
        }
        return std::make_tuple(max_err,max_dim);
    }
};

inline ErrorSingleDimensionStandard error_single_dimension_standard() { return ErrorSingleDimensionStandard(); }

class ErrorSingleDimensionSize {
    float norm(float f) const { return std::abs(f); }
    double norm(double f) const { return std::abs(f); }
    template<typename V>
    auto norm(const V& v) const {
        auto i = v.begin();
        auto s = norm(*i); ++i;
        while (i!=v.end()) {
            s += norm(*i); ++i;
        }
        return s;
    }

    double size_factor;

public:
    // This function found the axis that produce the most error
    // and return max_err value and the dimension
    template<typename R>
    auto operator()(const R& region) const {
        // Compute the error for the different dimensions
        auto max_err = norm(region.error(0)); 
        std::size_t max_dim = 0;
        max_err += std::abs(size_factor*(region.range().max(0) - region.range().min(0))); 
        decltype(max_err) err;
        for (std::size_t d = 1; d<R::dimensions; ++d) {
            err = norm(region.error(d));
            err += std::abs(size_factor*(region.range().max(d) - region.range().min(d)));
            if (err>max_err) {
                max_err = err; max_dim = d;
            }
        }
        return std::make_tuple(max_err,max_dim);
    }

    ErrorSingleDimensionSize(double sf) : size_factor(sf) { }
};

inline ErrorSingleDimensionSize error_single_dimension_size(double size_factor = 0.01) { return ErrorSingleDimensionSize(size_factor); }

class ErrorRelativeSingleDimensionSize {
    float norm(float f) const { return std::abs(f); }
    double norm(double f) const { return std::abs(f); }
    template<typename V>
    auto norm(const V& v) const {
        auto i = v.begin();
        auto s = norm(*i); ++i;
        while (i!=v.end()) {
            s += norm(*i); ++i;
        }
        return s;
    }

    double size_factor;
    double offset;

public:
    template<typename R>
    auto operator()(const R& region) const {
        auto max_err = norm(region.error(0)); std::size_t max_dim = 0;
        double den = std::max(norm(region.integral()),offset);
        max_err /= den;
        max_err += std::abs(size_factor*(region.range().max(0) - region.range().min(0)));
        decltype(max_err) err;
        for (std::size_t d = 1; d<R::dimensions; ++d) {
            err = norm(region.error(d))/den;
            err += std::abs(size_factor*(region.range().max(d) - region.range().min(d)));
            if (err>max_err) {
                max_err = err; max_dim = d;
            }
        }
        return std::make_tuple(max_err,max_dim);
    }

    ErrorRelativeSingleDimensionSize(double sf, double offset) : size_factor(sf),offset(offset) { }
};

inline ErrorRelativeSingleDimensionSize error_relative_single_dimension_size(double size_factor = 1.e-5, double offset = 1.e-6) { return ErrorRelativeSingleDimensionSize(size_factor, offset); }

class ErrorPartiallyRelativeSingleDimensionSize {
    float norm(float f) const { return std::abs(f); }
    double norm(double f) const { return std::abs(f); }
    template<typename V>
    auto norm(const V& v) const {
        auto i = v.begin();
        auto s = norm(*i); ++i;
        while (i!=v.end()) {
            s += norm(*i); ++i;
        }
        return s;
    }

    double size_factor;
    std::size_t relative_dimensions;
    double offset;

public:
    template<typename R>
    auto operator()(const R& region) const {
        auto max_err = norm(region.error(0)); std::size_t max_dim = 0;
        double den = std::max(norm(region.integral()),offset);
        max_err /= den;
        max_err *= region.range().volume();
        max_err += std::abs(size_factor*(region.range().max(0) - region.range().min(0)));
        decltype(max_err) err;
        for (std::size_t d = 1; d<R::dimensions; ++d) {
            err = norm(region.error(d));
            if (d<relative_dimensions) {
                err/=den;
                err *= region.range().volume();
            }
            err += std::abs(size_factor*(region.range().max(d) - region.range().min(d)));
            if (err>max_err) {
                max_err = err; max_dim = d;
            }
        }
        return std::make_tuple(max_err,max_dim);
    }

    ErrorPartiallyRelativeSingleDimensionSize(double sf, std::size_t relative_dimensions, double offset) : size_factor(sf),relative_dimensions(relative_dimensions),offset(offset) { }
};

inline ErrorPartiallyRelativeSingleDimensionSize error_partially_relative_single_dimension_size(double size_factor = 1.e-5, std::size_t relative_dimensions = 2, double offset = 1.e-6) { return ErrorPartiallyRelativeSingleDimensionSize(size_factor, relative_dimensions, offset); }




class ErrorSingleDimensionSquare {
    float norm(float f) const { return f*f; }
    double norm(double f) const { return f*f; }
    template<typename V>
    auto norm(const V& v) const {
        auto i = v.begin();
        auto s = norm(*i); ++i;
        while (i!=v.end()) {
            s += norm(*i); ++i;
        }
        return s;
    }

    double size_factor;

public:
    template<typename R>
    auto operator()(const R& region) const {
        auto max_err = norm(region.error(0)); std::size_t max_dim = 0;
        max_err += std::abs(size_factor*(region.range().max(0) - region.range().min(0))); 
        decltype(max_err) err;
        for (std::size_t d = 1; d<R::dimensions; ++d) {
            err = norm(region.error(d));
            err += std::abs(size_factor*(region.range().max(d) - region.range().min(d)));
            if (err>max_err) {
                max_err = err; max_dim = d;
            }
        }
        return std::make_tuple(max_err,max_dim);
    }

    ErrorSingleDimensionSquare(double sf) : size_factor(sf) { }
};

inline ErrorSingleDimensionSquare error_single_dimension_square(double size_factor = 0.01) { return ErrorSingleDimensionSquare(size_factor); }


