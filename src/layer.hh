#ifndef UDNN_LAYER_HH
#define UDNN_LAYER_HH

#include "tensor.hh"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <iostream>

enum class DType { Int8, Int16, Int32, Int64, Float, Double };

class LayerBase {
public:
  virtual TensorSize in_size() const = 0;
  virtual TensorSize out_size() const = 0;
  virtual DType in_type() const = 0;
  virtual DType out_type() const = 0;

  virtual const TensorBase *out_base() const = 0;

  std::string name;

  inline LayerBase *connect(LayerBase *next) {
    if (!next) {
      next_ = nullptr;
      return nullptr;
    }
    if (next->in_size() != out_size())
      throw std::invalid_argument(
          "Tensor dimension mismatch: " + next->in_size().str() + " -> " +
          out_size().str());
    if (next->in_type() != out_type())
      throw std::invalid_argument("Tensor type mismatch");
    next_ = next;
    return next;
  }
  inline LayerBase *next() const { return next_; }
  virtual void forward(void *data) = 0;

  ~LayerBase() = default;

private:
  LayerBase *next_ = nullptr;
};

template <typename T> class Layer : public LayerBase {
public:
  inline Layer(const TensorSize &in_size, const TensorSize &out_size)
      : in_size_(in_size), out_(out_size) {}

  inline Layer() = default;

  inline TensorSize in_size() const override { return in_size_; }
  inline TensorSize out_size() const override { return out_.size; }
  inline DType in_type() const override { return get_type<T>(); }
  inline DType out_type() const override { return get_type<T>(); }

  inline virtual const Tensor<T> &out() const { return out_; }
  inline const TensorBase *out_base() const override { return &out_; }

  virtual void forward(const Tensor<T> &input) = 0;

  // noop for layers that doesn't have weights
  inline virtual void load_weights(const Tensor<T> &) {}
  // first one is weights, second one is bias
  inline virtual void load_bias(const Tensor<T> &) {}
  inline virtual bool has_weights() const { return false; }
  inline virtual bool has_bias() const { return false; }
  inline virtual TensorSize weights_size() const { return {0, 0, 0, 0}; }
  inline virtual const Tensor<T> *get_weights() const { return nullptr; }
  inline virtual const Tensor<T> *get_bias() const { return nullptr; }
  inline virtual TensorSize weight_size() const { return {0, 0, 0, 0}; }
  inline virtual TensorSize bias_size() const { return {0, 0, 0, 0}; }

  inline void forward(void *data) override {
    // do not copy the data
    auto t = Tensor<T>(data, in_size(), TensorSize::default_stride(in_size()),
                       false);
    forward(t);
  }

protected:
  TensorSize in_size_;
  Tensor<T> out_;

private:
  template <typename V> inline static DType get_type() {
    static_assert(std::is_fundamental<V>(),
                  "Template type has to be numeric types");
    if (std::is_same<V, int8_t>())
      return DType::Int8;
    else if (std::is_same<V, int16_t>())
      return DType::Int16;
    else if (std::is_same<V, int32_t>())
      return DType::Int32;
    else if (std::is_same<V, int64_t>())
      return DType::Int64;
    else if (std::is_same<V, float>())
      return DType::Float;
    else if (std::is_same<V, double>())
      return DType::Double;
    else
      throw std::runtime_error("Unable to determine types");
  }
};

template <typename T> class Conv2DLayer : public Layer<T> {
public:
  uint32_t filter_size;
  uint32_t num_filters;

  inline Conv2DLayer(const TensorSize &in_size, uint32_t filter_size, uint32_t num_filters)
    : Layer<T>(in_size, {in_size.y - (filter_size-1), in_size.x - (filter_size-1), num_filters}),
      weights_(filter_size, filter_size, in_size.c, num_filters),
      bias_(1, 1, num_filters) {}

  inline void set_weight(uint32_t y, uint32_t x, uint32_t c, uint32_t k, T value) {
    weights_(y, x, c, k) = value;
  }

  inline void load_weights(const Tensor<T> &weight) override {
    weights_.load(weight);
  }

  inline TensorSize weights_size() const { return weights_.size; }

  inline void forward(const Tensor<T> &in) override {
    auto filter_dims = this->in_size_.c * weights_.size.y * weights_.size.x;
    auto in2 = typename Tensor<T>::vector_type(filter_dims);
    auto w = typename Tensor<T>::vector_type(filter_dims);
    auto prod = typename Tensor<T>::vector_type(filter_dims);

    for(int c = 0; c < this->out_.size.c; ++c) {
      for(int y = 0; y < this->out_.size.y; ++y) {
        for(int x = 0; x < this->out_.size.x; ++x) {
          int i = 0;
          for (int cc = 0; cc < this->in_size_.c; ++cc) {
            for(int yy = 0; yy < weights_.size.y; ++yy) {
              for(int xx = 0; xx < weights_.size.x; ++xx) {
                in2[i] = in(y + yy, x + xx, cc);
                w[i] = weights_(yy, xx, cc, c);
                i++;
              }
            }
          }

          xsimd::transform (in2.begin(), in2.end(), w.begin(), prod.begin(),
            [](auto const &a, auto const &b) { return a * b; });
          
          auto sum = xsimd::reduce (prod.begin(), prod.end(), (T) 0);
            
          this->out_(y, x, c) = sum + bias_(0, 0, c);
        }
      }
    }
  }

  inline TensorSize weight_size() const override { return weights_.size; }
  inline TensorSize bias_size() const override { return bias_.size; }
  inline virtual const Tensor<T> *get_weights() const { return &weights_; }
  inline virtual const Tensor<T> *get_bias() const { return &bias_; }

  inline bool has_bias() const override { return true; }
  inline bool has_weights() const override { return true; }
  inline void load_bias(const Tensor<T> &bias) override { bias_.load(bias); }

private:
  Tensor<T> weights_;
  Tensor<T> bias_;
};

template <typename T> class MaxPoolingLayer : public Layer<T> {
public:
  inline explicit MaxPoolingLayer(const TensorSize &in_size, uint32_t pool_size)
    : Layer<T>(in_size, {in_size.y / pool_size, in_size.x / pool_size, in_size.c}),
      pool_size_(pool_size) {}

  inline void forward(const Tensor<T> &in) override {
    const int max_size = pool_size_ * pool_size_;
    auto maxer = typename Tensor<T>::vector_type(max_size);

    for(int c = 0; c < this->out_.size.c; ++c) {
      for(int y = 0; y < this->out_.size.y; ++y) {
        for(int x = 0; x < this->out_.size.x; ++x) {
          
          int idx = 0;
          for(int yy = 0; yy < pool_size_; ++yy) {
            for(int xx = 0; xx < pool_size_; ++xx) {
              maxer[idx++] = in(y * pool_size_ + yy, x * pool_size_ + xx, c);
            }
          }
          this->out_(y, x, c) = xsimd::reduce(maxer.begin(), maxer.end(), maxer[0],
            [](const auto &a, const auto &b) { return xsimd::max(a, b); });

        }
      }
    }
  }

private:
  uint32_t pool_size_;
};

template <typename T> class DropoutLayer : public Layer<T> {
public:
  inline explicit DropoutLayer(const TensorSize &in_size, const float rate, int seed)
    : Layer<T>(in_size, {in_size.y, in_size.x, in_size.c}),
      rate(rate) {
        srand(seed);
      }

  inline void forward(const Tensor<T> &in) override {
    int total_num = this->in_size_.y * this->in_size_.x * this->in_size_.c;
    int num_to_zero = int ((float) total_num * rate);
    int start = rand() % total_num;

    auto in2 = typename Tensor<T>::vector_type(in.data(), in.data() + total_num);
    auto mask = typename Tensor<T>::vector_type (total_num);
    auto out2 = typename Tensor<T>::vector_type(total_num);

    for(int i = 0; i < total_num; ++i)
      mask[i] = (i < num_to_zero) ? (T) 0 : (T) 1;

    xsimd::transform (in2.begin(), in2.end(), mask.begin(), out2.begin(),
      [](const auto &a, const auto &b) { return a * b; });

    for(int i = 0; i < total_num; ++i)
      this->out_.data()[i] = out2[i];
  }

private:
  const float rate;
};

template <typename T> class FlattenLayer : public Layer<T> {
public:
  inline explicit FlattenLayer(const TensorSize &in_size)
    : Layer<T>(in_size, {1, in_size.y * in_size.x * in_size.c, 1}) {}

  inline void forward(const Tensor<T> &in) override {
    auto total_size = in.size.x * in.size.y * in.size.c * in.size.k;
    auto in2 = typename Tensor<T>::vector_type(in.data(), in.data() + total_size);
    xsimd::transform(in2.begin(), in2.end(), this->out_.begin(),
      [](auto const &a) { return a; });
  }
};

template <typename T> class DenseLayer : public Layer<T> {
public:
  inline DenseLayer(const TensorSize &in_size, uint32_t out_size)
    : Layer<T>(in_size, {1, out_size, 1}),
      weights_(in_size.x, out_size,  1),
      bias_(1, out_size, 1),
      prod(in_size.x) {
        w2 = new T*[out_size];
        for (int i = 0; i < out_size; ++i)
          w2[i] = new T[in_size.x];
      }
  
  ~DenseLayer() {
    for (int i = 0; i < this->out_.size.x; ++i)
      delete[] w2[i];
    delete[] w2;
  }

  inline void set_weight(uint32_t y, uint32_t x, uint32_t c, T value) {
    weights_(y, x, c) = value;
    w2[x][y] = value;
  }

  inline void forward(const Tensor<T> &in) override {
    auto xDim = in.size.x;
    auto in2 = typename Tensor<T>::vector_type(in.data(), in.data() + xDim);
    for (int l = 0; l < this->out_.size.x; ++l) {
      xsimd::transform(in2.begin(), in2.end(), w2[l], prod.begin(),
        [](auto const &a, auto const &b) { return a * b; });

      auto sum = xsimd::reduce (prod.begin(), prod.end(), (T) 0);
      
      this->out_(0, l, 0) = sum + bias_(0, l, 0);
    }
  }

  inline bool has_bias() const override { return true; }
  inline bool has_weights() const override { return true; }

  inline void load_weights(const Tensor<T> &weight) override {
    weights_.load(weight);

    for (int l = 0; l < this->out_.size.x; ++l) {
      for (int x = 0; x < this->in_size_.x; ++x) {
        auto idx = l * this->out_.size.x + x;
        w2[l][x] = weights_(x, l, 0);
      }
    }
  }
  inline TensorSize weight_size() const override { return weights_.size; }
  inline TensorSize bias_size() const override { return bias_.size; }

  inline void load_bias(const Tensor<T> &bias) override { bias_.load(bias); }

  inline TensorSize weights_size() const { return weights_.size; }
  inline virtual const Tensor<T> *get_weights() const { return &weights_; }
  inline virtual const Tensor<T> *get_bias() const { return &bias_; }

protected:
  Tensor<T> weights_;
  Tensor<T> bias_;

  T** w2;
  typename Tensor<T>::vector_type prod;
};

template <typename T> class ActivationLayer : public Layer<T> {
public:
  inline explicit ActivationLayer(const TensorSize &size)
      : Layer<T>(size, size) {}

  inline void forward(const Tensor<T> &in) override {
    for(int c = 0; c < this->in_size_.c; ++c) {
      for(int y = 0; y < this->in_size_.y; ++y) {
        for (int x = 0; x < this->in_size_.x; ++x) {
          this->out_(y, x, c) = activate_function(in(y, x, c));
        }
      }
    }
  }

protected:
  inline virtual T activate_function(T value) { return value; }
};

template <typename T> class ReLuActivationLayer : public ActivationLayer<T> {
public:
  inline explicit ReLuActivationLayer(const TensorSize &size)
      : ActivationLayer<T>(size),
        zeros(size.x * size.y * size.c * size.k) {
          for(int i = 0; i < size.x * size.y * size.c * size.k; ++i)
            zeros[i] = (T) 0;
        }

  inline void forward(const Tensor<T> &in) override {
    auto total_size = in.size.x * in.size.y * in.size.c * in.size.k;
    auto in2 = typename Tensor<T>::vector_type(in.data(), in.data() + total_size);
    xsimd::transform(in2.begin(), in2.end(), zeros.begin(), this->out_.begin(),
      [](auto const &a, auto const &b) { return xsimd::max(a, b); });
  }

protected:
  inline T activate_function(T value) override {
    return value > (T) 0 ? value : (T) 0;
  }

private:
  typename Tensor<T>::vector_type zeros;
};

template <typename T> class SigmoidActivationLayer : public ActivationLayer<T> {
public:
  inline explicit SigmoidActivationLayer(const TensorSize &size)
      : ActivationLayer<T>(size),
        total_size(this->in_size_.x * this->in_size_.y * this->in_size_.c * this->in_size_.k),
        out2(total_size) {}

  inline void forward(const Tensor<T> &in) override {
    auto in2 = typename Tensor<double>::vector_type(in.data(), in.data() + total_size);
    
    using b_type = xsimd::simd_type<double>;
    auto inc = Tensor<double>::simd_size();
    auto size = in2.size();
    // size for which the vectorization is possible
    auto vec_size = size - size % inc;
    for (std::size_t i = 0; i < vec_size; i += inc) {
      b_type a_vec = xsimd::load_aligned(&in2[i]);
      b_type r_vec = 1.0 / (1.0 + xsimd::exp(-a_vec));
      xsimd::store_aligned(&out2[i], r_vec);
    }
    // Remaining part that cannot be vectorize
    for (auto i = vec_size; i < size; ++i) {
      out2[i] = activate_function(in2[i]);
    }

    for(int i = 0; i < total_size; ++i) {
      this->out_.data()[i] = out2[i];
    }
  }

protected:
  inline T activate_function(T value) override {
      return (T) 1 / ((T) 1 + std::exp(-value));
  }
private:
  const int total_size;
  typename Tensor<double>::vector_type out2;
};

#endif // UDNN_LAYER_HH
