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
    : Layer<T>(in_size, {in_size.y - (filter_size-1), in_size.x - (filter_size-1), in_size.c - (filter_size-1)}),
      weights_(filter_size, filter_size, filter_size, num_filters),
      bias_(in_size.y - (filter_size-1), in_size.x - (filter_size-1), in_size.c * num_filters) {}

  inline void set_weight(uint32_t y, uint32_t x, uint32_t c, uint32_t k, T value) {
    weights_(y, x, c, k) = value;
  }

  inline void load_weights(const Tensor<T> &weight) override {
    weights_.load(weight);
  }

  inline TensorSize weights_size() const { return weights_.size; }

  inline void forward(const Tensor<T> &in) override {
    for(int k = 0; k < weights_.size.k; ++k) {
      for(int c = 0; c < this->out_.size.c; ++c) {
        for(int y = 0; y < this->out_.size.y; ++y) {
          for (int x = 0; x < this->out_.size.x; ++x) {
  
            T sum = (T) 0;
            for(int cc = 0; cc < weights_.size.c; ++cc) {
              for(int yy = 0; yy < weights_.size.y; ++yy) {
                for(int xx = 0; xx < weights_.size.x; ++xx) {
                  sum += in(y + yy, x + xx, c + cc) * weights_(yy, xx, cc, k);
                }
              }
            }
  
            this->out_(y, x, c + k) = sum + bias_(y, x, c);
          }
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
  inline explicit MaxPoolingLayer(const TensorSize &in_size,
                                  uint32_t pool_size) {}

  inline void forward(const Tensor<T> &in) override {}

private:
  uint32_t pool_size_;
};

template <typename T> class FlattenLayer : public Layer<T> {
public:
  inline explicit FlattenLayer(const TensorSize &in_size)
    : Layer<T>(in_size, {1, in_size.y * in_size.x * in_size.c, 1}) {}

  inline void forward(const Tensor<T> &in) override {
      auto rows = this->in_size_.y;
      auto cols = this->in_size_.x;
      auto ch   = this->in_size_.c;
      for(int c = 0; c < ch; ++c) {
        for(int y = 0; y < rows; ++y) {
          for(int x = 0; x < cols; ++x) {
            auto idx = c + y * cols * ch + x * ch;
            this->out_(0, idx, 0) = in(y, x, c);
          }
        }
      }
  }
};

template <typename T> class DenseLayer : public Layer<T> {
public:
  inline DenseLayer(const TensorSize &in_size, uint32_t out_size)
    : Layer<T>(in_size, {1, out_size, 1}),
      weights_(in_size.x, out_size,  1),
      bias_(1, out_size, 1) {}

  inline void set_weight(uint32_t y, uint32_t x, uint32_t c, T value) {
    weights_(y, x, c) = value;
  }

  inline void forward(const Tensor<T> &in) override {
    auto L = this->out_.size.x;
      
    for (int l = 0; l < L; ++l)
    {
      T sum = (T) 0;
      for (int x = 0; x < this->in_size_.x; ++x)
        sum += in(0, x, 0) * weights_(x, l, 0);

      this->out_(0, l, 0) = sum + bias_(0, l, 0);
    }
  }

  inline bool has_bias() const override { return true; }
  inline bool has_weights() const override { return true; }

  inline void load_weights(const Tensor<T> &weight) override {
    weights_.load(weight);
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
};

template <typename T> class ActivationLayer : public Layer<T> {
public:
  inline explicit ActivationLayer(const TensorSize &size)
      : Layer<T>(size, size) {}

  inline void forward(const Tensor<T> &in) {
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
      : ActivationLayer<T>(size) {}

protected:
  inline T activate_function(T value) override {
    return value > (T) 0 ? value : (T) 0;
  }
};

template <typename T> class SigmoidActivationLayer : public ActivationLayer<T> {
public:
  inline explicit SigmoidActivationLayer(const TensorSize &size)
      : ActivationLayer<T>(size) {}

protected:
  inline T activate_function(T value) override {
      return (T) 1 / ((T) 1 + std::exp(-value));
  }
};

#endif // UDNN_LAYER_HH
