#include "../extern/googletest/googletest/include/gtest/gtest.h"
#include "../src/udnn.hh"

TEST(layer, conv) {
  Conv2DLayer<int8_t> layer({4, 4, 3}, 3, 1);
  // this will be aggregation
  for (uint32_t y = 0; y < 3; y++) {
    for (uint32_t x = 0; x < 3; x++) {
      for (uint32_t c = 0; c < 3; c++) {
        layer.set_weight(y, x, c, 0, 1);
      }
    }
  }
  // inputs are all 2
  Tensor<int8_t> input(4, 4, 3);
  for (uint32_t y = 0; y < 4; y++) {
    for (uint32_t x = 0; x < 4; x++) {
      for (uint32_t c = 0; c < 3; c++) {
        input(y, x, c, 0) = 2;
      }
    }
  }

  layer.forward(input);
  auto const &out = layer.out();
  EXPECT_EQ(out.size.x, 2);
  EXPECT_EQ(out.size.y, 2);
  EXPECT_EQ(out.size.c, 1);
  EXPECT_EQ(out(1, 1, 0), 3 * 3 * 3 * 2);
}

TEST(layer, maxpool) {
  MaxPoolingLayer<int8_t> layer({4, 4, 3}, 2);

  Tensor<int8_t> input(4, 4, 3);
  for (uint32_t y = 0; y < 4; y++) {
    for (uint32_t x = 0; x < 4; x++) {
      for (uint32_t c = 0; c < 3; c++) {
        input(y, x, c, 0) = x + y;
      }
    }
  }

  layer.forward(input);
  auto const &out = layer.out();
  EXPECT_EQ(out.size.x, 2);
  EXPECT_EQ(out.size.y, 2);
  EXPECT_EQ(out.size.c, 3);
  EXPECT_EQ(out(0, 0, 0), 2);
  EXPECT_EQ(out(1, 1, 1), 6);
}

TEST(layer, dropout) {
  DropoutLayer<int8_t> zero_layer({2, 2, 2}, 0.0f, 0);
  DropoutLayer<int8_t> one_layer({2, 2, 2}, 1.0f, 0);

  Tensor<int8_t> input(2, 2, 2);
  for (uint32_t y = 0; y < 2; y++) {
    for (uint32_t x = 0; x < 2; x++) {
      for (uint32_t c = 0; c < 2; c++) {
        input(y, x, c, 0) = x + y;
      }
    }
  }

  zero_layer.forward(input);
  auto const& zero_out = zero_layer.out();

  one_layer.forward(input);
  auto const& one_out = one_layer.out();

  EXPECT_EQ(zero_out(0, 0, 0), input(0, 0, 0));
  EXPECT_EQ(one_out(1, 1, 1), 0);
}

TEST(layer, fc) {
  DenseLayer<int8_t> fc_layer({1, 8, 1}, 2);
  Tensor<int8_t> input_value(1, 8, 1);
  // they all have weights 1
  for (auto x = 0; x < 8; x++) {
    fc_layer.set_weight(x, 0, 0, 1.0f);
    fc_layer.set_weight(x, 1, 0, 1.0f);
    input_value(0, x, 0) = 2.0f;
  }

  // the sum should be 8, 8
  fc_layer.forward(input_value);
  auto const &out = fc_layer.out();
  EXPECT_EQ(out(0, 0, 0), 8*2);
}

TEST(layer, relu) {
  ReLuActivationLayer<float> relu({2, 2, 2});
  Tensor<float> input_value(2, 2, 2);
  for (auto y = 0; y < 2; y++) {
    for (auto x = 0; x < 2; x++) {
      for (auto c = 0; c < 2; c++) {
        input_value(x, y, c) = y * x - c * 2;
      }
    }
  }
  relu.forward(input_value);
  auto const &out = relu.out();
  EXPECT_EQ(out(0, 0, 1), 0);
  EXPECT_EQ(out(1, 1, 0), 1);
}

TEST(layer, sigmoid) {
  const int dim = 10;
  SigmoidActivationLayer<float> sigmoid({dim, dim, dim});
  Tensor<float> input_value(dim, dim, dim);
  for (auto y = 0; y < dim; y++) {
    for (auto x = 0; x < dim; x++) {
      for (auto c = 0; c < dim; c++) {
        input_value(x, y, c) = y * x - c * dim;
      }
    }
  }
  sigmoid.forward(input_value);
  auto const &out = sigmoid.out();
  EXPECT_EQ(out(0, 0, 0), 0.5);
  EXPECT_EQ(out(dim-1, dim-1, 0), 1.0);
  EXPECT_TRUE((out(0, 0, dim-1) < 0.1) && (out(0, 0, dim-1) > -0.1));
}

TEST(layer, flatten) {
  FlattenLayer<int8_t> f({3, 2, 1});
  EXPECT_EQ(f.out_size().x, 6);
  EXPECT_EQ(f.out_size().y, 1);
  EXPECT_EQ(f.out_size().c, 1);

  Tensor<int8_t> tensor(3, 2, 1);
  int8_t count = 0;
  for (uint32_t y = 0; y < 3; y++)
    for (uint32_t x = 0; x < 2; x++)
      tensor(y, x, 0) = count++;

  f.forward(tensor);
  auto out = f.out();
  for (uint32_t i = 0; i < 6; i++)
    EXPECT_EQ(out(0, i, 0), i);
}

TEST(layer, connect) {
  DenseLayer<int8_t> fc_layer({1, 4, 1}, 2);
  ReLuActivationLayer<float> relu0({1, 2, 2});
  EXPECT_THROW(fc_layer.connect(&relu0), std::invalid_argument);
  ReLuActivationLayer<int8_t> relu1({1, 2, 1});
  EXPECT_NO_THROW(fc_layer.connect(&relu1));
}

TEST(layer, types) {
  Conv2DLayer<int8_t> layer({4, 4, 3}, 3, 1);
  EXPECT_EQ(layer.in_type(), DType::Int8);
  ReLuActivationLayer<float> relu({2, 2, 2});
  EXPECT_EQ(relu.in_type(), DType::Float);
  EXPECT_EQ(relu.out_type(), DType::Float);
}