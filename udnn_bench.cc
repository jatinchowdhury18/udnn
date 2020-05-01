#include <iostream>
#include <chrono>
#include <cstdlib>
#include "src/udnn.hh"
using namespace std::chrono;

using dtype = float;

float get_rand_float()
{
    return 2.0f * static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 1.0f;
}

void gen_random_weights_for_layer(Layer<dtype>* layer)
{
    if(layer->has_weights())
    {
        auto weights = *layer->get_weights();
        auto weights_size = layer->weights_size();

        for(int k = 0; k < weights_size.k; ++k)
            for(int c = 0; c < weights_size.c; ++c)
                for(int x = 0; x < weights_size.x; ++x)
                    for(int y = 0; y < weights_size.y; ++y)
                        weights(y, x, c, k) = get_rand_float();
    }

    if(layer->has_bias())
    {
        auto biases = *layer->get_bias();
        auto bias_size = layer->bias_size();

        for(int k = 0; k < bias_size.k; ++k)
            for(int c = 0; c < bias_size.c; ++c)
                for(int x = 0; x < bias_size.x; ++x)
                    for(int y = 0; y < bias_size.y; ++y)
                        biases(y, x, c, k) = get_rand_float();
    }
}

int main()
{
    srand(0);

    std::cout << "Creating model layers..." << std::endl;
    Conv2DLayer<dtype> conv0({32, 32, 1}, 3, 32);
    ReLuActivationLayer<dtype> relu0({30, 30, 32});
    Conv2DLayer<dtype> conv1({30, 30, 32}, 3, 32);
    ReLuActivationLayer<dtype> relu1({28, 28, 32});
    MaxPoolingLayer<dtype> maxpool1({28, 28, 32}, 2);
    Conv2DLayer<dtype> conv2({14, 14, 32}, 3, 64);
    ReLuActivationLayer<dtype> relu2({12, 12, 64});
    MaxPoolingLayer<dtype> maxpool2({12, 12, 64}, 2);
    FlattenLayer<dtype> flatten3({6, 6, 64});
    DenseLayer<dtype> dense3({1, 2304, 1}, 512);
    ReLuActivationLayer<dtype> relu3({1, 512, 1});
    DenseLayer<dtype> dense4({1, 512, 1}, 10);
    SigmoidActivationLayer<dtype> sigmoid4({1, 10, 1});

    //////////////////////////////////////////////////////
    std::cout << "Generating random model weights..." << std::endl;
    gen_random_weights_for_layer(&conv0);
    gen_random_weights_for_layer(&conv1);
    gen_random_weights_for_layer(&conv2);
    gen_random_weights_for_layer(&dense3);
    gen_random_weights_for_layer(&dense4);

    //////////////////////////////////////////////////////
    std::cout << "Adding layers to model..." << std::endl;
    Model my_model;
    my_model.add_layer("conv0", &conv0);
    my_model.add_layer("relu0", &relu0);
    my_model.add_layer("conv1", &conv1);
    my_model.add_layer("relu1", &relu1);
    my_model.add_layer("maxpool1", &maxpool1);
    my_model.add_layer("conv2", &conv2);
    my_model.add_layer("relu2", &relu2);
    my_model.add_layer("maxpool2", &maxpool2);
    my_model.add_layer("flatten3", &flatten3);
    my_model.add_layer("dense3", &dense3);
    my_model.add_layer("relu3", &relu3);
    my_model.add_layer("dense4", &dense4);
    my_model.add_layer("sigmoid4", &sigmoid4);

    //////////////////////////////////////////////////////
    std::cout << "Running inference..." << std::endl;

    Tensor<dtype> tensor(32, 32, 1);
    constexpr int numTrials = 100;

    auto start = high_resolution_clock::now();
    for (int i = 0; i < numTrials; ++i)
        my_model.predict(&tensor);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    std::cout << "Full time: " << duration.count() << " seconds" << std::endl;

    return 0;
}
