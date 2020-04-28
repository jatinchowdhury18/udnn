import tensorflow as tf
import numpy as np
import random
import time
import udnn

def bench_flatten(N=1000, size=5, dtype="int8", useTF=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())

    total_size = size**3
    input_tensor = np.reshape(
        np.array([random.randint(-127, 127) for i in range(total_size)],
                 dtype=dtype), (1, size, size, size))

    flatten = udnn.Flatten((size, size, size), dtype=dtype)
    input_tensor_udnn = np.reshape(input_tensor, (size, size, size))

    start = time.time()
    if useTF:
        for _ in range(N):
            tf_out = model.predict(input_tensor)

    else:
        for _ in range(N):
            flatten.forward(input_tensor_udnn)

    elapsed = time.time() - start
    avg = elapsed / N

    print(f'Ran flatten {N} times for {size}x{size}x{size} input: took {elapsed} seconds')
    print(f'Avg: {avg} seconds / op')
    return avg


def bench_dense(N=1000, size=5, dtype="int8", useTF=False):
    tf.random.set_seed(0)
    model = tf.keras.Sequential()
    input_shape = size
    output_shape = int(size/2)
    model.add(tf.keras.layers.Dense(output_shape, input_shape=(input_shape,),
                                    dtype=dtype,
                                    bias_initializer="random_uniform"))
    layer = udnn.Dense((1, input_shape, 1), dtype, output_shape)
    weights = model.weights[0].numpy().reshape(layer.weights.shape)
    bias = model.weights[1].numpy().reshape(layer.bias.shape)
    weights_t = udnn.tensor(weights, dtype=dtype)
    bias_t = udnn.tensor(bias, dtype=dtype)
    layer.load_weights(weights_t)
    layer.load_bias(bias_t)

    input_tensor = np.reshape(
        np.array([random.randint(-127, 127) for i in range(input_shape)],
                 dtype=dtype), (1, input_shape))
    input_tensor_udnn = input_tensor.reshape((1, input_shape, 1))

    start = time.time()
    if useTF:
        for _ in range(N):
            tf_out = model.predict(input_tensor)

    else:
        for _ in range(N):
            layer.forward(input_tensor_udnn)

    elapsed = time.time() - start
    avg = elapsed / N

    print(f'Ran dense {N} times for {size}x{size}x{size} input: took {elapsed} seconds')
    print(f'Avg: {avg} seconds / op')
    return avg


def bench_conv(N=1000, size=5, dtype="int8", useTF=False):
    tf.random.set_seed(0)
    input_size = size
    num_filter = size
    kernel_size = int(size/2)
    input_channel = 1
    input_shape = (1, input_size, input_size, input_channel)
    model = tf.keras.Sequential()
    conv = tf.keras.layers.Conv2D(num_filter, (kernel_size, kernel_size),
                                  dtype=dtype,
                                  bias_initializer="random_uniform")
    model.add(conv)
    model.build(input_shape)

    weights = model.weights[0].numpy()
    layer = udnn.Conv2D((input_size, input_size, input_channel), dtype,
                   kernel_size, num_filter)
    weights_t = udnn.tensor(weights, dtype=dtype)

    bias = model.weights[1].numpy().reshape(layer.bias.shape)
    bias_t = udnn.tensor(bias, dtype=dtype)
    layer.load_weights(weights_t)
    layer.load_bias(bias_t)

    total_num = input_size * input_size * input_channel
    data = [i for i in range(total_num)]
    input_tensor = np.array(data, dtype=dtype).reshape((1, input_size,
                                                            input_size,
                                                            input_channel))
    
    input_tensor_udnn = np.reshape(input_tensor,
                              (input_size, input_size, input_channel))

    start = time.time()
    if useTF:
        for _ in range(N):
            tf_out = model.predict(input_tensor)

    else:
        for _ in range(N):
            layer.forward(input_tensor_udnn)

    elapsed = time.time() - start
    avg = elapsed / N

    print(f'Ran conv {N} times for {size}x{size}x{size} input: took {elapsed} seconds')
    print(f'Avg: {avg} seconds / op')
    return avg


def bench_relu(N=1000, size=5, dtype="int8", useTF=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.ReLU())

    total_size = size**3
    relu = udnn.ReLu((size, size, size), dtype=dtype)
    
    input_tensor = np.reshape(
        np.array([random.uniform(-1.0, 1.0) for i in range(total_size)],
                 dtype=dtype), (1, size, size, size))
    input_tensor_udnn = np.reshape(input_tensor, (size, size, size))

    start = time.time()
    if useTF:
        for _ in range(N):
            tf_out = model.predict(input_tensor)

    else:
        for _ in range(N):
            relu.forward(input_tensor_udnn)

    elapsed = time.time() - start
    avg = elapsed / N

    print(f'Ran relu {N} times for {size}x{size}x{size} input: took {elapsed} seconds')
    print(f'Avg: {avg} seconds / op')
    return avg


def bench_sigmoid(N=1000, size=5, dtype="int8", useTF=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.ReLU())

    total_size = size**3
    sig = udnn.Sigmoid((size, size, size), dtype=dtype)
    
    input_tensor = np.reshape(
        np.array([random.uniform(-1.0, 1.0) for i in range(total_size)],
                 dtype=dtype), (1, size, size, size))
    input_tensor_udnn = np.reshape(input_tensor, (size, size, size))

    start = time.time()
    if useTF:
        for _ in range(N):
            tf_out = model.predict(input_tensor)

    else:
        for _ in range(N):
            sig.forward(input_tensor_udnn)

    elapsed = time.time() - start
    avg = elapsed / N

    print(f'Ran sigmoid {N} times for {size}x{size}x{size} input: took {elapsed} seconds')
    print(f'Avg: {avg} seconds / op')
    return avg


def bench_maxpool(N=1000, size=5, dtype="int8", useTF=False):
    tf.random.set_seed(0)
    input_size = size
    pool_size = int(size/2)
    input_channel = 1
    input_shape = (1, input_size, input_size, input_channel)
    model = tf.keras.Sequential()
    max_pool = tf.keras.layers.MaxPool2D((pool_size, pool_size),
                                  dtype=dtype)
    model.add(max_pool)
    model.build(input_shape)

    layer = udnn.MaxPooling((input_size, input_size, input_channel), dtype, pool_size)

    total_num = input_size * input_size * input_channel
    data = [i for i in range(total_num)]
    input_tensor = np.array(data, dtype=dtype).reshape((1, input_size,
                                                            input_size,
                                                            input_channel))

    input_tensor_udnn = np.reshape(input_tensor,
                              (input_size, input_size, input_channel))
    
    start = time.time()
    if useTF:
        for _ in range(N):
            tf_out = model.predict(input_tensor)

    else:
        for _ in range(N):
            layer.forward(input_tensor_udnn)

    elapsed = time.time() - start
    avg = elapsed / N

    print(f'Ran maxpool {N} times for {size}x{size}x{size} input: took {elapsed} seconds')
    print(f'Avg: {avg} seconds / op')
    return avg


if __name__ == "__main__":
    op = 'dense'
    pack = 'fast'

    if op == 'flatten':
        dims = np.array([10, 20, 50, 100, 150, 200])
        dtypes = ['int8', 'int16', 'float32', 'double']
    elif op == 'dense':
        dtypes = ['float32', 'double']
        dims = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    elif op == 'conv':
        dtypes = ['float32', 'double']
        dims = np.array([20, 30, 40, 50, 60,])
    elif op == 'relu' or op == 'sigmoid':
        dtypes = ['float32', 'double']
        dims = np.array([10, 50, 100, 150, 200, 250])
    elif op == 'maxpool':
        dtypes = ['int8', 'int16', 'float32', 'double']
        dims = np.array([50, 100, 200, 500, 1000, 2000, 5000])
    else:
        exit()

    useTF = False
    if pack == 'tf':
        useTF = True

    for dtype in dtypes:
        times = []
        for d in dims:
            if op == 'flatten':
                T = bench_flatten(N=500, size=d, useTF=useTF, dtype=dtype)
            elif op == 'dense':
                T = bench_dense(N=200, size=d, useTF=useTF, dtype=dtype)
            elif op == 'conv':
                T = bench_conv(N=200, size=d, useTF=useTF, dtype=dtype)
            elif op == 'relu':
                T = bench_relu(N=200, size=d, useTF=useTF, dtype=dtype)
            elif op == 'sigmoid':
                T = bench_sigmoid(N=200, size=d, useTF=useTF, dtype=dtype)
            elif op == 'maxpool':
                T = bench_maxpool(N=200, size=d, useTF=useTF, dtype=dtype)
            times.append(T)

        results = np.array([dims, times], dtype=np.float32)
        np.savetxt(f'bench_results/{pack}/{op}_{dtype}.csv', results)
