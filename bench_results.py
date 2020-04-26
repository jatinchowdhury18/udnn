import numpy as np
import matplotlib.pyplot as plt

ops = ['flatten', 'dense', 'conv', 'relu', 'sigmoid']

for op in ops:
    if op in ['dense', 'conv', 'relu', 'sigmoid']:
        dtypes = ['float32', 'double']
    else:
        dtypes = ['int8', 'int16', 'float32', 'double']

    plt.figure()
    for dtype in dtypes:
        Xtf = np.loadtxt(f'bench_results/tf/{op}_{dtype}.csv')
        Xslow = np.loadtxt(f'bench_results/slow/{op}_{dtype}.csv')

        plt.plot(Xtf[0,:], Xtf[1,:], label=f'tf {dtype}')
        plt.plot(Xslow[0,:], Xslow[1,:], label=f'UDNN slow {dtype}')

    plt.legend()
    plt.xlabel('Dimension')
    plt.ylabel('Seconds')
    plt.title(f'{op} Benchmark ({dtype})')

    plt.savefig(f'bench_results/{op}.png')
