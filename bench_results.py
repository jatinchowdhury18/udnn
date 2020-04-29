import numpy as np
import matplotlib.pyplot as plt

ops = ['flatten', 'dense', 'conv', 'relu', 'sigmoid', 'maxpool']

for op in ops:
    if op in ['dense', 'conv', 'relu', 'sigmoid']:
        dtypes = ['float32', 'double']
    else:
        dtypes = ['int8', 'int16', 'float32', 'double']

    for dtype in dtypes:
        plt.figure()
        Xtf = np.loadtxt(f'bench_results/tf/{op}_{dtype}.csv')
        if op != 'maxpool':
            Xslow = np.loadtxt(f'bench_results/slow/{op}_{dtype}.csv')
        Xfast = np.loadtxt(f'bench_results/fast/{op}_{dtype}.csv')

        L = len(Xtf[0,:])

        plt.plot(Xtf[0,:L], Xtf[1,:L], label=f'tf')
        if op != 'maxpool':
            plt.plot(Xslow[0,:L], Xslow[1,:L], label=f'UDNN slow')
        plt.plot(Xfast[0,:L], Xfast[1,:L], label=f'UDNN fast')

        plt.legend()
        plt.xlabel('Dimension')
        plt.ylabel('Seconds')
        plt.title(f'{op} Benchmark ({dtype})')

        plt.savefig(f'bench_results/{op}_{dtype}.png')
