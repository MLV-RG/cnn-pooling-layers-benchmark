from benchmark import Benchmark

for method in ['max', 'adaptive_max', 'fractional', 'average', 'mixed', 'gated', 'tree', 'l2', 'stochastic', 'fuzzy',
               'overlapping', 'spectral', 'wavelet', 'lip', 'softpool']:
    for dataset in ['cifar10', 'cifar100', 'mnist']:
        benchmark = Benchmark(pooling_function=method, dataset=dataset, epochs=300)
        benchmark.start_train()
