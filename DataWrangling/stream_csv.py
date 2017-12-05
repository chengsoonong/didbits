"""Use a generator with numpy loadtxt"""

import numpy as np


def file_stream(file_name):
    for line in open(file_name):
        yield line


def demo_use():
    file_name = 'census_abs2011_summary.csv'
    # Skip the header, return 3 columns: the age, weekly rental and total family income
    data = np.loadtxt(file_stream(file_name), delimiter=',', usecols=(0, 2, 3), skiprows=1)
    print('the loaded data')
    print(data.shape)
    print(data)


if __name__ == '__main__':
    demo_use()
