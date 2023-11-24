import numpy as np

n = 1024 * 1024
n_caption = '1M'
dims = [4, 8, 16, 32, 64]
grid_sizes = [128, 256, 512, 1024]
single_precision = True


def create_normal_data_file(file_name, loc=0.0, scale=1.0, size=None):
    data = np.random.normal(loc=loc, scale=scale, size=size)
    data.astype('float32' if single_precision else 'float64').tofile(file_name)


for dim in dims:
    points_file = f'points-{n_caption}-{dim}.bin'
    print(f'Generating {points_file} ...')
    create_normal_data_file(points_file, size=(n, dim))

    for grid_size in grid_sizes:
        grid_file = f'grid-{grid_size}-{dim}.bin'
        print(f'Generating {grid_file} ...')
        create_normal_data_file(grid_file, size=(grid_size, dim))

for grid_size in grid_sizes:
    grid_file = f'grid2d-{grid_size}.bin'
    print(f'Generating {grid_file} ...')
    create_normal_data_file(grid_file, size=(grid_size, 2))
