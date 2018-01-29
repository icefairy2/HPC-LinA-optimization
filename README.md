# LinA
Teaching code for high performance computing lab.
This assignment focuses on optimizing the linear acoustics algorithm
using any parallelization technique possible (MPI, OpenMP, single core
vectorization) and other improvements.

## Optimization approach

### MPI

The MPI optimization consists in splitting the grid data computation equally
among the MPI processes. Initially, all processes form their own copy of
`materialGrid` and `degreesOfFreedomGrid`.

The user is able to configure
how many columns and rows from the grids each process has to compute by
passing the `-a` and `-b` command line parameters. The sub-grids have to
divide the whole grid perfectly, therefore `-x` and `-y` should be divided
by `-a` and `-b` respectively.

Example run command:
```
mpirun -np 2 build/lina -s 0 -x 10 -y 10 -a 10 -b 5
```

According to its rank, each process will compute its processing limits
as follows:

```c
m_Xfrom = mpiRank * m_pX % m_X;
m_Xto = m_Xfrom + m_pX;
m_Yfrom = mpiRank * m_pX / m_X * m_pY % m_Y;
m_Yto = m_Yfrom + m_pY;
```

where `m_pX` and `m_pY` are the values specified by `-a` and `-b` parameters.

During the first part of the main loop in `Simulator.cpp` all oprations are
local to an element. However, the second part interacts with neighbouring
elements. Thus, to compute the next iteration borders of the sub-grids,
each process needs to get the rows and column of the neighbouring ranks
encompassing its own sub-grid before the second processing part.

This acquisition is realized in 2 steps. First, each even ranked process
sends its last row to the underneath neighbouring process. At the same time,
the odd ranked processes hand their first row to the process above. Then,
the same procedure happens in reverse order, odd processes handing their last
row and even ones their first row.

```c
for (int i = m_Yfrom / m_pY; i < m_Yfrom / m_pY + 2; i++)
  if (i % 2 == 0) {
    int under = coordsToRank(periodicIndex(m_Yto, m_Y), m_Xfrom);
    MPI_Sendrecv(&m_data[(m_Yto - 1) * m_X + m_Xfrom], m_pX * sizeof(T), MPI_BYTE, under, 0,
                 &m_data[periodicIndex(m_Yto, m_Y) * m_X + m_Xfrom], m_pX * sizeof(T), MPI_BYTE, under, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    int above = coordsToRank(periodicIndex(m_Yfrom - 1, m_Y), m_Xfrom);
    MPI_Sendrecv(&m_data[m_Yfrom * m_X + m_Xfrom], m_pX * sizeof(T), MPI_BYTE, above, 0,
                 &m_data[periodicIndex(m_Yfrom - 1, m_Y) * m_X + m_Xfrom], m_pX * sizeof(T), MPI_BYTE, above, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
```

The same approach is applied for bordering left and right columns. However,
as the columns are not contiguous in memory, we have created an MPI type
to specify how many elements are in the column and the memory gap between them.

```c
MPI_Type_vector(m_pY, sizeof(T), m_X * sizeof(T), MPI_BYTE, &MPI_YGHOST);
MPI_Type_commit(&MPI_YGHOST);
```

The final column sharing is as follows:
```c
for (int i = m_Xfrom / m_pX; i < m_Xfrom / m_pX + 2; i++)
  if (i % 2 == 0) {
    int right = coordsToRank(m_Yfrom, periodicIndex(m_Xto, m_X));
    MPI_Sendrecv(&m_data[m_Yfrom * m_X + m_Xto - 1], 1, MPI_YGHOST, right, 0,
                 &m_data[m_Yfrom * m_X + periodicIndex(m_Xto, m_X)], 1, MPI_YGHOST, right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    int left = coordsToRank(m_Yfrom, periodicIndex(m_Xfrom - 1, m_X));
    MPI_Sendrecv(&m_data[m_Yfrom * m_X + m_Xfrom], 1, MPI_YGHOST, left, 0,
                 &m_data[m_Yfrom * m_X + periodicIndex(m_Xfrom - 1, m_X)], 1, MPI_YGHOST, left, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
```

If the wavefield output is not done in parallel, then it shall remain the
rank 0 process' job to output it. To do this, it needs to gather the
`degreesOfFreedomGrid` from all other processes at the beginning of each
time step. The `Grid.h` now provides and implementation for that with the
`gather` method.

```c++
void Grid<T>::gather(int root) {
  if (mpiRank == root) {
    for (int i = 0; i < mpiSize; i++)
      if (i != root) {
        std::pair<int, int> coords = rankToCoords(i);
        MPI_Recv(&m_data[coords.first * m_X + coords.second], 1, MPI_SUBGRID, i, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
  } else {
    MPI_Send(&m_data[m_Yfrom * m_X + m_Xfrom], 1, MPI_SUBGRID, root, 0,
             MPI_COMM_WORLD);
  }
}
```

As with the columns, the sub-grids are not stored completely in a
contiguous memory area and thus we have create the `MPI_SUBGRID` type, defining
the length of each subgrid row and the memory gap between each consecutive rows.

```c
MPI_Type_vector(m_pY, m_pX * sizeof(T), m_X * sizeof(T), MPI_BYTE, &MPI_SUBGRID);
MPI_Type_commit(&MPI_SUBGRID);
```

The same gather procedure is applied at the end of the `simulate` function
so that the error output in the main function may have the final computed
`degreesOfFreedomGrid`.

Testing revealed a speedup proportional to the number of MPI processes.

### OpenMP

As processing each element in the grid is independent from the rest,
we can parallelize the two for loops iteration over the grid in the
`Simulation.cpp`.

```c
#pragma omp parallel for collapse(2)
for (int y = ylimits.first; y < ylimits.second; ++y) {
  for (int x = xlimits.first; x < xlimits.second; ++x) {
```

As the for loops remain unchanged, this approach can be applied on both
the original and the MPI version.


### Parallel Wavefield Output

When we enable output only the master process can write to the file. This means
that all information needs to be gathered at the master process leading to
overhead. In order to strive for a better time we tried to allow for parallel
writes to a file. For this purpose we use the hdf5 file format which allows
for writing multiple scientific datasets to the same file. The format functions
as follows:

* Each dataset is written to a specific path in the file. For example: Pressure
  goes to filename:/pressure, uvvel goes to filename:/u and so on.
* Each process writes to a hyperslab governed by the starting and ending indexes
  (of the Grid) that it is working on.

This process is illustrated in the following figure:

![Hyperslabs per process](https://support.hdfgroup.org/HDF5/Tutor/image/pimg034.gif)

The XDMF format that paraview uses has readers for HDF5 files built in. Hence,
we only need to point to the correct path for the dataset as illustrated above
and change the format from 'Binary' to 'HDF'.

After these changes no gather step needs to be done at the master and each
process is responsible for writing its own wavefield output.

## Compile and run

To run the implementation, one must:
- clone the repository on *CooLMUC3*
- load the hdf5 module: `module load hdf5/mpi/1.8.15`
- compile: `export ORDER=2;bash -x compilescript`
- create a script to run the code with mpi (e.g. it should contain an
mpirun command as `mpirun -np 2 build/lina -s 0 -x 10 -y 10 -a 10 -b 5 -o output/test`
.
