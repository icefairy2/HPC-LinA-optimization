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
