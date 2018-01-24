#ifndef GRID_H_
#define GRID_H_

#include <cstring>
#include <utility>
#include <mpi.h>

template<typename T>
class Grid {
public:
  Grid(int X, int Y);
  ~Grid();
  
  /// Implements periodic boundary conditions
  inline int periodicIndex(int i, int N) {
    int pi;
    if (i >= 0) {
      pi = i % N;
    } else {
      pi = N-1 - (-i-1)%N;
    }
    return pi;
  }
  
  inline T& get(int x, int y) {
    return m_data[periodicIndex(y, m_Y) * m_X + periodicIndex(x, m_X)];
  }
  
  inline int X() const {
    return m_X;
  }
  
  inline int Y() const {
    return m_Y;
  }
  
  inline std::pair<int, int> getYlimits() const;
  inline std::pair<int, int> getXlimits() const;
  void gather(int root = 0);
  void gatherGhost();

private:
  int m_X;
  int m_Y;
  T* m_data;

  int m_Xfrom, m_Xto;
  int m_Yfrom, m_Yto;
  int mpiRank, mpiSize;
};

template<typename T>
Grid<T>::Grid(int X, int Y)
  : m_X(X), m_Y(Y)
{
  m_data = new T[X*Y];
  memset(m_data, 0, X*Y*sizeof(T));

  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  m_Yfrom = m_Y * mpiRank / mpiSize;
  m_Yto = m_Y * (mpiRank + 1) / mpiSize;
  m_Xfrom = 0;
  m_Xto = m_X;
}

template<typename T>
Grid<T>::~Grid() {
  delete[] m_data;
}

template <typename T>
std::pair<int, int> Grid<T>::getYlimits() const {
  return std::make_pair(m_Yfrom, m_Yto);
}

template <typename T>
std::pair<int, int> Grid<T>::getXlimits() const {
  return std::make_pair(m_Xfrom, m_Xto);
}

template <typename T>
void Grid<T>::gather(int root) {
  if (mpiRank == root) {
    for (int i = 0; i < mpiSize; i++)
      if (i != root)
        MPI_Recv(&m_data[m_Y * i / mpiSize * m_X], m_Y / mpiSize * m_X * sizeof(T), MPI_BYTE, i, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    MPI_Send(&m_data[m_Yfrom * m_X], m_Y / mpiSize * m_X * sizeof(T), MPI_BYTE, root, 0,
             MPI_COMM_WORLD);
  }
}

template <typename T>
void Grid<T>::gatherGhost() {
  if (mpiSize == 1)
    return;

  int count = m_X * sizeof(T);
  for (int i = mpiRank; i < mpiRank + 2; i++)
    if (i % 2 == 0) {
      int other = (mpiRank + 1) % mpiSize;
      MPI_Sendrecv(&m_data[(m_Yto - 1) * m_X], count, MPI_BYTE, other, 0,
                   &m_data[periodicIndex(m_Yto, m_Y) * m_X], count, MPI_BYTE, other, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      int other = (mpiRank + mpiSize - 1) % mpiSize;
      MPI_Sendrecv(&m_data[m_Yfrom * m_X], count, MPI_BYTE, other, 0,
                   &m_data[periodicIndex(m_Yfrom - 1, m_Y) * m_X], count, MPI_BYTE, other, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

#endif // GRID_H_
