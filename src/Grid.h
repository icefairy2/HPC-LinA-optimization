#ifndef GRID_H_
#define GRID_H_

#include <cstring>
#include <utility>
#include <mpi.h>

template<typename T>
class Grid {
public:
  Grid(int X, int Y, int pX, int pY);
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
  bool checkCoordsRank(int x, int y) const;
  void gather(int root = 0);
  void gatherGhost();

private:
  inline std::pair<int, int> rankToCoords(int i) const;
  inline int coordsToRank(int y, int x) const;

  int m_X, m_Y;
  int m_pX, m_pY;
  T* m_data;

  int m_Xfrom, m_Xto;
  int m_Yfrom, m_Yto;
  int mpiRank, mpiSize;
  MPI_Datatype MPI_YGHOST, MPI_SUBGRID;
};

template<typename T>
Grid<T>::Grid(int X, int Y, int pX, int pY)
  : m_X(X), m_Y(Y), m_pX(pX), m_pY(pY)
{
  m_data = new T[X*Y];
  memset(m_data, 0, X*Y*sizeof(T));

  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  m_Xfrom = mpiRank * m_pX % m_X;
  m_Xto = m_Xfrom + m_pX;
  m_Yfrom = mpiRank * m_pX / m_X * m_pY % m_Y;
  m_Yto = m_Yfrom + m_pY;
  MPI_Type_vector(m_pY, sizeof(T), m_X * sizeof(T), MPI_BYTE, &MPI_YGHOST);
  MPI_Type_commit(&MPI_YGHOST);
  MPI_Type_vector(m_pY, m_pX * sizeof(T), m_X * sizeof(T), MPI_BYTE, &MPI_SUBGRID);
  MPI_Type_commit(&MPI_SUBGRID);
}

template<typename T>
Grid<T>::~Grid() {
  delete[] m_data;
}

template <typename T>
inline bool Grid<T>::checkCoordsRank(int x, int y) const {
  return coordsToRank(y, x) == mpiRank;
}

template <typename T>
inline std::pair<int, int> Grid<T>::rankToCoords(int i) const {
  return std::make_pair(i * m_pX / m_X * m_pY % m_Y, i * m_pX % m_X);
}

template <typename T>
int Grid<T>::coordsToRank(int y, int x) const {
  return x / m_pX + (y / m_pY) * (m_X / m_pX);
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

template <typename T>
void Grid<T>::gatherGhost() {
  if (m_Y / m_pY > 1) {
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
  }

  if (m_X / m_pX > 1) {
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
  }
}

#endif // GRID_H_
