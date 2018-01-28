#include "WaveFieldWriter.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <limits>
#include <hdf5.h>

#include "basisfunctions.h"
#include "GEMM.h"

WaveFieldWriter::WaveFieldWriter(std::string const& baseName, GlobalConstants const& globals, double interval, int pointsPerDim)
  : m_step(0), m_interval(interval), m_lastTime(-std::numeric_limits<double>::max()), m_pointsPerDim(pointsPerDim)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    if (!baseName.empty()) {
        // Only root proccess writest he xdmf
        if (m_rank == 0) {
            m_xdmf.open((baseName + ".xdmf").c_str());
            m_xdmf  << "<?xml version=\"1.0\" ?>" << std::endl
            << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\">" << std::endl
            << "<Xdmf Version=\"2.0\">" << std::endl
            << "  <Domain>" << std::endl
            << "    <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\"" << m_pointsPerDim * globals.Y << " " << m_pointsPerDim * globals.X << "\"/>" << std::endl
            << "    <Geometry GeometryType=\"ORIGIN_DXDY\">" << std::endl
            << "      <DataItem Format=\"XML\" Dimensions=\"2\">0.0 0.0</DataItem>" << std::endl
            << "      <DataItem Format=\"XML\" Dimensions=\"2\">" << globals.hy / m_pointsPerDim << " " << globals.hx / m_pointsPerDim << "</DataItem>" << std::endl
            << "    </Geometry>" << std::endl
            << "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">" << std::endl;
        }

        // int gridSize = m_pointsPerDim * m_pointsPerDim * globals.X * globals.Y;
        int gridSize = m_pointsPerDim * m_pointsPerDim * globals.pX * globals.pY;
        m_pDimX = globals.pX;
        m_pDimY = globals.pY;

        m_pressure = new float[gridSize];
        m_uvel = new float[gridSize];
        m_vvel = new float[gridSize];
        
        std::size_t lastFound = 0;
        std::size_t found;
        while ((found = baseName.find("/", lastFound+1)) != std::string::npos) {
          lastFound = found;
        }
        if (lastFound > 0) {
          ++lastFound;
        }
        m_dirName = baseName.substr(0, lastFound);
        m_baseName = baseName.substr(lastFound);
        
        unsigned subGridSize = m_pointsPerDim * m_pointsPerDim;
        m_subsampleMatrix = new double[subGridSize * NUMBER_OF_BASIS_FUNCTIONS];
        double subGridSpacing = 1.0 / (m_pointsPerDim + 1);
        for (int bf = 0; bf < NUMBER_OF_BASIS_FUNCTIONS; ++bf) {
          for (int y = 0; y < m_pointsPerDim; ++y) {
            for (int x = 0; x < m_pointsPerDim; ++x) {
              double xi = (x+1) * subGridSpacing;
              double eta = (y+1) * subGridSpacing;
              m_subsampleMatrix[bf * subGridSize + (y * m_pointsPerDim + x)] = (*basisFunctions[bf])(xi, eta);
            }
          }
        }
        m_subsamples = new double[subGridSize * NUMBER_OF_QUANTITIES];
        memset(m_subsamples, 0, subGridSize * NUMBER_OF_QUANTITIES * sizeof(double));
    }
}

WaveFieldWriter::~WaveFieldWriter()
{
  if (!m_baseName.empty()) {
    if (m_rank == 0) {
        m_xdmf  << "    </Grid>" << std::endl
                << "  </Domain>" << std::endl
                << "</Xdmf>" << std::endl;
        m_xdmf.close();
    }
    

    delete[] m_subsamples;
    delete[] m_subsampleMatrix;
    delete[] m_pressure;
    delete[] m_uvel;
    delete[] m_vvel;
  }
}

void WaveFieldWriter::writeTimestep(double time, Grid<DegreesOfFreedom>& degreesOfFreedomGrid, bool forceWrite)
{
  if (!m_baseName.empty() && (time >= m_lastTime + m_interval || forceWrite)) {
    m_lastTime = time;
    
    std::stringstream pressureFileName, uvelFileName, vvelFileName, hdf5FileName;
    pressureFileName << m_baseName << "_pressure" << m_step << ".bin";
    uvelFileName << m_baseName << "_u" << m_step << ".bin";
    vvelFileName << m_baseName << "_v" << m_step << ".bin";
    hdf5FileName << m_baseName << "_" << m_step << ".h5";
    
    if (m_rank == 0) {
        m_xdmf  << "      <Grid Name=\"step_" << m_step << "\" GridType=\"Uniform\">" << std::setw(0) << std::endl
                << "        <Topology Reference=\"/Xdmf/Domain/Topology[1]\"/>" << std::endl
                << "        <Geometry Reference=\"/Xdmf/Domain/Geometry[1]\"/>" << std::endl
                << "        <Time Value=\"" << time << "\"/>" << std::endl
                << "        <Attribute Name=\"pressure\" Center=\"Node\">" << std::endl
                << "          <DataItem Format=\"HDF\" DataType=\"Float\" Precision=\"4\" Dimensions=\"" << m_pointsPerDim * degreesOfFreedomGrid.Y() << " " << m_pointsPerDim * degreesOfFreedomGrid.X() << "\">" << std::endl
                << "            " << hdf5FileName.str() << ":/pressure"<< std::endl
                << "          </DataItem>" << std::endl
                << "        </Attribute>" << std::endl
                << "        <Attribute Name=\"u\" Center=\"Node\">" << std::endl
                << "          <DataItem Format=\"HDF\" DataType=\"Float\" Precision=\"4\" Dimensions=\"" << m_pointsPerDim * degreesOfFreedomGrid.Y() << " " << m_pointsPerDim * degreesOfFreedomGrid.X() << "\">" << std::endl
                << "            " << hdf5FileName.str() << ":/u" << std::endl
                << "          </DataItem>" << std::endl
                << "       </Attribute>" << std::endl
                << "        <Attribute Name=\"v\" Center=\"Node\">" << std::endl
                << "          <DataItem Format=\"HDF\" DataType=\"Float\" Precision=\"4\" Dimensions=\"" << m_pointsPerDim * degreesOfFreedomGrid.Y() << " " << m_pointsPerDim * degreesOfFreedomGrid.X() << "\">" << std::endl
                << "            " << hdf5FileName.str() << ":/v"<< std::endl
                << "          </DataItem>" << std::endl
                << "        </Attribute>" << std::endl
                << "      </Grid>" << std::endl;
    }

    std::pair<int, int> yLims = degreesOfFreedomGrid.getYlimits();
    std::pair<int, int> xLims = degreesOfFreedomGrid.getXlimits();

    unsigned subGridSize = m_pointsPerDim * m_pointsPerDim;
    for (int y = yLims.first; y < yLims.second; ++y) {
      for (int x = xLims.first; x < xLims.second; ++x) {
    // for (int y = 0; y < degreesOfFreedomGrid.Y(); ++y) {
    //   for (int x = 0; x < degreesOfFreedomGrid.X(); ++x) {
        DGEMM(  subGridSize, NUMBER_OF_QUANTITIES, NUMBER_OF_BASIS_FUNCTIONS,
                1.0, m_subsampleMatrix, subGridSize,
                degreesOfFreedomGrid.get(x, y), NUMBER_OF_BASIS_FUNCTIONS,
                0.0, m_subsamples, subGridSize );

        for (int ysub = 0; ysub < m_pointsPerDim; ++ysub) {
          for (int xsub = 0; xsub < m_pointsPerDim; ++xsub) {
            unsigned subIndex = ysub * m_pointsPerDim + xsub;
            // unsigned targetIndex = (y*m_pointsPerDim+ysub)*m_pointsPerDim*degreesOfFreedomGrid.X() + (x*m_pointsPerDim+xsub);
            unsigned targetIndex    = ((y-yLims.first)*m_pointsPerDim+ysub)*m_pointsPerDim*m_pDimX + ((x-xLims.first)*m_pointsPerDim+xsub);
            m_pressure[targetIndex] = m_subsamples[0 * subGridSize + subIndex];
            m_uvel[targetIndex] = m_subsamples[1 * subGridSize + subIndex];
            m_vvel[targetIndex] = m_subsamples[2 * subGridSize + subIndex];
          }
        }
      }
    }

    /* WRITE HDF5 FILES */
    hid_t       file_id, filespace, memspace, plist_id, dataspace_id;  /* identifiers */
    hid_t       dataset_p_id, dataset_u_id, dataset_v_id;
    hsize_t dims[2], chunk_dims[2], offset[2], stride[2], count[2], block[2];

    // Specify dimensions
    dims[0] = m_pointsPerDim * degreesOfFreedomGrid.Y();
    dims[1] = m_pointsPerDim * degreesOfFreedomGrid.X();
    chunk_dims[0] = m_pDimY * m_pointsPerDim;
    chunk_dims[1] = m_pDimX * m_pointsPerDim;
    count[0] = 1;
    count[1] = 1;
    block[0] = chunk_dims[0];
    block[1] = chunk_dims[1];
    offset[0] = yLims.first * m_pointsPerDim;
    offset[1] = xLims.first * m_pointsPerDim;

    /*std::cout<<"SizesX: "<<chunk_dims[0]<<", "<<dims[0]<<std::endl;
    std::cout<<"SizesY: "<<chunk_dims[1]<<", "<<dims[1]<<std::endl;
    std::cout<<"OffsetX: "<<offset[0]<<", OffsetY: "<<offset[1]<<std::endl;*/

    // Set up file access property list with parallel I/O access
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    // Create a new file collectively and release property list identifier.
    file_id = H5Fcreate((m_dirName + hdf5FileName.str()).c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    // Create dataspace for dataset
    filespace = H5Screate_simple(2, dims, NULL);
    memspace = H5Screate_simple(2, chunk_dims, NULL);

    // Create chunked dataset.
    plist_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist_id, 2, chunk_dims);
    dataset_p_id = H5Dcreate(file_id, "/pressure", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    dataset_u_id = H5Dcreate(file_id, "/u", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    dataset_v_id = H5Dcreate(file_id, "/v", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    H5Pclose(plist_id);
    H5Sclose(filespace);

    // Write Pressure Stuff
    // Select Hyperslab in file 
    filespace = H5Dget_space(dataset_p_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, block);
    // Create property list for collective dataset write
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    H5Dwrite(dataset_p_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, m_pressure);
    // Close stuff
    H5Sclose(filespace);
    H5Pclose(plist_id);
    H5Dclose(dataset_p_id);

    // Write uvel Stuff
    // Select Hyperslab in file 
    filespace = H5Dget_space(dataset_u_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, block);
    // Create property list for collective dataset write
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    H5Dwrite(dataset_u_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, m_uvel);
    // Close stuff
    H5Sclose(filespace);
    H5Pclose(plist_id);
    H5Dclose(dataset_u_id);

    // Write vvel Stuff
    // Select Hyperslab in file
    filespace = H5Dget_space(dataset_v_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, block);
    // Create property list for collective dataset write
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    H5Dwrite(dataset_v_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, m_vvel);
    // Close stuff
    H5Sclose(filespace);
    H5Pclose(plist_id);
    H5Dclose(dataset_v_id);

    // Close/release resources.
    H5Sclose(memspace);
    H5Fclose(file_id);

    /////////////////////
    ////// Write HDF5 Single Process ///////
    hid_t       dataset_id;
    /*// Create file
    file_id = H5Fcreate((m_dirName + hdf5FileName.str()).c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    // Create a dataspace for the dataset
    dataspace_id = H5Screate_simple(2, dims, NULL);
    // Create dataset for Pressure
    dataset_id = H5Dcreate(file_id, "/pressure", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_pressure);
    // Create dataset for uvel
    dataset_id = H5Dcreate(file_id, "/u", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_uvel);
    // Create dataset for vvel
    dataset_id = H5Dcreate(file_id, "/v", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_vvel);
    // Cleanup
    status = H5Sclose(dataspace_id);
    status = H5Fclose(file_id);*/
    //////////////////////

    /* END WRITE HDF5 FILES */
    
    /*FILE* pressureFile = fopen((m_dirName + pressureFileName.str()).c_str(), "wb");
    fwrite(m_pressure, sizeof(float), subGridSize*degreesOfFreedomGrid.X()*degreesOfFreedomGrid.Y(), pressureFile);
    fclose(pressureFile);
    
    FILE* uFile = fopen((m_dirName + uvelFileName.str()).c_str(), "wb");
    fwrite(m_uvel, sizeof(float), subGridSize*degreesOfFreedomGrid.X()*degreesOfFreedomGrid.Y(), uFile);
    fclose(uFile);
    
    FILE* vFile = fopen((m_dirName + vvelFileName.str()).c_str(), "wb");
    fwrite(m_vvel, sizeof(float), subGridSize*degreesOfFreedomGrid.X()*degreesOfFreedomGrid.Y(), vFile);
    fclose(vFile);*/
    
    ++m_step;
  }
}
