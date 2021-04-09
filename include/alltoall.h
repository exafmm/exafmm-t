#ifndef alltoall_h
#define alltoall_h
#include "exafmm_t.h"
#include "mpi_utils.h"

namespace exafmm_t {
  //! Use alltoall to get recv count and calculate displacement from it
  void getCountAndDispl(std::vector<int> & sendCount, std::vector<int> & sendDispl,
                        std::vector<int> & recvCount, std::vector<int> & recvDispl) {
    MPI_Alltoall(&sendCount[0], 1, MPI_INT, &recvCount[0], 1, MPI_INT, MPI_COMM_WORLD);
    for (int irank=0; irank<MPISIZE-1; irank++) {
      sendDispl[irank+1] = sendDispl[irank] + sendCount[irank];
      recvDispl[irank+1] = recvDispl[irank] + recvCount[irank];
    }
  }

  //! Alltoallv for bodies
  template <typename T>
  void alltoallBodies(Bodies<T>& sendBodies, std::vector<int> & sendBodyCount, std::vector<int> & sendBodyDispl,
                      Bodies<T>& recvBodies, std::vector<int> & recvBodyCount, std::vector<int> & recvBodyDispl) {
    MPI_Datatype MPI_BODY;
    MPI_Type_contiguous(sizeof(sendBodies[0]), MPI_CHAR, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);
    recvBodies.resize(recvBodyDispl[MPISIZE-1]+recvBodyCount[MPISIZE-1]);
    MPI_Alltoallv(&sendBodies[0], &sendBodyCount[0], &sendBodyDispl[0], MPI_BODY,
                  &recvBodies[0], &recvBodyCount[0], &recvBodyDispl[0], MPI_BODY, MPI_COMM_WORLD);
  }
 
  //! Alltoallv for cells
  template <typename T>
  void alltoallCells(Nodes<T>& sendCells, std::vector<int>& sendCellCount, std::vector<int>& sendCellDispl,
                     Nodes<T>& recvCells, std::vector<int>& recvCellCount, std::vector<int>& recvCellDispl) {
    // Copy cells to cell bases, cell data
    std::vector<NodeBase> sendCellBases(sendCellDispl[MPISIZE-1] + sendCellCount[MPISIZE-1]);
    size_t nsurf = sendCells[0].up_equiv.size();
    std::vector<T> sendCellData(sendCellBases.size()*nsurf);

    for (int irank=0, ic=0; irank<MPISIZE; irank++) {
      for (int i=sendCellDispl[irank]; i<sendCellDispl[irank]+sendCellCount[irank]; i++) {
        sendCellBases[i].x = sendCells[i].x;
        sendCellBases[i].r = sendCells[i].r;
        sendCellBases[i].key = sendCells[i].key;
        sendCellBases[i].is_leaf = sendCells[i].is_leaf;
        sendCellBases[i].nsrcs = sendCells[i].nsrcs;
        for (int n=0; n<nsurf; n++) {
          sendCellData[ic++] = sendCells[i].up_equiv[n];
        }
      }
    }
    //! Send cell bases
    MPI_Datatype MPI_CELL_BASE;
    MPI_Type_contiguous(sizeof(sendCellBases[0]), MPI_CHAR, &MPI_CELL_BASE);
    MPI_Type_commit(&MPI_CELL_BASE);
    std::vector<NodeBase> recvCellBases(recvCellDispl[MPISIZE-1]+recvCellCount[MPISIZE-1]);
    MPI_Alltoallv(&sendCellBases[0], &sendCellCount[0], &sendCellDispl[0], MPI_CELL_BASE,
                  &recvCellBases[0], &recvCellCount[0], &recvCellDispl[0], MPI_CELL_BASE, MPI_COMM_WORLD);
    //! Send cell data
    MPI_Datatype MPI_CELL_DATA;
    MPI_Type_contiguous(sizeof(T)*nsurf, MPI_CHAR, &MPI_CELL_DATA);
    MPI_Type_commit(&MPI_CELL_DATA);
    std::vector<T> recvCellData(recvCellBases.size()*nsurf);
    MPI_Alltoallv(&sendCellData[0], &sendCellCount[0], &sendCellDispl[0], MPI_CELL_DATA,
                  &recvCellData[0], &recvCellCount[0], &recvCellDispl[0], MPI_CELL_DATA, MPI_COMM_WORLD);
    //! Copy cell bases, cell data to cells
    recvCells.resize(recvCellBases.size());
    for (int irank=0, ic=0; irank<MPISIZE; irank++) {
      for (int i=recvCellDispl[irank]; i<recvCellDispl[irank]+recvCellCount[irank]; i++) {
        recvCells[i].x = recvCellBases[i].x;
        recvCells[i].r = recvCellBases[i].r;
        recvCells[i].key = recvCellBases[i].key;
        recvCells[i].is_leaf = recvCellBases[i].is_leaf;
        recvCells[i].nsrcs = recvCellBases[i].nsrcs;

        recvCells[i].up_equiv.resize(nsurf, 0);
        // recvCells[i].L.resize(nsurf, 0);
        for (int n=0; n<nsurf; n++) {
          recvCells[i].up_equiv[n] += recvCellData[ic++];
        }
      }
    }
  }
}
#endif
