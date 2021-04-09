#ifndef partition_h
#define partition_h
#include "alltoall.h"
#include "hilbert.h"

namespace exafmm_t {
  /**
   * @brief MPI_Allreduce local bounds to get global bounds.
   *
   * @tparam T Target's value type (real or complex).
   * @param sources Vector of sources.
   * @param targets Vector of targets.
   * @param x0 Coordinates of the center of the global bounding box.
   * @param r0 Radius of the bounding box.
   */
  template <typename T>
  void allreduceBounds(const Bodies<T>& sources, const Bodies<T>& targets, vec3& x0, real_t& r0) {
    vec3 localXmin, localXmax, globalXmin, globalXmax;
    localXmin = sources[0].X;
    localXmax = sources[0].X;
    for (size_t b=0; b<sources.size(); b++) {
      localXmin = min(sources[b].X, localXmin);
      localXmax = max(sources[b].X, localXmax);
    }
    for (size_t b=0; b<targets.size(); b++) {
      localXmin = min(targets[b].X, localXmin);
      localXmax = max(targets[b].X, localXmax);
    }
    MPI_Allreduce(&localXmin[0], &globalXmin[0], 3, MPI_REAL_T, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localXmax[0], &globalXmax[0], 3, MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    x0 = (globalXmax + globalXmin) / 2;
    r0 = fmax(max(x0-globalXmin), max(globalXmax-x0));
    r0 *= 1.00001;
  }

  /**
   * @brief Radix sort based on key and return permutation index.
   *
   * @param key Vector of keys.
   * @param value Vector of permutation indices (after sorting).
   * @param size Size of the vector.
   */
  void radixsort(std::vector<int> & key, std::vector<int> & value, int size) {
    const int bitStride = 8;
    const int stride = 1 << bitStride;
    const int mask = stride - 1;
    int maxKey = 0;
    int bucket[stride];
    std::vector<int> buffer(size);
    std::vector<int> permutation(size);
    for (int i=0; i<size; i++)
      if (key[i] > maxKey)
        maxKey = key[i];
    while (maxKey > 0) {
      for (int i=0; i<stride; i++)
        bucket[i] = 0;
      for (int i=0; i<size; i++)
        bucket[key[i] & mask]++;
      for (int i=1; i<stride; i++)
        bucket[i] += bucket[i-1];
      for (int i=size-1; i>=0; i--)
        permutation[i] = --bucket[key[i] & mask];
      for (int i=0; i<size; i++)
        buffer[permutation[i]] = value[i];
      for (int i=0; i<size; i++)
        value[i] = buffer[i];
      for (int i=0; i<size; i++)
        buffer[permutation[i]] = key[i];
      for (int i=0; i<size; i++)
        key[i] = buffer[i] >> bitStride;
      maxKey >>= bitStride;
    }
  }

  /**
   * @brief Partition sources and targets into different ranks.
   *
   * @tparam T Target's value type (real or complex).
   * @param sources Vector of sources.
   * @param targets Vector of targets.
   * @param x0 Coordinates of the center of the global bounding box.
   * @param r0 Radius of the bounding box.
   * @param offset Hilbert key bin for each rank. For example, the key of
   *               rank 0 ranges from offset[0] to offset[1].
   * @param level Determines how many bins are used in partitioning sources.
   */
  template <typename T>
  void partition(Bodies<T> & sources, Bodies<T> & targets,
                 vec3 x0, real_t r0, std::vector<int>& offset, int level) {
    const int nsrcs = sources.size();
    const int ntrgs = targets.size();
    const int numBins = 1 << 3 * level;
    std::vector<int> localHist(numBins, 0);
    // Get local histogram of hilbert key bins
    std::vector<int> src_key(nsrcs);
    std::vector<int> src_index(nsrcs);
    for (int b=0; b<nsrcs; b++) {
      ivec3 iX = get3DIndex(sources[b].X, level, x0, r0);
      src_key[b] = getKey(iX, level, false);  // without level offset
      src_index[b] = b;
      localHist[src_key[b]]++;
    }
    // Sort sources according to keys
    std::vector<int> src_key2 = src_key;
    radixsort(src_key, src_index, nsrcs);  // sort index based on key
    Bodies<T> src_buffer = sources;
    for (int b=0; b<nsrcs; b++) {
      sources[b] = src_buffer[src_index[b]];
      src_key[b] = src_key2[src_index[b]];
    }

    // Sort targets according to keys
    std::vector<int> trg_key(ntrgs);
    std::vector<int> trg_index(ntrgs);
    for (int b=0; b<nsrcs; b++) {
      ivec3 iX = get3DIndex(targets[b].X, level, x0, r0);
      trg_key[b] = getKey(iX, level, false);  // without level offset
      trg_index[b] = b;
    }
    std::vector<int> trg_key2 = trg_key;
    radixsort(trg_key, trg_index, ntrgs);
    Bodies<T> trg_buffer = targets;
    for (int b=0; b<ntrgs; b++) {
      targets[b] = trg_buffer[trg_index[b]];
      trg_key[b] = trg_key2[trg_index[b]];
    }

    // Get Global histogram of hilbert key bins
    std::vector<int> globalHist(numBins);
    MPI_Allreduce(&localHist[0], &globalHist[0], numBins, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Calculate offset of global histogram for each rank
    offset.resize(MPISIZE+1);
    offset[0] = 0;
    for (int i=0, irank=0, count=0; i<numBins; i++) {
      count += globalHist[i];
      if (irank * nsrcs < count) {
        offset[irank] = i;
        irank++;
      }
    }
    offset[MPISIZE] = numBins;
    std::vector<int> sendBodyCount(MPISIZE, 0);
    std::vector<int> recvBodyCount(MPISIZE, 0);
    std::vector<int> sendBodyDispl(MPISIZE, 0);
    std::vector<int> recvBodyDispl(MPISIZE, 0);
    // Use the offset as the splitter for partitioning
    for (int irank=0, b=0; irank<MPISIZE; irank++) {
      while (b < int(sources.size()) && src_key[b] < offset[irank+1]) {
        sendBodyCount[irank]++;
        b++;
      }
    }
    // Use alltoall to get recv count and calculate displacement from it
    getCountAndDispl(sendBodyCount, sendBodyDispl, recvBodyCount, recvBodyDispl);
    src_buffer.resize(recvBodyDispl[MPISIZE-1]+recvBodyCount[MPISIZE-1]);
    // Alltoallv for sources (defined in alltoall.h)
    alltoallBodies(sources, sendBodyCount, sendBodyDispl, src_buffer, recvBodyCount, recvBodyDispl);
    sources = src_buffer;

    // alltoallv for targets
    std::fill(sendBodyCount.begin(), sendBodyCount.end(), 0);
    std::fill(recvBodyCount.begin(), recvBodyCount.end(), 0);
    std::fill(sendBodyDispl.begin(), sendBodyDispl.end(), 0);
    std::fill(recvBodyDispl.begin(), recvBodyDispl.end(), 0);
    for (int irank=0, b=0; irank<MPISIZE; irank++) {
      while (b < int(targets.size()) && trg_key[b] < offset[irank+1]) {
        sendBodyCount[irank]++;
        b++;
      }
    }
    getCountAndDispl(sendBodyCount, sendBodyDispl, recvBodyCount, recvBodyDispl);
    trg_buffer.resize(recvBodyDispl[MPISIZE-1]+recvBodyCount[MPISIZE-1]);
    alltoallBodies(targets, sendBodyCount, sendBodyDispl, trg_buffer, recvBodyCount, recvBodyDispl);
    targets = trg_buffer;
  }
}
#endif
