#ifndef geometry_h
#define geometry_h
#include "exafmm_t.h"

namespace exafmm_t {
  /**
   * @brief Given a box, calculate the coordinates of surface points.
   *
   * @param p Order of expansion.
   * @param r0 Half side length of the bounding box (root node).
   * @param level Level of the box. 
   * @param c Coordinates of the center of the box. 
   * @param alpha Ratio between the side length of surface box and original box.
   *              Use 2.95 for upward check and downward equivalent surface,
   *              use 1.05 for upward equivalent and downward check surface.
   * 
   * @return Vector of coordinates of surface points. 
   */
  RealVec surface(int p, real_t r0, int level, real_t* c, real_t alpha);

  /**
   * @brief Generate the convolution grid of a given box.
   *
   * @param p Order of expansion.
   * @param r0 Half side length of the bounding box (root node).
   * @param level Level of the box.
   * @param c Coordinates of the center of the box.
   *
   * @return Vector of coordinates of convolution grid.
   */
  RealVec convolution_grid(int p, real_t r0, int level, real_t* c);

  /**
   * @brief Generate the mapping from surface points to convolution grid used in FFT.
   *
   * @param p Order of expansion.
   * 
   * @return A mapping from upward equivalent surface point index to convolution grid index.
   */
  std::vector<int> generate_surf2conv_up(int p);

  /**
   * @brief Generate the mapping from surface points to convolution grid used in IFFT.
   *
   * @param p Order of expansion.
   * 
   * @return A mapping from downward check surface point index to convolution grid index.
   */
  std::vector<int> generate_surf2conv_dn(int p);

  /**
   * @brief Compute the hash value of a relative position (coordinates).
   *
   * @param coord Coordinates that represent a relative position.
   *
   * @return Hash value of the relative position.
   */
  int hash(ivec3& coord);

  /**
   * @brief Compute the coordinates of possible relative positions for operator t.
   *
   * @param max_r Max range.
   * @param min_r Min range.
   * @param step Step.
   * @param t Operator type (e.g. M2M, M2L)
   */
  void init_rel_coord(int max_r, int min_r, int step, Mat_Type t);

  //! Generate a map that maps indices of M2L_Type to indices of M2L_Helper_Type
  void generate_M2L_index_map();

  //! Compute the relative positions for all operators and generate M2L index mapping.
  void init_rel_coord();
} // end namespace
#endif
