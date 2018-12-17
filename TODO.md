###TODO
-------------

- [ ] Non-parent-level M2L (Tingyu)
- [ ] Add Helmholtz kernel (Tingyu, Chenwu)
- [ ] Separate sources/target (Tingyu)
- [ ] Write wrapper for BEM (Tingyu)
- [ ] Store precomputation matrix (Tingyu)
- [ ] Laplace P2P on GPU (Elket)
- [ ] Compare exafmm vs. exafmm-t (Rio)

###LONG TERM
-------------
- [ ] GPU kernels
- [ ] MPI
- [ ] Stokes
- [ ] CI, unit tests
- [ ] configure files

## Naming conventions:
- M2M_U / M2M_V -> UC2E_U / UC2E_V
- L2L_U / L2L_V -> DC2E_U / DC2E_V
- precomputation matrix: matrix_TYPE
- function name: build_tree(args) (excludes kernels)
- Typename: 
  - RealVec, ComplexVec
  - except. simdvec, ivec3 (user-defined vec from vec.h)
- member name: is_leaf
  - except. numTargets -> ntrgs, numNodes -> nnodes
  - upward prefix -> up_check / up_equiv
  - downward prefix -> dn_check
  - source / target prefix -> src_coord / trg_value
  - Node::node_id -> idx_M2L
  - Node::P2Llist -> P2L_list
- vector-type: Keys, Nodes, Bodies
- global:
  - const: ALLUPPERCASE, Xmin0 -> XMIN0
  - vector: lower_case, ex. rel_coord, matrix_TYPE
- M2LData:
  - fft_vec -> fft_offset / ifft_offset in time domain
  - ifft_scl -> ifft_scale
  - interac_vec -> interaction_offset_f
  - interac_dsp -> interaction_count_offset (offset of interaction counts)
- variable names in time / frequency domain:
  - time: up_equiv
  - frequency: up_equiv_f
  - M2L related functions: inputdata -> fft_in (what it actually is)
- surface coord / convolution grid
  - upward check surface: up_check_surf
  - convolution grid: variable:conv_grid, function: convolution_grid()
- misc
  - PrecompM2M -> precompute_M2M()
  - precomputation local variable: M_c2e -> matrix_c2e
  - buffer0, buffer1
  - allUpwardEquiv -> all_up_equiv
  - FFT function names:
    - FFT_UpEquiv -> fft_up_equiv
    - FFT_Check2Equiv -> ifft_dn_check
    - M2LListHadamard -> hadamard_product 
- adjust whitespaces, alignments
