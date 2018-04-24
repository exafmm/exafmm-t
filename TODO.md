###TODO
-------------

- [ ] Remove mutable
- [ ] Check grad answer in CheckFMMOutput (Chenwu)
- [ ] Merge potential and grad (Chenwu)
- [ ] Move P2M, M2M, M2L, L2L, L2P to Kernel class (Chenwu)
- [ ] trg/src_coord/value > pt_coord/pt_trg/pt_src
- [ ] Unify node data types (Tingyu)
- [ ] Simplify tree structure (Tingyu)
- [ ] split fmm_tree header -> tree_construction, eval_setup, eval (Tingyu)
  - remove circular dependency of classes, use forward declaration if necessary
- [ ] get 2:1 balanced tree working, refer to pvfmm's repo, test with plummer distribution (Tingyu)

- [ ] change macros for constants to global const variables
- [ ] define global n1,n2,n3,n3\_,fftsize, replace the local copys with the global ones in M2L precomputation & evaluation
- [ ] remove node_lst from FMM_Tree, organize traversal calls in CollectNodeData
- [ ] test whether fft_in needs to be aligned in M2L (performance-wise)
- [ ] compare & update benchmarks in TIME
- [ ] naming conventions
- [ ] rename headers, modify include guards
