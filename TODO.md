###TODO
-------------

- [x] Push history of pvfmm to exafmm-t (Tingyu)
- [x] remove CollectNodeData and node_lst from fmm_tree (Tingyu)
- [x] remove getPermR, getPermC from precomputation header (Chenwu)
- [x] Remove unnecessary "{}" (Chenwu)
- [x] Use code beautifier (Chenwu)
- [ ] Simplify tree structure (Tingyu)
- [ ] Unify node data types (Tingyu)
- [x] Vector -> std::vector with customized allocator (Tingyu)
- [ ] split fmm_tree header -> tree_construction, eval_setup, eval (Tingyu)
  - remove circular dependency of classes, use forward declaration if necessary
- [ ] get 2:1 balanced tree working, refer to pvfmm's repo, test with plummer distribution (Tingyu)

- [ ] change macros for constants to global const variables
- [ ] define global n1,n2,n3,n3\_,fftsize, replace the local copys with the global ones in M2L precomputation & evaluation
- [ ] remove node_lst from FMM_Tree, organize traversal calls in CollectNodeData
- [ ] test whether fft_in needs to be aligned in M2L (performance-wise)
- [ ] compare & update benchmarks in TIME
- [ ] naming conventions
