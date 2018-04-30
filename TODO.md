###TODO
-------------

- [x] Remove mutable (Chenwu)
- [ ] Check grad answer in CheckFMMOutput (Chenwu)
- [ ] Merge potential and grad (Chenwu)
- [x] Move P2M to Kernel class (Chenwu)
- [x] Move M2M to Kernel class (Chenwu)
- [x] Move L2L to Kernel class (Chenwu)
- [ ] Move L2P to Kernel class (Chenwu)
- [ ] Move P2L to Kernel class (Chenwu)
- [ ] Move M2P to Kernel class (Chenwu)
- [ ] Move P2P to Kernel class (Chenwu)
- [ ] Move M2L to Kernel class (Chenwu)
- [x] trg/src_coord/value > pt_coord/pt_trg/pt_src (Tingyu)
- [ ] Unify node data types (Tingyu)
- [ ] Simplify tree structure (Tingyu)
- [ ] split fmm_tree header -> tree_construction, eval_setup, eval (Tingyu)
  - remove circular dependency of classes, use forward declaration if necessary
- [ ] get 2:1 balanced tree working, refer to pvfmm's repo, test with plummer distribution (Tingyu)

- [ ] change macros for constants to global const variables
- [ ] define global n1,n2,n3,n3\_,fftsize, replace the local copys with the global ones in M2L precomputation & evaluation
- [ ] remove node_lst from FMM_Tree, organize traversal calls in CollectNodeData
- [ ] test whether fft_in needs to be aligned in M2L (performance-wise)
- [ ] naming conventions
- [ ] rename headers, modify include guards
