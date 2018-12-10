# exafmm-t debug

case:
N = 10000, ncrit = 300, max_level=2, Random uniform distribution

checklist:
- [ ] check precomputation matrix
- [x] check particle coordinates and charges
- [ ] check P2P result
- [ ] compare 2:1 tree and nodes
- [ ] check leafs' upward_equiv after P2M
- [ ] check nonleafs's upward_equiv after M2M 
- [ ] check interaction list
- [ ] check downward check after M2L
- [ ] check downward check after L2L
- [ ] check potential after L2P & P2P
