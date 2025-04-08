# Progressive-Quantum-Algorithm
The code implementation of the PQA algorithm.The core idea of PQA is to construct a subgraph that is likely to contain the MIS solution of the target graph and then solve the MIS problem on this subgraph to obtain an approximate solution. To construct such a subgraph, PQA starts with a small-scale initial subgraph and progressively expands its graph size utilizing heuristic expansion strategies. After each expansion, PQA solves the MIS problem on the newly generated subgraph. In each run, PQA repeats the expansion and solving process until a predefined stopping condition is reached. 

# Version
mindspore quantum 0.7.0
python 3.7
