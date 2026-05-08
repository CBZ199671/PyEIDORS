# p-refinement forward experiment

- Created: `2026-05-08T11:52:09.311835+00:00`
- Mesh dim: `2`
- Mesh refinement: `10`
- Orders: `1, 2, 3, 4`
- Reference for error columns: highest order in this run, `P4`; this is a convergence proxy, not an analytic truth.

| P order | potential dofs | conductivity dofs | solve seconds | rel L2 delta vs ref |
|---:|---:|---:|---:|---:|
| 1 | 121 | 200 | 0.00661314 | 0.730048 |
| 2 | 441 | 200 | 0.00604637 | 0.457747 |
| 3 | 961 | 200 | 0.00880525 | 0.207836 |
| 4 | 1681 | 200 | 0.0171384 | 0 |
