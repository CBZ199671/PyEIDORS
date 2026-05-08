# p-refinement forward experiment

- Created: `2026-05-08T11:54:35.048483+00:00`
- Mesh dim: `2`
- Mesh refinement: `10`
- Orders: `1, 2, 3, 4`
- Reference for error columns: highest order in this run, `P4`; this is a convergence proxy, not an analytic truth.

| P order | potential dofs | conductivity dofs | solve seconds | rel L2 delta vs ref |
|---:|---:|---:|---:|---:|
| 1 | 1989 | 3832 | 0.0195592 | 0.015913 |
| 2 | 7809 | 3832 | 0.390205 | 0.00333454 |
| 3 | 17461 | 3832 | 0.553991 | 0.000881011 |
| 4 | 30945 | 3832 | 0.829689 | 0 |
