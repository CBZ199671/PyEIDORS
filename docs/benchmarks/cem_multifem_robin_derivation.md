# Robin–transconductance realization of the complete electrode model

## Scope

This note proves the algebra used by the MFEM, FreeFEM, GetFEM, NGSolve, and
FEniCSx implementations in the controlled P1 experiment. The proof is a
finite-element identity, not an approximation to EIDORS. EIDORS' augmented
complete-electrode matrix and the Robin–transconductance construction are two
eliminations of the same discrete weak problem.

Let \(V_h\subset H^1(\Omega)\) be a conforming scalar finite-element space,
with nodal basis \(\{\phi_i\}_{i=1}^N\). The experiment fixes straight-sided
triangles and Lagrange P1, although the following algebra applies to any
conforming \(V_h\). There are \(L\) disjoint electrodes \(e_\ell\), positive
conductivity \(\sigma\), positive contact impedances \(z_\ell\), balanced
currents \(\mathbf 1^T I=0\), and the gauge \(\mathbf 1^T U=0\).

## Weak Robin problem

The complete electrode conditions are

\[
u+z_\ell\sigma\partial_nu=U_\ell\quad\text{on }e_\ell,
\qquad
\int_{e_\ell}\sigma\partial_nu\,ds=I_\ell.
\]

For a prescribed electrode-voltage vector \(U\), substitution of
\(\sigma\partial_nu=(U_\ell-u)/z_\ell\) into Green's identity gives

\[
\int_\Omega\sigma\nabla u_h\cdot\nabla v_h\,dx
+\sum_{\ell=1}^L\frac1{z_\ell}\int_{e_\ell}u_hv_h\,ds
=\sum_{\ell=1}^L\frac{U_\ell}{z_\ell}
\int_{e_\ell}v_h\,ds.
\]

Define the native FEM blocks

\[
K_{ij}=\int_\Omega\sigma\nabla\phi_i\cdot\nabla\phi_j\,dx,
\quad
B_{ij}=\sum_\ell\frac1{z_\ell}\int_{e_\ell}\phi_i\phi_j\,ds,
\]

\[
(C_+)_{i\ell}=\frac1{z_\ell}\int_{e_\ell}\phi_i\,ds,
\qquad
D_{\ell\ell}=\frac{|e_\ell|}{z_\ell}.
\]

Writing \(A_R=K+B\), the voltage-driven body solve is therefore

\[
A_Ru=C_+U. \tag{1}
\]

The integrated electrode currents are

\[
I_\ell=\frac1{z_\ell}
\left(|e_\ell|U_\ell-\int_{e_\ell}u_h\,ds\right),
\]

or, in matrix form,

\[
I=DU-C_+^Tu. \tag{2}
\]

## Transconductance Schur operator

Eliminating \(u\) from (1)–(2) gives

\[
I=TU,
\qquad
T=D-C_+^TA_R^{-1}C_+. \tag{3}
\]

Thus \(T\) is the electrode transconductance map: electrode voltage in,
integrated electrode current out. Constant voltage produces constant body
potential and zero current, so \(T\mathbf 1=0\). The singular full matrix
\(T\) must not be inverted.

Let \(Q\in\mathbb R^{L\times(L-1)}\) have orthonormal columns spanning
\(\mathbf 1^\perp\). For balanced currents, the unique zero-mean solution is

\[
(Q^TTQ)y=Q^TI,
\qquad U=Qy,
\qquad u=A_R^{-1}C_+U. \tag{4}
\]

## Positivity and uniqueness

Take the solution \(u=A_R^{-1}C_+U\) for any electrode vector \(U\). From
\(u^TA_Ru=u^TC_+U\), direct expansion gives

\[
\begin{aligned}
U^TTU
&=U^TDU-u^TC_+U\\
&=u^TKu+
\sum_{\ell=1}^L\frac1{z_\ell}
\int_{e_\ell}(u_h-U_\ell)^2\,ds
\ge 0.
\end{aligned} \tag{5}
\]

Equality requires zero body-field energy and zero contact mismatch. On a
connected domain this means \(u_h\) and all \(U_\ell\) equal one common
constant. Consequently

\[
\ker T=\operatorname{span}\{\mathbf1\},
\qquad Q^TTQ\ \text{is symmetric positive definite}. \tag{6}
\]

Equation (4) therefore has one and only one solution for every balanced
current vector.

## Exact equivalence to the augmented CEM matrix

With the EIDORS sign convention \(C_-=-C_+\), the gauge-constrained augmented
system is

\[
\begin{bmatrix}
A_R&C_-&0\\
C_-^T&D&\mathbf1\\
0&\mathbf1^T&0
\end{bmatrix}
\begin{bmatrix}u\\U\\\lambda\end{bmatrix}
=
\begin{bmatrix}0\\I\\0\end{bmatrix}. \tag{7}
\]

Its first block row gives (1). Substitution into the second block row gives
\(TU+\lambda\mathbf1=I\); projection by \(Q^T\) removes the gauge multiplier
and gives exactly (4). Conversely, a solution of (4) with
\(u=A_R^{-1}C_+U\) satisfies (7) with the compatible gauge multiplier.
Therefore the augmented and Robin–transconductance methods produce identical
discrete \(u\) and \(U\) in exact arithmetic.

## Controlled comparison consequence

The five general FEM packages need only their ordinary diffusion, boundary
mass, and boundary linear-form assembly mechanisms. No package-specific CEM
element is required. Fairness nevertheless requires each package to import
the identical labelled mesh, use straight P1 and real `float64`, assemble
\(K,B,C_+,D\) natively, and export the topology and blocks actually used.
EIDORS is the standard implementation comparator; exact rational arithmetic
and the independent disk solution provide the discrete and continuum truths.
No speed or timing quantity belongs to this experiment.
