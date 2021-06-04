# ExtrinsicGaugeEquivariantVectorGPs

JAX based library implementing the kernels for and experiments using extrinsic gauge equivariant vector field Gaussian Processes. Contains code to run the experiments in the paper `Vector-valued Gaussian Processes on RiemannianManifolds via Gauge-Equivariant Projected Kernels`.

## Library

The library contains the following functionality:

### Kernels

- An abstract kernel interface that is easy to extend.
- Implementation of any tensorflow-probabilty kernels
- A general implementation of scalar Matern and squared exponential kernels on compact manifolds, utilising the Laplacian eignefunctions and eigne values on the manifold.
- An abstract manifold and embedded manifold class which can be easily extended for user defined manifolds.
- Implementation of $\mathbb{R}^n$, $\mathbb{S}^1$ and $\mathbb{S}^2$ manifolds.
- Implementation of arbitrary product kernels.
- Implementation of extrinsic projection kernels for general manifolds.

### Gaussian processes

- Regular Gausssian proccess conditinng and sampling.
- Inducing point sparse GPs based on pathwise sampling.  


## Examples

The repository also contains 3 examples.

### Toy circle
`examples/circle_sine.py` is a simple demonstration of fitting a scalar function on a compact manifold.

### Dynamics learning
`examples/dynamical_system_non_conservative.py` is an example of using a manifold vector valued kernel to learn the dynamics of a non-conservative mechanical system. It contains comparisions of using a proper manifold kernel, and an innapropriate Euclidean kernel.

### Weather interpolation
`examples/wind_interpolation` contains an example of interpolating wind data over the globe using a vector valued manifold kernel. More instructions on running this example can be found in that directory.