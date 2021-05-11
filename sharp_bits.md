# Sharp Bits

- Silent error when you try to request higher index eigenfunctions than have been precomputed for product manifolds. Due to a Jax issue with error checking in JIT code. Not an issue if you use the kernel classes only.
- Precomupting eigenvalues for the product manifolds is exponential in the number of manifolds producted. Better soultion needed.
- Do not do euclidean_kernel x projection_kernel(manifold_kernel x euclidean_kernel), it will be really slow as right now it will do a cube of basis functions, not just a square.