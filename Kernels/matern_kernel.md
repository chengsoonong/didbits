# Matern kernel with Gaussian processes

## Recent papers

This paper looks at the Laplacian of a weighted graph, and defines a Gaussian process with a Matern kernel on it. Won best paper at AISTATS.
https://avt.im/publications/2020/10/30/Graph-Matern-GP
If your graph nodes live in Euclidean space (e.g. they are physical spatial locations), then one can kernelise the node features directly.
https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-4/Isotropic-covariance-functions-on-graphs-and-their-edges/10.1214/19-AOS1896.short
In the other direction, you can consider the graph Matern kernel to be a special case of the Matern kernel on Riemanian manifolds (when you use a manifold learning method that returns a graph).
https://arxiv.org/abs/2006.10160

## Kernels and Covariances

http://gpss.cc/gpws14/KernelDesign.pdf

- Given an index set X with elements x_i
- A kernel is between elements of X, i.e. k(x_i, x_j)
- A process Z is indexed by x_i, i.e. has elements Z(x_i)
- A Gaussian process has covariance cov(Z(x_i), Z(x_j))
- (under some conditions) k(x_i, x_j) = cov(Z(x_i), Z(x_j))

Definition of a reproducing kernel Hilbert space, Section 1.1.1. of
http://www.ong-home.my/papers/ong05thesis-accepted.pdf


## Motivation and History

- https://stats.stackexchange.com/questions/322523/what-is-the-rationale-of-the-mat%C3%A9rn-covariance-function
- https://sites.stat.washington.edu/NRCSE/pdf/trs81.pdf
- https://www.jstor.org/stable/pdf/2332724.pdf

## References

- Kernels and stochastic processes, Berliner and Thomas-Agnan, 2014 https://link.springer.com/book/10.1007%2F978-1-4419-9096-9
- Bochner's theorem, Theorem 2.9 in  van den Berg, C., Christensen, J. P. R., Ressel, P. Harmonic Analysis on Semigroups, 1984
- Geostatistical process, Chapter 4 in Cressie, Wikle, Statistics for Spatiotemporal data, 2011
- 1D Gaussian Markov process, Appendix B of Rasmussen, Williams, Gaussian processes, 2005, http://www.gaussianprocess.org/gpml/chapters/RWB.pdf
- Kernels, Chapter 1 of Wahba, Spline models for Observational data, 1990 https://doi.org/10.1137/1.9781611970128
