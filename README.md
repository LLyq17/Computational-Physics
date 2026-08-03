# Computational-Physics
An undergraduate course I taught at Hebei Normal University

## 1. Introduction and why interpolation
Interpolation is a numerical technique used to estimate values of a function at points that are not directly given by the data. In scientific computing and experimental physics, measurements are often available only at discrete sample points, while the underlying behavior is continuous. Interpolation provides a practical way to reconstruct a smooth curve between known values, which is useful for data analysis, visualization, and numerical simulation.

Interpolation is especially important because many physical quantities are observed only at selected positions or times. Instead of treating the data as a set of isolated points, we can use interpolation to obtain a reasonable approximation of the function between those points. Compared with direct measurement, interpolation is often more efficient and helps us understand the trend of the data.

## 1.1 Lagrange interpolation
Lagrange interpolation constructs a polynomial that passes through all given data points. For points $(x_0,y_0), (x_1,y_1), \dots, (x_n,y_n)$, the interpolation polynomial is written as a sum of basis polynomials:

$$
P(x)=\sum_{i=0}^{n} y_i \prod_{j\ne i}\frac{x-x_j}{x_i-x_j}
$$

This method is straightforward and is often used when the number of sample points is small. Its advantage is that the polynomial is guaranteed to pass through every given point. However, for large numbers of data points, the resulting polynomial may become unstable and oscillatory.

## 1.2 Newton interpolation
Newton interpolation is another polynomial interpolation method. It expresses the interpolation polynomial in a form based on finite differences, making it convenient for adding new data points incrementally. In general, the Newton form can be written as:

$$
P(x)=a_0+a_1(x-x_0)+a_2(x-x_0)(x-x_1)+\cdots
$$

The coefficients are determined by divided differences, which provide a systematic way to build the polynomial. Newton interpolation is especially useful in practice because it allows the interpolation polynomial to be updated easily when more data become available.

## 1.3 Cubic spline interpolation
Cubic spline interpolation is a piecewise polynomial method in which the interval between neighboring data points is represented by a cubic polynomial. The pieces are connected so that the function values, first derivatives, and second derivatives are continuous at the knots. This produces a smooth curve and avoids the large oscillations that may appear in high-degree polynomial interpolation.

Because of its smoothness and stability, cubic spline interpolation is widely used in engineering, computer graphics, and scientific data fitting. It is often preferred when a visually smooth and physically reasonable curve is needed.
