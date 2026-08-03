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


### Code example: linear Lagrange interpolation for $\sin(x)$
Here is a simple Python example using linear Lagrange interpolation on the function $\sin(x)$.

```python
import numpy as np
import matplotlib.pyplot as plt


def lagrange_linear_interpolation(x, x0, x1, y0, y1):
    l0 = (x - x1) / (x0 - x1)
    l1 = (x - x0) / (x1 - x0)
    return y0 * l0 + y1 * l1

x_true = np.linspace(-np.pi, np.pi, 400)
y_true = np.sin(x_true)

x_nodes = np.array([-np.pi / 2, 0.0, np.pi / 2, np.pi])
y_nodes = np.sin(x_nodes)

x_interp = np.linspace(-np.pi, np.pi, 400)
y_interp = np.empty_like(x_interp)

for i in range(len(x_nodes) - 1):
    mask = (x_interp >= x_nodes[i]) & (x_interp <= x_nodes[i + 1])
    y_interp[mask] = lagrange_linear_interpolation(
        x_interp[mask], x_nodes[i], x_nodes[i + 1], y_nodes[i], y_nodes[i + 1]
    )

plt.figure(figsize=(8, 4.5))
plt.plot(x_true, y_true, label="sin(x)", color="tab:blue", linewidth=2)
plt.plot(x_interp, y_interp, label="Linear Lagrange interpolation", color="tab:red", linestyle="--", linewidth=2)
plt.scatter(x_nodes, y_nodes, color="black", zorder=5, label="Sample points")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Lagrange linear interpolation for sin(x)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
```

The corresponding figure is stored in [examples/lagrange_linear_interpolation.png](examples/lagrange_linear_interpolation.png).


## 1.2 Newton interpolation
Newton interpolation is another polynomial interpolation method. It expresses the interpolation polynomial in a form based on finite differences, making it convenient for adding new data points incrementally. In general, the Newton form can be written as:

$$
P(x)=a_0+a_1(x-x_0)+a_2(x-x_0)(x-x_1)+\cdots
$$

The coefficients are determined by divided differences, which provide a systematic way to build the polynomial. Newton interpolation is especially useful in practice because it allows the interpolation polynomial to be updated easily when more data become available.

## 1.3 Cubic spline interpolation
Cubic spline interpolation is a piecewise polynomial method in which the interval between neighboring data points is represented by a cubic polynomial. The pieces are connected so that the function values, first derivatives, and second derivatives are continuous at the knots. This produces a smooth curve and avoids the large oscillations that may appear in high-degree polynomial interpolation.

Because of its smoothness and stability, cubic spline interpolation is widely used in engineering, computer graphics, and scientific data fitting. It is often preferred when a visually smooth and physically reasonable curve is needed.
