# Computational-Physics
An undergraduate course I taught at Hebei Normal University (see:https://llyq17.github.io/Computational-Physics/README.html)

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
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange


# 如果 Figures 文件夹不存在，则自动创建
os.makedirs("Figures", exist_ok=True)


def linear_interpolation(x_nodes, y_nodes, x):
    """
    一次拉格朗日插值，即分段线性插值。

    对每个计算点，选取其所在区间的两个相邻节点，
    再调用 scipy.interpolate.lagrange 构造一次多项式。
    """

    y_linear = np.empty_like(x, dtype=float)
    n_nodes = len(x_nodes)

    for k, x_value in enumerate(x):
        if x_value <= x_nodes[0]:
            start = 0
        elif x_value >= x_nodes[-1]:
            start = n_nodes - 2
        else:
            right = np.searchsorted(x_nodes, x_value, side="right")
            start = right - 1

        polynomial = lagrange(
            x_nodes[start:start + 2],
            y_nodes[start:start + 2]
        )
        y_linear[k] = polynomial(x_value)

    return y_linear

# 在区间 [0, 2π] 上构造 9 个等距插值节点
x_nodes = np.linspace(0.0, 2.0 * np.pi, 9)
y_nodes = np.sin(x_nodes)

# 构造致密网格，用于绘图和误差计算
x = np.linspace(0.0, 2.0 * np.pi, 2000)
y_true = np.sin(x)

# 一次拉格朗日插值
y_linear = linear_interpolation(x_nodes, y_nodes, x)
error_linear = np.abs(y_linear - y_true)

# 一次拉格朗日插值曲线
plt.figure(figsize=(10, 6))
plt.scatter(x_nodes, y_nodes, s=60, label="Nine data points", zorder=3)
plt.plot(x, y_linear, linewidth=2, label="Degree-1 Lagrange interpolation")
plt.plot(x, y_true, "--", linewidth=2, label="True function: sin(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Piecewise Linear Lagrange Interpolation")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig("Figures/linear_lagrange.png", dpi=300, bbox_inches="tight")
plt.show()

# 一次拉格朗日插值误差
plt.figure(figsize=(10, 6))
plt.plot(x, error_linear, "--", linewidth=2, label="Absolute error")
plt.xlabel("x")
plt.ylabel(r"$|P_1(x)-\sin(x)|$")
plt.title("Error of Piecewise Linear Lagrange Interpolation")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig("Figures/linear_lagrange_error.png", dpi=300, bbox_inches="tight")
plt.show()
```
The resulting plot is shown below:

![](Figures/linear_lagrange.png)
![](Figures/linear_lagrange_error.png)
The corresponding figure is stored in [Figures/linear_lagrange.png](Figures/linear_lagrange.png). and [Figures/linear_lagrange_error.png](Figures/linear_lagrange_error.png)


### Code example: quadratic Lagrange interpolation for $\sin(x)$
Here is a simple Python example using quadratic Lagrange interpolation on the function $\sin(x)$.

```python
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange


# 如果 Figures 文件夹不存在，则自动创建
os.makedirs("Figures", exist_ok=True)


def quadratic_interpolation(x_nodes, y_nodes, x):
    """
    二次拉格朗日插值，即局部抛物线插值。

    对每个计算点，选取附近的三个连续节点，
    再调用 scipy.interpolate.lagrange 构造二次多项式。
    """

    y_quadratic = np.empty_like(x, dtype=float)
    n_nodes = len(x_nodes)

    for k, x_value in enumerate(x):
        nearest = int(np.argmin(np.abs(x_nodes - x_value)))
        start = nearest - 1
        start = max(start, 0)
        start = min(start, n_nodes - 3)

        polynomial = lagrange(
            x_nodes[start:start + 3],
            y_nodes[start:start + 3]
        )
        y_quadratic[k] = polynomial(x_value)

    return y_quadratic

# 在区间 [0, 2π] 上构造 9 个等距插值节点
x_nodes = np.linspace(0.0, 2.0 * np.pi, 9)
y_nodes = np.sin(x_nodes)

# 构造致密网格，用于绘图和误差计算
x = np.linspace(0.0, 2.0 * np.pi, 2000)
y_true = np.sin(x)

# 二次拉格朗日插值
y_quadratic = quadratic_interpolation(x_nodes, y_nodes, x)
error_quadratic = np.abs(y_quadratic - y_true)

# 二次拉格朗日插值曲线
plt.figure(figsize=(10, 6))
plt.scatter(x_nodes, y_nodes, s=60, label="Nine data points", zorder=3)
plt.plot(x, y_quadratic, linewidth=2, label="Degree-2 Lagrange interpolation")
plt.plot(x, y_true, "--", linewidth=2, label="True function: sin(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Local Quadratic Lagrange Interpolation")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig("Figures/quadratic_lagrange.png", dpi=300, bbox_inches="tight")
plt.show()

# 二次拉格朗日插值误差
plt.figure(figsize=(10, 6))
plt.plot(x, error_quadratic, "--", linewidth=2, label="Absolute error")
plt.xlabel("x")
plt.ylabel(r"$|P_2(x)-\sin(x)|$")
plt.title("Error of Local Quadratic Lagrange Interpolation")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig("Figures/quadratic_lagrange_error.png", dpi=300, bbox_inches="tight")
plt.show()
```
The resulting plot is shown below:

![](Figures/quadratic_lagrange.png)
![](Figures/quadratic_lagrange_error.png)
The corresponding figure is stored in [Figures/quadratic_lagrange.png](Figures/quadratic_lagrange.png). and [Figures/quadratic_lagrange_error.png](Figures/quadratic_lagrange_error.png)

### Code example: Lagrange interpolation for $\sin(x)$
Here is a simple Python example using Lagrange interpolation on the function $\sin(x)$.

```python
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange


# 如果 Figures 文件夹不存在，则自动创建
os.makedirs("Figures", exist_ok=True)

# 在区间 [0, 2π] 上构造 9 个等距插值节点
x_nodes = np.linspace(0.0, 2.0 * np.pi, 9)
y_nodes = np.sin(x_nodes)

# 构造致密网格，用于绘图和误差计算
x = np.linspace(0.0, 2.0 * np.pi, 2000)
y_true = np.sin(x)

# 八次全局拉格朗日插值
polynomial_degree8 = lagrange(x_nodes, y_nodes)
y_degree8 = polynomial_degree8(x)
error_degree8 = np.abs(y_degree8 - y_true)

# 八次全局拉格朗日插值曲线
plt.figure(figsize=(10, 6))
plt.scatter(x_nodes, y_nodes, s=60, label="Nine data points", zorder=3)
plt.plot(x, y_degree8, linewidth=2, label="Degree-8 Lagrange interpolation")
plt.plot(x, y_true, "--", linewidth=2, label="True function: sin(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Degree-8 Global Lagrange Interpolation")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig("Figures/degree8_lagrange.png", dpi=300, bbox_inches="tight")
plt.show()

# 八次全局拉格朗日插值误差
plt.figure(figsize=(10, 6))
plt.plot(x, error_degree8, "--", linewidth=2, label="Absolute error")
plt.xlabel("x")
plt.ylabel(r"$|P_8(x)-\sin(x)|$")
plt.title("Error of Degree-8 Global Lagrange Interpolation")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig("Figures/degree8_lagrange_error.png", dpi=300, bbox_inches="tight")
plt.show()

```
The resulting plot is shown below:

![](Figures/degree8_lagrange.png)
![](Figures/degree8_lagrange_error.png)
The corresponding figure is stored in [Figures/degree8_lagrange.png](Figures/degree8_lagrange.png). and [Figures/degree8_lagrange_error.png](Figures/degree8_lagrange_error.png)

The notebook [Code/lagrange_interpolation.ipynb](Code/lagrange_interpolation.ipynb) also compares several approaches,

The resulting plot is shown below:

![](Figures/lagrange_comparison.png)
![](Figures/lagrange_error_comparison.png)

The resulting comparison figure is stored in [Figures/lagrange_comparison.png](Figures/lagrange_comparison.png). and [Figures/lagrange_error_comparison.png](Figures/lagrange_error_comparison.png).

## 1.2 Newton interpolation
Newton interpolation is another polynomial interpolation method. It expresses the interpolation polynomial in a form based on finite differences, making it convenient for adding new data points incrementally. In general, the Newton form can be written as:

$$
P(x)=a_0+a_1(x-x_0)+a_2(x-x_0)(x-x_1)+\cdots+a_n(x-x_0)(x-x_1)\cdots(x-x_{n-1}),
$$

where the coefficients $a_k$ are determined by divided differences:

$$
a_0 = f(x_0),
$$

$$
a_1 = \frac{f(x_1)-f(x_0)}{x_1-x_0},
$$

$$
a_2 = \frac{\frac{f(x_2)-f(x_1)}{x_2-x_1}-\frac{f(x_1)-f(x_0)}{x_1-x_0}}{x_2-x_0},
$$

and similarly for higher orders. This formulation is especially useful because when a new data point is added, only the new divided differences are needed, so the interpolation polynomial can be updated efficiently.

### Code example: Newton interpolation for $\sin(x)$
Here is a simple Python example using Newton interpolation on the function $\sin(x)$.

```python
import os

import numpy as np
import matplotlib.pyplot as plt


# 如果 Figures 文件夹不存在，则自动创建
os.makedirs("Figures", exist_ok=True)


def divided_difference(x_nodes, y_nodes):
    """
    计算牛顿插值的差商表和差商系数。

    返回：
    table       完整差商表
    coefficients  牛顿插值多项式的系数
    """

    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)

    n = len(x_nodes)

    # 差商表第一列为函数值
    table = np.zeros((n, n), dtype=float)
    table[:, 0] = y_nodes

    # 逐阶计算差商
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (
                table[i + 1, j - 1] - table[i, j - 1]
            ) / (
                x_nodes[i + j] - x_nodes[i]
            )

    # 牛顿插值系数位于差商表第一行
    coefficients = table[0, :].copy()

    return table, coefficients


def newton_interpolation(x_nodes, coefficients, x):
    """
    根据牛顿差商系数计算插值多项式。

    采用嵌套乘法形式计算，避免直接展开高次多项式。
    """

    x_nodes = np.asarray(x_nodes, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    x = np.asarray(x, dtype=float)

    # 从最高阶系数开始进行嵌套计算
    y = np.full_like(x, coefficients[-1], dtype=float)

    for k in range(len(coefficients) - 2, -1, -1):
        y = coefficients[k] + (x - x_nodes[k]) * y

    return y


# 在区间 [0, 2π] 上构造 9 个等距插值节点
x_nodes = np.linspace(
    0.0,
    2.0 * np.pi,
    9
)

# 节点对应的函数值
y_nodes = np.sin(x_nodes)


# 构造致密网格，用于绘图和误差计算
x = np.linspace(
    0.0,
    2.0 * np.pi,
    2000
)

# 原函数真值
y_true = np.sin(x)


# 计算差商表和牛顿插值系数
difference_table, coefficients = divided_difference(
    x_nodes,
    y_nodes
)

# 计算牛顿插值结果
y_newton = newton_interpolation(
    x_nodes,
    coefficients,
    x
)

# 计算绝对误差
error_newton = np.abs(
    y_newton - y_true
)


print("牛顿插值系数：")
for i, value in enumerate(coefficients):
    print(f"a[{i}] = {value:.12e}")

print()
print(
    "牛顿插值最大绝对误差：",
    np.max(error_newton)
)


plt.figure(figsize=(10, 6))

plt.scatter(
    x_nodes,
    y_nodes,
    s=60,
    label="Nine data points",
    zorder=3
)

plt.plot(
    x,
    y_newton,
    linewidth=2,
    label="Newton interpolation"
)

plt.plot(
    x,
    y_true,
    "--",
    linewidth=2,
    label="True function: sin(x)"
)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Newton Interpolation of sin(x)")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    "Figures/newton_interpolation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 牛顿插值误差
# ==========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    error_newton,
    "--",
    linewidth=2,
    label="Absolute error"
)

plt.xlabel("x")
plt.ylabel(r"$|P_8(x)-\sin(x)|$")
plt.title("Absolute Error of Newton Interpolation")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    "Figures/newton_interpolation_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
```
The resulting plot is shown below:

![](Figures/newton_interpolation.png)
![](Figures/newton_interpolation.png)
The corresponding figure is stored in [Figures/newton_interpolation.png](Figures/newton_interpolation.png). and [Figures/newton_interpolation_error.png](Figures/newton_interpolation_error.png)


## 1.3 Cubic spline interpolation
Cubic spline interpolation is a piecewise polynomial method in which the interval between neighboring data points is represented by a cubic polynomial. On each subinterval $[x_i,x_{i+1}]$, one writes

$$
S_i(x)=a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3,
$$

where $a_i,b_i,c_i,d_i$ are coefficients determined by the interpolation conditions. The spline is constructed so that the function values, first derivatives, and second derivatives are continuous at the knots:

$$
S_i(x_i)=f(x_i), \qquad S_{i+1}(x_{i+1})=f(x_{i+1}),
$$

$$
S_i'(x_{i+1})=S_{i+1}'(x_{i+1}), \qquad S_i''(x_{i+1})=S_{i+1}''(x_{i+1}).
$$

This produces a smooth curve and avoids the large oscillations that may appear in high-degree polynomial interpolation. Because of its smoothness and stability, cubic spline interpolation is widely used in engineering, computer graphics, and scientific data fitting. It is often preferred when a visually smooth and physically reasonable curve is needed.

### Code example: Cubic spline interpolation for $\sin(x)$
Here is a simple Python example using Cubic spline interpolation on the function $\sin(x)$.

```python
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# 如果 Figures 文件夹不存在，则自动创建
os.makedirs("Figures", exist_ok=True)

# 在区间 [0, 2π] 上构造 9 个等距插值节点
x_nodes = np.linspace(
    0.0,
    2.0 * np.pi,
    9
)

# 节点对应的函数值
y_nodes = np.sin(x_nodes)


# 构造致密网格，用于绘图和误差计算
x = np.linspace(
    0.0,
    2.0 * np.pi,
    2000
)

# 原函数真值
y_true = np.sin(x)


# 构造自然三次样条插值函数
# bc_type="natural" 表示左右端点的二阶导数为 0
spline = CubicSpline(
    x_nodes,
    y_nodes,
    bc_type="natural"
)

# 计算样条插值结果
y_spline = spline(x)

# 计算绝对误差
error_spline = np.abs(
    y_spline - y_true
)

print(
    "三次样条插值最大绝对误差：",
    np.max(error_spline)
)

# ==========================================================
# 三次样条插值曲线
# ==========================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    x_nodes,
    y_nodes,
    s=60,
    label="Nine data points",
    zorder=3
)

plt.plot(
    x,
    y_spline,
    linewidth=2,
    label="Natural cubic spline interpolation"
)

plt.plot(
    x,
    y_true,
    "--",
    linewidth=2,
    label="True function: sin(x)"
)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Natural Cubic Spline Interpolation of sin(x)")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    "Figures/cubic_spline_interpolation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 三次样条插值误差
# ==========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    error_spline,
    "--",
    linewidth=2,
    label="Absolute error"
)

plt.xlabel("x")
plt.ylabel(r"$|S(x)-\sin(x)|$")
plt.title("Absolute Error of Natural Cubic Spline Interpolation")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    "Figures/cubic_spline_interpolation_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
```
The resulting plot is shown below:

![](Figures/cubic_spline_interpolation.png)
![](Figures/cubic_spline_interpolation_error.png)
The corresponding figure is stored in [Figures/cubic_spline_interpolation.png](Figures/cubic_spline_interpolation.png). and [Figures/cubic_spline_interpolation_error.png](Figures/cubic_spline_interpolation_error.png)


## 1.4 Runge's phenomenon

A classic warning about high-degree polynomial interpolation is Runge's phenomenon. If we interpolate the function

$$
f(x)=\frac{1}{1+x^2},\qquad x\in[-5,5],
$$

using many equally spaced nodes, the interpolating polynomial may oscillate strongly near the endpoints. This shows that increasing the degree of the polynomial does not always improve the approximation.

The notebook [Code/runge_interpolation_comparison.ipynb](Code/runge_interpolation_comparison.ipynb) compares several approaches:

The resulting comparison figure is stored in [Figures/Runge_comparison.png](Figures/Runge_comparison.png).

This example illustrates why piecewise and spline-based methods are often more stable than one single high-degree polynomial.


## 2. Introduction and why curve fitting
Curve fitting is the process of choosing a mathematical model and its parameters so that the model describes observed data as well as possible. In physics and experimental science, measurements are rarely exact; they contain noise, limited precision, and possible systematic effects. Curve fitting helps us recover the underlying trend and estimate the parameters of a physical model.

Curve fitting is important because it allows us to turn raw experimental data into useful quantitative information. Instead of merely connecting points, we aim to infer a model that can explain the data, predict new values, and reveal the relationships between variables.

## 2.1 Least squares
Least squares is one of the most common methods for fitting a model to data. It chooses the parameters that minimize the sum of squared residuals,

$$
\chi^2 = \sum_i \left[y_i - f(x_i;\theta)\right]^2,
$$

where $y_i$ are observed values, $f(x_i;\theta)$ are model predictions, and $\theta$ denotes the unknown parameters. This method is simple, widely used, and especially convenient when the measurement errors are approximately Gaussian.

## 2.2 Maximum likelihood
Maximum likelihood estimation treats the data as random outcomes generated by a probability model. The parameters are chosen to maximize the likelihood of observing the data that were actually measured. In many cases, if the errors are assumed to be Gaussian, maximum likelihood leads to the same optimization problem as least squares, up to constant factors.

The likelihood function is written as

$$
\mathcal{L}(\theta) = p(D \mid \theta),
$$

and the maximum-likelihood estimate is obtained by finding

$$
\hat{\theta}_{\mathrm{MLE}} = \arg\max_{\theta} \mathcal{L}(\theta).
$$

## 2.3 Bayesian analysis
Bayesian analysis goes a step further by treating model parameters as random variables with prior distributions. Using the likelihood of the data and the prior information, we update to a posterior distribution,

$$
p(\theta \mid D) \propto p(D \mid \theta)\, p(\theta),
$$

where $D$ denotes the observed data. This framework provides a principled way to include prior knowledge, estimate uncertainty, and compare different models.

