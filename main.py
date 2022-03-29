import numpy as np
import random
import matplotlib.pyplot as plt


# import plotly.graph_objects as go
# import plotly.express as px
# import time
# from plotly.subplots import make_subplots


# plt.rcParams["figure.figsize"] = (20, 10)
# random.seed(1)

def gradient_descent(gradient_func, start_point, iterations, eps):
    current_point = start_point
    points = [current_point]
    for it in range(iterations):
        grad = gradient_func.next_gradient(current_point)
        next_point = current_point - grad
        distance = np.linalg.norm(current_point - next_point)
        if distance < eps:
            return points, it
        current_point = next_point
        points.append(current_point)
    return points, iterations


class MeanSquaredError:
    def function(self, regression, points, state):
        res = 0.0
        for p in points:
            res += (p[1] - regression.function(state, p[0])) ** 2
        return res / len(points)

    def gradient(self, regression, points, state):
        res = np.array([0.0] * len(state))
        for p in points:
            res -= 2 * (p[1] - regression.function(state, p[0])) * regression.gradient(p[0])
        return res / len(points)


class CommonGradientDescent:
    def __init__(self, regression, points, n, error_func, step, mu=None, beta1=None, beta2=None):
        self.regression = regression
        self.points = points
        self.n = n
        self.error_func = error_func
        self.step = step
        if mu:
            self.mu = mu
        if beta1:
            self.beta1 = beta1
        if beta2:
            self.beta2 = beta2


class StandartGradient(CommonGradientDescent):
    def __init__(self, regression, points, n, error_func, step):
        super().__init__(regression, points, n, error_func, step)

    def next_gradient(self, current_point):
        result = self.step * self.error_func.gradient(self.regression,
                                                      random.sample(self.points, self.n),
                                                      current_point)
        return result


class MomentumGradient(CommonGradientDescent):
    def __init__(self, regression, points, n, error_func, mu, step):
        super().__init__(regression, points, n, error_func, step, mu)
        self.prev_gradient = np.array([0.0] * (len(points[0][0]) + 1))

    def next_gradient(self, current_point):
        result = self.mu * self.prev_gradient + self.step * self.error_func.gradient(
            self.regression, random.sample(self.points, self.n),
            current_point
        )
        self.prev_gradient = result
        return result


class NesterovGradient(CommonGradientDescent):
    def __init__(self, regression, points, n, error_func, mu, step):
        super().__init__(regression, points, n, error_func, step, mu)
        self.prev_gradient = np.array([0.0] * (len(points[0][0]) + 1))

    def next_gradient(self, current_point):
        result = self.mu * self.prev_gradient + self.step * self.error_func.gradient(
            self.regression,
            random.sample(self.points, self.n),
            current_point + self.mu * self.prev_gradient
        )
        self.prev_gradient = result
        return result


class AdagradGradient(CommonGradientDescent):
    def __init__(self, regression, points, n, error_func, step):
        super().__init__(regression, points, n, error_func, step)
        self.s = np.array([0.0] * (len(points[0][0]) + 1))

    def next_gradient(self, current_point):
        current_gradient = self.error_func.gradient(self.regression, random.sample(self.points, self.n), current_point)
        self.s = self.s + np.square(current_gradient)
        result = np.multiply(self.step / np.sqrt(self.s), current_gradient)
        return result


class RMSPropGradient(CommonGradientDescent):
    def __init__(self, regression, points, n, error_func, mu, step):
        super().__init__(regression, points, n, error_func, step, mu)
        self.s = np.array([0.0] * (len(points[0][0]) + 1))

    def next_gradient(self, current_point):
        current_gradient = self.error_func.gradient(self.regression, random.sample(self.points, self.n), current_point)
        self.s = self.mu * self.s + (1 - self.mu) * np.square(current_gradient)
        result = np.multiply(self.step / np.sqrt(self.s), current_gradient)
        return result


class AdamGradient(CommonGradientDescent):
    def __init__(self, regression, points, n, error_func, beta1, beta2, step):
        super().__init__(regression, points, n, error_func, step, beta1=beta1, beta2=beta2)
        self.g = np.array([0.0] * (len(points[0][0]) + 1))
        self.v = np.array([0.0] * (len(points[0][0]) + 1))
        self.it = 1

    def next_gradient(self, current_point):
        current_gradient = self.error_func.gradient(self.regression, random.sample(self.points, self.n), current_point)
        self.g = self.beta1 * self.g + (1 - self.beta1) * current_gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(current_gradient)
        self.it = self.it + 1
        g_temp = self.g / (1 - self.beta1 ** (self.it - 1))
        v_temp = self.v / (1 - self.beta2 ** (self.it - 1))
        result = self.step * g_temp / (np.sqrt(v_temp) + 1e-8)
        return result


class LinearRegression:
    def __init__(self):
        self.function_calls = 0
        self.gradient_calls = 0

    def function(self, state, point):
        self.function_calls += 1

        res = state[0]
        for i in range(len(point)):
            res += state[i + 1] * point[i]
        return res

    def gradient(self, point):
        self.gradient_calls += 1

        return np.concatenate(([1.0], point))


def generate_points(f, number_of_dimensions, number_of_points):
    shifts = [random.uniform(-10, 10) for _ in range(number_of_dimensions)]
    multipliers = [random.uniform(0.1, 2) for _ in range(number_of_dimensions)]

    res = []
    for i in range(number_of_points):
        point = []
        for j in range(number_of_dimensions):
            x = (random.uniform(-5, 5) + shifts[j]) * multipliers[j]
            point.append(x)
        res.append((point, f(point)))
    return res


def f(xs):
    result = 5
    for i in range(len(xs)):
        result += (2 + i) * xs[i]
    return result


def check():
    regression = LinearRegression()
    error_func = MeanSquaredError()
    step = 1e-4
    eps = 1e-5
    train_points = generate_points(f, 50, 100)
    iterations = 1000

    ps = gradient_descent(
        gradient_func=StandartGradient(
            regression=regression,
            points=train_points,
            n=len(train_points),
            error_func=error_func,
            step=step
        ),
        start_point=np.array([0.0] * (len(train_points[0][0]) + 1)),
        iterations=iterations,
        eps=eps
    )[0]
    plt.plot(range(len(ps)), [error_func.function(regression, train_points, x) for x in ps])
    plt.savefig('check.png')


def task1():
    regression = LinearRegression()
    error_func = MeanSquaredError()
    step = 1e-4
    eps = 1e-1
    number_of_points = 200
    dimension_size = 40
    train_points = generate_points(f, dimension_size, number_of_points)

    batch_sizes = range(1, number_of_points + 1, 2)
    iterations = []
    for batch_size in batch_sizes:
        iterations.append(gradient_descent(
            gradient_func=StandartGradient(
                regression=regression,
                points=train_points,
                n=batch_size,
                error_func=error_func,
                step=step
            ),
            start_point=np.array([0.0] * (len(train_points[0][0]) + 1)),
            iterations=1000,
            eps=eps
        )[1])
    plt.title('SGD')
    plt.xlabel('Batch Size')
    plt.ylabel('Needed iterations to reach ok quality')
    plt.plot(batch_sizes, iterations)
    plt.savefig('task1.png')


def scale_points(points, dimension_size):
    scaled_points = []
    shifts = []
    multipliers = []
    for i in range(dimension_size):
        mn = points[0][0][i]
        mx = mn
        for j in range(len(points)):
            mn = min(mn, points[j][0][i])
            mx = max(mx, points[j][0][i])
        shifts.append((mn + mx) / 2)
        multipliers.append(2 / (mx - mn))

    for p in points:
        scaled_points.append(([(p[0][i] - shifts[i]) * multipliers[i] for i in range(dimension_size)], p[1]))

    return scaled_points


def task2():
    regression = LinearRegression()
    error_func = MeanSquaredError()
    step = 1e-4
    eps = 1e-1
    number_of_points = 100
    dimension_size = 40
    train_points = generate_points(f, dimension_size, number_of_points)
    scaled_points = scale_points(train_points, dimension_size)

    batch_sizes = range(1, number_of_points + 1, 2)
    iterations = []
    for batch_size in batch_sizes:
        iterations.append(gradient_descent(
            gradient_func=StandartGradient(
                regression=regression,
                points=scaled_points,
                n=batch_size,
                error_func=error_func,
                step=step
            ),
            start_point=np.array([0.0] * (len(train_points[0][0]) + 1)),
            iterations=1000,
            eps=eps
        )[1])
    plt.title('Scaled SGD')
    plt.xlabel('Batch Size')
    plt.ylabel('Needed iterations to reach ok quality')
    plt.plot(batch_sizes, iterations)
    plt.savefig('task2.png')


task2()
# data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
# scaler = MinMaxScaler()
# scaler.fit(data)
# data = scaler.transform(data)
