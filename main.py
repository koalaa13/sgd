import numpy as np
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# plt.rcParams["figure.figsize"] = (20, 10)
random.seed(1)


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


def gradient_descent(gradient_func: CommonGradientDescent, start_point, iterations, eps):
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
    eps = 1e-1
    number_of_points = 100
    dimension_size = 40
    train_points = generate_points(f, dimension_size, number_of_points)
    scaled_points = scale_points(train_points, dimension_size)

    batch_sizes = range(1, number_of_points + 1, 2)
    plt.xlabel('Batch Size')
    plt.ylabel('Needed iterations to reach ok quality')
    steps = [5e-2, 1e-4]
    legend = []

    for step in steps:
        scaled_iterations = []
        iterations = []
        for batch_size in batch_sizes:
            trajectory, its = gradient_descent(
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
            )
            scaled_iterations.append(its)
            trajectory, its = gradient_descent(
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
            )
            iterations.append(its)
            print("batch_size" + str(batch_size) + "finished with step = " + str(step))
        legend.append(plt.plot(batch_sizes, iterations, label='SGD with step = ' + str(step))[0])
        legend.append(plt.plot(batch_sizes, scaled_iterations, label='Scaled SGD with step = ' + str(step))[0])
    plt.suptitle('Task2')
    plt.legend(handles=legend)
    plt.savefig('task2.png')


def draw_iterations_to_error_graphics():
    regression = LinearRegression()
    error_func = MeanSquaredError()
    eps = 1e-1
    number_of_points = 100
    dimension_size = 40
    step = 5e-2
    train_points = generate_points(f, dimension_size, number_of_points)
    batch_size = 50

    trajectory, its = gradient_descent(
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
    )
    plt.plot(range(len(trajectory)), [error_func.function(regression, train_points, x) for x in trajectory])
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.savefig('check.png')


def task3():
    number_of_points = 100
    dimension_size = 40
    train_points = generate_points(f, dimension_size, number_of_points)
    train_points = scale_points(train_points, dimension_size)
    methods = {
        "GD": StandartGradient,
        "SGD": StandartGradient,
        "Momentum": MomentumGradient,
        "Nesterov": NesterovGradient,
        "AdaGrad": AdagradGradient,
        "RMSProp": RMSPropGradient,
        "Adam": AdamGradient
    }
    plots = []
    for label, gradient in methods.items():
        regression = LinearRegression()
        error_func = MeanSquaredError()

        n = len(train_points) // 10
        eps = 1e-2
        step = 1e-2
        mu = 0.9
        beta1 = 0.9
        beta2 = 0.999

        if label == "GD":
            n = len(train_points)
        if label == "SGD":
            pass
        if label == "Momentum":
            pass
        if label == "Nesterov":
            step = 1e-3
        if label == "AdaGrad":
            step = 50
        if label == "RMSProp":
            step = 10
        if label == "Adam":
            step = 10

        start_time = time.time()
        trajectory = gradient_descent(
            gradient_func=gradient(
                regression=regression,
                points=train_points,
                n=n,
                error_func=error_func,
                step=step,
                mu=mu,
                beta1=beta1,
                beta2=beta2,
            ),
            start_point=np.array([0.0] * (len(train_points[0][0]) + 1)),
            iterations=500,
            eps=eps
        )[0][1:]

        print(label)
        print(f'function calls: {regression.function_calls}')
        print(f'gradient calls: {regression.gradient_calls}')
        print(f'seconds: {time.time() - start_time}')
        plots.append(
            plt.plot(range(len(trajectory)), [error_func.function(regression, train_points, x) for x in trajectory],
                     label=label)[0])

    plt.suptitle('Task3-4')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.legend(handles=plots)
    plt.savefig('task3-4.png')

def task5():
    number_of_points = 100
    dimension_size = 1

    fig = make_subplots(rows=7, cols=1,
                        subplot_titles=("GD", "SGD", "Momentum", "Nesterov", "AdaGrad", "RMSProp", "Adam"))
    methods = {
        "GD": StandartGradient,
        "SGD": StandartGradient,
        "Momentum": MomentumGradient,
        "Nesterov": NesterovGradient,
        "AdaGrad": AdagradGradient,
        "RMSProp": RMSPropGradient,
        "Adam": AdamGradient
    }

    error_func = MeanSquaredError()
    regression = LinearRegression()

    def g(xs):
        return 10 - 4 * xs[0]

    train_points = generate_points(g, dimension_size, number_of_points)

    xs = list(range(-2, 12))
    ys = list(range(-10, 2))
    zs = []
    for y in ys:
        layer = []
        for x in xs:
            layer.append(error_func.function(regression, train_points, [x, y]))
        zs.append(layer)

    row = 0
    for label, gradient in methods.items():
        row += 1

        n = len(train_points) // 20
        eps = 1e-5
        step = 1e-3
        mu = 0.9
        beta1 = 0.9
        beta2 = 0.999

        if label == "GD":
            n = len(train_points)
        if label == "SGD":
            pass
        if label == "Momentum":
            pass
        if label == "Nesterov":
            step = 3e-4
            mu = 0.85
        if label == "AdaGrad":
            step = 2
        if label == "RMSProp":
            step = 2e-1
        if label == "Adam":
            step = 2e-1

        trajectory = gradient_descent(
            gradient_func=gradient(
                regression=regression,
                points=train_points,
                n=n,
                error_func=error_func,
                step=step,
                mu=mu,
                beta1=beta1,
                beta2=beta2,
            ),
            start_point=np.array([0.0] * (len(train_points[0][0]) + 1)),
            iterations=1000,
            eps=eps
        )[0]

        fig.add_trace(go.Contour(x=xs, y=ys, z=zs), row=row, col=1)
        fig.add_trace(go.Scatter(x=[p[0] for p in trajectory], y=[p[1] for p in trajectory], mode='lines',
                                 line=dict(color='lightgreen')), row=row, col=1)

    fig.update_layout(height=3000, width=1400)
    fig.write_image('task5.png')

# check()
task5()
