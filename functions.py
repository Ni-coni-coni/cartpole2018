import numpy as np

def sphere(X):
    lam = X.shape[1]
    eval = np.sum(X ** 2, axis = 0)
    feasible = np.ones(lam)
    return (eval, feasible)

def ellipsoid(X):
    d = X.shape[0]
    lam = X.shape[1]
    mem = np.arange(d).reshape(d, 1)
    elli = 1000 ** (mem / (d - 1))
    eval = np.sum((X * elli) ** 2, axis = 0)
    feasible = np.ones(lam)
    return (eval, feasible)

def rosenbrock(X):
    d = X.shape[0]
    lam = X.shape[1]
    eval = np.sum(100 * (X[1:, :] - X[0:d-1, :] ** 2) ** 2 + (X[0:d-1, :] - 1) ** 2, axis = 0)
    feasible = np.ones(lam)
    return (eval, feasible)

def rastrigin(X):
    d = X.shape[0]
    lam = X.shape[1]
    eval = 10 * d + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=0)
    feasible = np.ones(lam)
    return (eval, feasible)

if __name__ == "__main__":
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    eval, feasible = rastrigin(X)
    print(eval)
    print(feasible)