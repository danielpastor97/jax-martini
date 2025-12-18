import pdb
import numpy as np
from scipy.stats import wasserstein_distance as scipy_w1
import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad

from ott.geometry import pointcloud, costs
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


def wasserstein_1d(u, v, u_weights=None, v_weights=None):
    u = jnp.asarray(u, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)

    if u_weights is None:
        u_weights = jnp.full(u.shape, 1.0 / u.size, dtype=jnp.float64)
    else:
        u_weights = jnp.asarray(u_weights, dtype=jnp.float64)

    if v_weights is None:
        v_weights = jnp.full(v.shape, 1.0 / v.size, dtype=jnp.float64)
    else:
        v_weights = jnp.asarray(v_weights, dtype=jnp.float64)

    # Sort u and v with their weights
    u_sort_idx = jnp.argsort(u)
    v_sort_idx = jnp.argsort(v)

    u = u[u_sort_idx]
    v = v[v_sort_idx]
    u_weights = u_weights[u_sort_idx]
    v_weights = v_weights[v_sort_idx]

    # Merge all support points
    all_vals = jnp.concatenate([u, v])
    all_weights = jnp.concatenate([u_weights, -v_weights])
    sort_idx = jnp.argsort(all_vals)
    all_vals = all_vals[sort_idx]
    all_weights = all_weights[sort_idx]

    # Compute CDF difference over each interval
    diffs = jnp.cumsum(all_weights)
    dx = all_vals[1:] - all_vals[:-1]
    avg_heights = jnp.abs(diffs[:-1])

    return jnp.sum(dx * avg_heights)


def sinkhorn_1d(u, v, u_weights=None, v_weights=None, epsilon=1e-3):
    u = jnp.asarray(u).reshape(-1, 1)
    v = jnp.asarray(v).reshape(-1, 1)

    n, m = u.shape[0], v.shape[0]

    if u_weights is None:
        u_weights = jnp.full(n, 1.0 / n)
    else:
        u_weights = jnp.asarray(u_weights)

    if v_weights is None:
        v_weights = jnp.full(m, 1.0 / m)
    else:
        v_weights = jnp.asarray(v_weights)

    cost_fn = costs.PNormP(p=1.0)
    geom = pointcloud.PointCloud(u, v, epsilon=epsilon, cost_fn=cost_fn)
    prob = linear_problem.LinearProblem(geom, a=u_weights, b=v_weights)

    out = sinkhorn.Sinkhorn()(prob)
    return out.reg_ot_cost


class TestExactWasserstein(unittest.TestCase):

    def test_matches_scipy_equal_weights(self):

        n_tests = 100

        for _ in range(n_tests):
            n = np.random.randint(5, 100)
            m = np.random.randint(5, 100)
            u = np.random.normal(loc=0.0, scale=2.0, size=n)
            v = np.random.normal(loc=1.0, scale=2.0, size=m)

            d_jax = float(wasserstein_1d(u, v))
            d_scipy = scipy_w1(u, v)
            self.assertAlmostEqual(d_jax, d_scipy, places=6)

    def test_matches_scipy_weighted_random(self):

        n_tests = 100

        for _ in range(n_tests):
            n = np.random.randint(5, 100)
            m = np.random.randint(5, 100)
            u = np.random.uniform(-10, 10, size=n)
            v = np.random.uniform(-10, 10, size=m)

            u_weights = np.random.rand(n)
            u_weights /= u_weights.sum()

            v_weights = np.random.rand(m)
            v_weights /= v_weights.sum()

            d_jax = float(wasserstein_1d(u, v, u_weights, v_weights))
            d_scipy = scipy_w1(u, v, u_weights, v_weights)
            self.assertAlmostEqual(d_jax, d_scipy, places=6)

    def test_matches_scipy_weighted_manual_case(self):
        u = np.array([0.0, 1.0, 3.0])
        v = np.array([2.0, 4.0])
        u_weights = np.array([0.2, 0.5, 0.3])
        v_weights = np.array([0.6, 0.4])

        d_jax = float(wasserstein_1d(u, v, u_weights, v_weights))
        d_scipy = scipy_w1(u, v, u_weights, v_weights)

        self.assertAlmostEqual(d_jax, d_scipy, places=6)

    def test_edge_case_identical_distributions(self):
        x = np.linspace(-1, 1, 50)
        self.assertAlmostEqual(float(wasserstein_1d(x, x)), 0.0, places=8)

    def test_edge_case_degenerate(self):
        # One point vs another
        u = np.array([0.0])
        v = np.array([2.0])
        self.assertAlmostEqual(float(wasserstein_1d(u, v)), 2.0, places=6)

        # Same point
        self.assertAlmostEqual(float(wasserstein_1d([1.0], [1.0])), 0.0, places=6)

    def test_jit_compatible(self):
        u = jnp.array([0.0, 1.0, 3.0])
        v = jnp.array([2.0, 4.0])
        u_weights = jnp.array([0.2, 0.5, 0.3])
        v_weights = jnp.array([0.6, 0.4])

        compiled_fn = jit(wasserstein_1d)
        result = float(compiled_fn(u, v, u_weights, v_weights))
        expected = scipy_w1(
            np.array(u), np.array(v), np.array(u_weights), np.array(v_weights)
        )

        self.assertAlmostEqual(result, expected, places=6)

    def test_grad_compatible_on_points(self):
        def loss_fn(u):
            v = jnp.array([1.0, 2.0])
            return wasserstein_1d(u, v)

        u_init = jnp.array([0.0, 3.0])
        grad_fn = grad(loss_fn)
        g = grad_fn(u_init)

        self.assertEqual(g.shape, u_init.shape)
        self.assertTrue(jnp.all(jnp.isfinite(g)))

    def test_grad_wrt_weights_matches_finite_difference(self):
        u = jnp.array([0.0, 1.0, 3.0])
        v = jnp.array([2.0, 4.0])
        v_weights = jnp.array([0.6, 0.4])

        def loss_fn(weights):
            weights = weights / jnp.sum(weights)  # ensure normalization
            return wasserstein_1d(u, v, weights, v_weights)

        u_weights_init = jnp.array([0.2, 0.5, 0.3])
        g = grad(loss_fn)(u_weights_init)

        self.assertEqual(g.shape, u_weights_init.shape)
        self.assertTrue(jnp.all(jnp.isfinite(g)))

        # Finite difference check
        eps = 1e-5
        fd_grad = []
        for i in range(len(u_weights_init)):
            delta = jnp.zeros_like(u_weights_init).at[i].set(eps)
            up = u_weights_init + delta
            down = u_weights_init - delta
            up = up / jnp.sum(up)
            down = down / jnp.sum(down)
            fd = (loss_fn(up) - loss_fn(down)) / (2 * eps)
            fd_grad.append(fd)
        fd_grad = jnp.array(fd_grad)

        # Check close to autodiff
        diff = jnp.abs(g - fd_grad)
        self.assertTrue(jnp.all(diff < 1e-3))


if __name__ == "__main__":
    unittest.main()
