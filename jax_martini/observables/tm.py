import jax
from jax import jit

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optimistix as optx


def calculate_apl(t, apl0, c_p_g, dAPL, k, Tm):
    return apl0 + c_p_g * t + dAPL / (1 + jnp.exp(-k * (t - Tm)))


def apl_residual(params, args):

    simulated_temperatures, simulated_apls = args

    # coefficients ordering: [apl0, c_p_g, dAPL, k, Tm]
    apl0, c_p_g, dAPL, k, Tm = params[0], params[1], params[2], params[3], params[4]

    apl_calc = calculate_apl(simulated_temperatures, apl0, c_p_g, dAPL, k, Tm)

    residual = simulated_apls - apl_calc
    return residual


@jit
def get_guess(sim_apls, temps):
    apl0 = jnp.min(sim_apls) - 0.0001 * 276
    c_p_g = 1e-4
    dAPL = jnp.max(sim_apls) - jnp.min(sim_apls)
    k = 1.0
    Tm = jnp.median(temps)
    return jnp.array([apl0, c_p_g, dAPL, k, Tm])


@jit
def get_apl_params(sim_apls, sim_temps, max_steps=5000):
    init_guess = get_guess(sim_apls, sim_temps)

    # optimistix
    solver = optx.LevenbergMarquardt(rtol=1e-3, atol=1e-3)
    soln = optx.least_squares(
        apl_residual,
        solver,
        init_guess,
        args=(sim_temps, sim_apls),
        max_steps=max_steps,
    )
    opt_vals = soln.value

    return opt_vals
