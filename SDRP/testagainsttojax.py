import torch
import jax
import jax.numpy as jnp

from pyRDDLGym_jax import FuzzyLogic as TorchLogic
from SDRP.logic   import FuzzyLogic as JaxLogic

from pyRDDLGym_jax import SigmoidComparison as TorchComp, SoftRounding as TorchRound, SoftControlFlow as TorchControl
from SDRP.logic  import SigmoidComparison as JaxComp,  SoftRounding as JaxRound,  SoftControlFlow as JaxControl


# =============================================================================
# CREATE LOGIC OBJECTS
# =============================================================================

torch_logic = TorchLogic(
    comparison=TorchComp(10000.0),
    rounding=TorchRound(10000.0),
    control=TorchControl(10000.0)
)

jax_logic = JaxLogic(
    comparison=JaxComp(10000.0),
    rounding=JaxRound(10000.0),
    control=JaxControl(10000.0)
)


# =============================================================================
# HELPERS
# =============================================================================

def to_torch(x):
    return torch.tensor(x, dtype=torch.float32)

def to_jax(x):
    return jnp.asarray(x, dtype=jnp.float32)

def compare_tensors(name, tx, jx, eps=1e-4):
    diff = torch.max(torch.abs(tx - torch.tensor(jx)))
    print(f"{name:20s} | max diff = {diff.item():.6f}")
    if diff > eps:
        print("❌ MISMATCH!")
    else:
        print("✓ OK")


# =============================================================================
# 1. LOGICAL OPERATORS
# =============================================================================

def test_compare_logical():
    print("\n=== COMPARE LOGICAL ===")
    init_t = {}
    init_j = {}

    x_t = to_torch([0.1, 0.9, -1.0, 2.0])
    y_t = to_torch([0.2, 0.5, -2.0, 2.0])

    x_j = to_jax([0.1, 0.9, -1.0, 2.0])
    y_j = to_jax([0.2, 0.5, -2.0, 2.0])

    ops = [
        ("AND",        torch_logic.logical_and,   jax_logic.logical_and),
        ("OR",         torch_logic.logical_or,    jax_logic.logical_or),
        ("NOT",        torch_logic.logical_not,   jax_logic.logical_not),
        ("GREATER",    torch_logic.greater,       jax_logic.greater),
        ("GE",         torch_logic.greater_equal, jax_logic.greater_equal),
        ("EQUAL",      torch_logic.equal,         jax_logic.equal),
    ]

    for name, opT, opJ in ops:
        fT = opT(0, init_t)
        fJ = opJ(0, init_j)

        if name == "NOT":
            t_out, _ = fT(x_t, init_t)
            j_out, _ = fJ(x_j, init_j)
        else:
            t_out, _ = fT(x_t, y_t, init_t)
            j_out, _ = fJ(x_j, y_j, init_j)

        compare_tensors(name, t_out, j_out)


# =============================================================================
# 2. ARGMAX / ARGMIN
# =============================================================================

def test_compare_argmax_argmin():
    print("\n=== COMPARE ARGMAX / ARGMIN ===")
    init_t = {}
    init_j = {}

    x = [2., 3., 5., 4.9, 4., 1., -1., -2.]

    x_t = to_torch(x)
    x_j = to_jax(x)

    t_argmax = torch_logic.argmax(0, init_t)
    j_argmax = jax_logic.argmax(0, init_j)

    t_argmin = torch_logic.argmin(1, init_t)
    j_argmin = jax_logic.argmin(1, init_j)

    t_max, _ = t_argmax(x_t, 0, init_t)
    j_max, _ = j_argmax(x_j, 0, init_j)

    t_min, _ = t_argmin(x_t, 0, init_t)
    j_min, _ = j_argmin(x_j, 0, init_j)

    compare_tensors("ARGMAX", t_max, j_max)
    compare_tensors("ARGMIN", t_min, j_min)


# =============================================================================
# 3. FORALL / EXISTS
# =============================================================================

def test_compare_quantifiers():
    print("\n=== COMPARE QUANTIFIERS ===")
    init_t = {}
    init_j = {}

    X = [[0.9, 0.4, 0.2],
         [1.0, 1.0, 0.7]]

    x_t = to_torch(X)
    x_j = to_jax(X)

    t_forall = torch_logic.forall(0, init_t)
    j_forall = jax_logic.forall(0, init_j)

    t_exists = torch_logic.exists(1, init_t)
    j_exists = jax_logic.exists(1, init_j)

    t_f, _ = t_forall(x_t, 1, init_t)
    j_f, _ = j_forall(x_j, 1, init_j)

    t_e, _ = t_exists(x_t, 1, init_t)
    j_e, _ = j_exists(x_j, 1, init_j)

    compare_tensors("FORALL", t_f, j_f)
    compare_tensors("EXISTS", t_e, j_e)


# =============================================================================
# 4. SOFT CONTROL FLOW
# =============================================================================

def test_compare_control_flow():
    print("\n=== COMPARE CONTROL FLOW ===")
    init_t = {}
    init_j = {}

    cond = [-10., -1., 0., 1., 10.]
    a     = [1.,1.,1.,1.,1.]
    b     = [0.,0.,0.,0.,0.]

    cond_t = to_torch(cond)
    cond_j = to_jax(cond)

    a_t = to_torch(a)
    a_j = to_jax(a)

    b_t = to_torch(b)
    b_j = to_jax(b)

    t_if = torch_logic.control_if(0, init_t)
    j_if = jax_logic.control_if(0, init_j)

    t_out, _ = t_if(cond_t, a_t, b_t, init_t)
    j_out, _ = j_if(cond_j, a_j, b_j, init_j)

    compare_tensors("IF", t_out, j_out)


# =============================================================================
# 5. ROUNDING
# =============================================================================

def test_compare_rounding():
    print("\n=== COMPARE ROUNDING ===")
    init_t = {}
    init_j = {}

    x = [2.1, 0.6, 1.99, -2.01, -3.2, -0.1, -1.01, 23.01]
    x_t = to_torch(x)
    x_j = to_jax(x)

    ops = [
        ("FLOOR", torch_logic.floor, jax_logic.floor),
        ("CEIL",  torch_logic.ceil,  jax_logic.ceil),
        ("ROUND", torch_logic.round, jax_logic.round),
    ]

    for name, opT, opJ in ops:
        fT = opT(0, init_t)
        fJ = opJ(0, init_j)

        t_out, _ = fT(x_t, init_t)
        j_out, _ = fJ(x_j, init_j)

        compare_tensors(name, t_out, j_out)


# =============================================================================
# 6. RANDOM DISTRIBUTIONS (SAME SEED)
# =============================================================================

def test_compare_random():
    print("\n=== COMPARE RANDOM ===")

    init_t = {}
    init_j = {}

    # same seed
    torch.manual_seed(42)
    key_j = jax.random.PRNGKey(42)

    # Bernoulli
    _ber_t = torch_logic.bernoulli(0, init_t)
    _ber_j = jax_logic.bernoulli(0, init_j)

    p_t = to_torch([0.3] * 10000)
    p_j = to_jax ([0.3] * 10000)

    t_s, _ = _ber_t(None, p_t, init_t)
    j_s, _ = _ber_j(key_j, p_j, init_j)

    compare_tensors("BERN", t_s.float().mean(), torch.tensor(j_s.mean()))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    test_compare_logical()
    test_compare_argmax_argmin()
    test_compare_quantifiers()
    test_compare_control_flow()
    test_compare_rounding()
    test_compare_random()
