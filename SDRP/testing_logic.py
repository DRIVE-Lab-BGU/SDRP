import torch
from SDRP.logic import FuzzyLogic, SigmoidComparison, SoftRounding, SoftControlFlow

logic = FuzzyLogic(
    comparison=SigmoidComparison(10000.0),
    rounding=SoftRounding(10000.0),
    control=SoftControlFlow(10000.0)
)

# ============================================================
# 1. Logical Operators: AND, OR, NOT, GREATER, EQUAL
# ============================================================

def test_logical():
    print("\n=== TEST LOGICAL ===")

    init_params = {}
    _and = logic.logical_and(0, init_params)
    _or  = logic.logical_or(1, init_params)
    _not = logic.logical_not(2, init_params)
    _gt  = logic.greater(3, init_params)
    _eq  = logic.equal(4, init_params)

    x = torch.tensor([0.1, 0.9, -1.0, 2.0])
    y = torch.tensor([0.2, 0.5, -2.0, 2.0])

    print("AND:", _and(x, y, init_params)[0])
    print("OR :", _or(x, y, init_params)[0])
    print("NOT:", _not(x, init_params)[0])
    print("GREATER:", _gt(x, y, init_params)[0])
    print("EQUAL:", _eq(x, y, init_params)[0])


# ============================================================
# 2. Argmax / Argmin
# ============================================================

def test_indexing():
    print("\n=== TEST INDEXING ===")

    init_params = {}
    _argmax = logic.argmax(0, init_params)
    _argmin = logic.argmin(1, init_params)

    values = torch.tensor([2., 3., 5., 4.9, 4., 1., -1., -2.])
    amax, _ = _argmax(values, 0, init_params)
    amin, _ = _argmin(values, 0, init_params)

    print("argmax fuzzy:", amax)
    print("argmin fuzzy:", amin)


# ============================================================
# 3. T-Norm Quantifiers (FORALL, EXISTS)
# ============================================================

def test_quantifiers():
    print("\n=== TEST QUANTIFIERS ===")

    init_params = {}
    _forall = logic.forall(0, init_params)
    _exists = logic.exists(1, init_params)

    x = torch.tensor([[0.9, 0.4, 0.2],
                      [1.0, 1.0, 0.7]])

    f, _ = _forall(x, 1, init_params)
    e, _ = _exists(x, 1, init_params)

    print("forall:", f)
    print("exists:", e)


# ============================================================
# 4. Soft control flow: if, switch, select
# ============================================================

def test_control():
    print("\n=== TEST CONTROL FLOW ===")

    init_params = {}
    _if = logic.control_if(0, init_params)
    _switch = logic.control_switch(1, init_params)

    cond = torch.tensor([-10., -1., 0., 1., 10.])
    a = torch.ones_like(cond)
    b = torch.zeros_like(cond)

    result_if, _ = _if(cond, a, b, init_params)
    print("soft if:", result_if)

    c1 = -10 * torch.ones(5)
    c2 = +5 * torch.ones(5)
    c3 = +10 * torch.ones(5)
    cases = torch.stack([c1, c2, c3])

    pred = torch.tensor([-1., 0., 0.5, 1.0, 2.0])
    result_sw, _ = _switch(pred, cases, init_params)

    print("soft switch:", result_sw)


# ============================================================
# 5. Random: Bernoulli, Discrete, Geometric
# ============================================================

def test_random():
    print("\n=== TEST RANDOM ===")

    init_params = {}
    key = torch.manual_seed(42)

    _ber = logic.bernoulli(0, init_params)
    _dis = logic.discrete(1, init_params)
    _geo = logic.geometric(2, init_params)

    # Bernoulli
    b, _ = _ber(key, torch.tensor([0.3] * 50000), init_params)
    print("Bernoulli mean:", b.float().mean())

    # Discrete (3 classes)
    prob = torch.tensor([0.1, 0.4, 0.5])
    prob = prob.unsqueeze(0).repeat(50000, 1)
    d, _ = _dis(key, prob, init_params)
    d = torch.round(d)
    print("Discrete frequencies:",
          [(d == i).float().mean().item() for i in range(3)])

    # Geometric
    g, _ = _geo(key, torch.tensor([0.3] * 50000), init_params)
    print("Geometric mean:", g.float().mean())


# ============================================================
# 6. Rounding: floor, ceil, round, mod
# ============================================================

def test_rounding():
    print("\n=== TEST ROUNDING ===")

    init_params = {}
    _floor = logic.floor(0, init_params)
    _ceil = logic.ceil(1, init_params)
    _round = logic.round(2, init_params)
    _mod = logic.mod(3, init_params)

    x = torch.tensor([2.1, 0.6, 1.99, -2.01, -3.2, -0.1, -1.01, 23.01])
    print("floor:", _floor(x, init_params)[0])
    print("ceil :", _ceil(x, init_params)[0])
    print("round:", _round(x, init_params)[0])
    print("mod  :", _mod(x, 2.0, init_params)[0])


# ============================================================
# 7. Gradient test (critical!)
# ============================================================

def test_gradients():
    print("\n=== TEST GRADIENTS ===")

    init_params = {}
    _and = logic.logical_and(0, init_params)

    x = torch.tensor([0.1, 0.6, 0.9], requires_grad=True)
    y = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)

    out, _ = _and(x, y, init_params)
    loss = out.sum()
    loss.backward()

    print("grad x:", x.grad)
    print("grad y:", y.grad)


# ============================================================
# 8. FULL PIPELINE TEST
# ============================================================

def test_pipeline():
    print("\n=== END-TO-END PIPELINE ===")

    init_params = {}

    _gt = logic.greater(0, init_params)
    _and = logic.logical_and(1, init_params)
    _if  = logic.control_if(2, init_params)

    x = torch.tensor([0.9, 0.1, 2.0])
    y = torch.tensor([0.5, 0.5, 0.5])

    c1, _ = _gt(x, y, init_params)
    c2, _ = _gt(x, torch.tensor(0.0), init_params)
    both, _ = _and(c1, c2, init_params)

    out, _ = _if(both, x, y, init_params)
    print("final pipeline output:", out)


def test_logical_basic():
    print("\n=== LOGICAL BASIC ===")
    init = {}
    _and = logic.logical_and(0, init)
    _or  = logic.logical_or(1, init)
    _not = logic.logical_not(2, init)
    _gt  = logic.greater(3, init)
    _eq  = logic.equal(4, init)

    x = torch.tensor([0.1, 0.9, -1.0, 2.0])
    y = torch.tensor([0.2, 0.5, -2.0, 2.0])

    print("AND :", _and(x, y, init)[0])
    print("OR  :", _or(x, y, init)[0])
    print("NOT :", _not(x, init)[0])
    print("GT  :", _gt(x, y, init)[0])
    print("EQ  :", _eq(x, y, init)[0])


def test_and_monotonicity():
    print("\n=== AND MONOTONICITY ===")
    init = {}
    _and = logic.logical_and(0, init)

    x1 = torch.tensor([0.1, 0.5, 0.9])
    x2 = x1 + 0.1
    y  = torch.tensor([0.2, 0.2, 0.2])

    out1, _ = _and(x1, y, init)
    out2, _ = _and(x2, y, init)

    print("monotonic:", torch.all(out1 <= out2))


def test_demorgan():
    print("\n=== DE MORGAN ===")
    init = {}
    _and = logic.logical_and(0, init)
    _or  = logic.logical_or(1, init)
    _not = logic.logical_not(2, init)

    x = torch.rand(5)
    y = torch.rand(5)

    L, _ = _not(_and(x, y, init)[0], init)
    nx, _ = _not(x, init)
    ny, _ = _not(y, init)
    R, _ = _or(nx, ny, init)

    print("L:", L)
    print("R:", R)
    print("close?:", torch.allclose(L, R, atol=1e-2))


def test_zero_weight_comparison():
    print("\n=== ZERO WEIGHT COMPARISON ===")

    logic0 = FuzzyLogic(
        comparison=SigmoidComparison(0.0),
        rounding=SoftRounding(0.0),
        control=SoftControlFlow(0.0),
    )

    init = {}
    _ge = logic0.greater_equal(0, init)

    x = torch.tensor([-100., -1., 0., 1., 100.])
    out, _ = _ge(x, 0, init)
    print("should all be ~0.5:", out)


# =============================================================================
# 2. INDEXING: ARGMAX / ARGMIN
# =============================================================================

def test_argmax_basic():
    print("\n=== ARGMAX BASIC ===")
    init = {}
    _argmax = logic.argmax(0, init)

    x = torch.tensor([1., 4., 9., 3.])
    out, _ = _argmax(x, 0, init)

    print("soft argmax:", out)
    print("max index:", torch.argmax(out).item())


def test_argmax_axis1():
    print("\n=== ARGMAX AXIS 1 ===")
    init = {}
    _argmax = logic.argmax(0, init)

    x = torch.tensor([
        [1., 7., 3.],
        [5., 1., 2.]
    ])
    out, _ = _argmax(x, 1, init)
    print(out)


def test_argmax_ties():
    print("\n=== ARGMAX TIES ===")
    init = {}
    _argmax = logic.argmax(0, init)

    x = torch.tensor([5., 5., 5., 5.])
    out, _ = _argmax(x, 0, init)
    print("expect uniform:", out)


# =============================================================================
# 3. QUANTIFIERS: FORALL / EXISTS
# =============================================================================

def test_forall_exists():
    print("\n=== QUANTIFIERS ===")

    init = {}
    _forall = logic.forall(0, init)
    _exists = logic.exists(1, init)

    x = torch.tensor([
        [0.9, 0.4, 0.2],
        [1.0, 1.0, 0.7]
    ])

    f, _ = _forall(x, 1, init)
    e, _ = _exists(x, 1, init)

    print("forall:", f)
    print("exists:", e)


def test_forall_one_bad():
    print("\n=== FORALL ONE BAD ===")
    init = {}
    _forall = logic.forall(0, init)

    x = torch.tensor([1.,1.,1.,1.,0.1])
    out, _ = _forall(x, 0, init)
    print(out)


def test_exists_one_good():
    print("\n=== EXISTS ONE GOOD ===")
    init = {}
    _exists = logic.exists(0, init)

    x = torch.tensor([0., 0., 0., 0.9])
    out, _ = _exists(x, 0, init)
    print(out)


# =============================================================================
# 4. SOFT CONTROL FLOW
# =============================================================================

def test_control_flow():
    print("\n=== CONTROL FLOW ===")

    init = {}
    _if = logic.control_if(0, init)
    _switch = logic.control_switch(1, init)

    cond = torch.tensor([-10., -2., 0., 2., 10.])
    a = torch.ones_like(cond)
    b = torch.zeros_like(cond)

    out_if, _ = _if(cond, a, b, init)
    print("soft if:", out_if)

    c1 = -10 * torch.ones(5)
    c2 = +5  * torch.ones(5)
    c3 = +10 * torch.ones(5)
    cases = torch.stack([c1, c2, c3])

    pred = torch.tensor([-1., 0., 0.5, 1.0, 2.0])
    out_sw, _ = _switch(pred, cases, init)
    print("soft switch:", out_sw)


# =============================================================================
# 5. RANDOM: BERNOULLI / DISCRETE / GEOMETRIC
# =============================================================================

def test_random_bernoulli_lln():
    print("\n=== BERNOULLI LLN ===")
    init = {}
    key = torch.manual_seed(123)

    _ber = logic.bernoulli(0, init)

    p = 0.7
    samples, _ = _ber(key, torch.tensor([p]*200000), init)
    print("empirical:", samples.float().mean())


def test_random_discrete_uniform():
    print("\n=== DISCRETE UNIFORM ===")
    init = {}
    key = torch.manual_seed(999)

    _dis = logic.discrete(0, init)

    prob = torch.tensor([1/3,1/3,1/3]).repeat(50000,1)
    samp, _ = _dis(key, prob, init)
    samp = samp.round()

    print("freq:", [(samp==i).float().mean().item() for i in range(3)])


def test_random_geometric_mean():
    print("\n=== GEOMETRIC MEAN ===")
    init = {}
    key = torch.manual_seed(1)
    _geo = logic.geometric(0, init)

    p = 0.4
    samples, _ = _geo(key, torch.tensor([p]*50000), init)

    print("empirical:", samples.float().mean().item())
    print("theoretical:", 1/p)


# =============================================================================
# 6. ROUNDING / CEIL / FLOOR / MOD
# =============================================================================

def test_rounding():
    print("\n=== ROUNDING ===")
    init = {}
    _floor = logic.floor(0, init)
    _ceil  = logic.ceil(1, init)
    _round = logic.round(2, init)
    _mod   = logic.mod(3, init)

    x = torch.tensor([2.1, 0.6, 1.99, -2.01, -3.2, -0.1, -1.01, 23.01])
    print("floor:", _floor(x, init)[0])
    print("ceil :", _ceil(x, init)[0])
    print("round:", _round(x, init)[0])
    print("mod  :", _mod(x, 2.0, init)[0])


def test_mod_negative():
    print("\n=== MOD NEGATIVE ===")
    init = {}
    _mod = logic.mod(0, init)

    x = torch.tensor([-10., -5., 0., 5., 9.])
    out, _ = _mod(x, 4.0, init)
    print(out)


# =============================================================================
# 7. GRADIENT CHECKS
# =============================================================================

def test_grad_and():
    print("\n=== GRAD AND ===")
    init = {}
    _and = logic.logical_and(0, init)

    x = torch.tensor([0.1, 0.6, 0.9], requires_grad=True)
    y = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)

    out, _ = _and(x, y, init)
    loss = out.sum()
    loss.backward()

    print("grad x:", x.grad)
    print("grad y:", y.grad)


def test_grad_argmax():
    print("\n=== GRAD ARGMAX ===")
    init = {}
    _argmax = logic.argmax(0, init)

    x = torch.tensor([0.2, 0.9, 0.7], requires_grad=True)
    out, _ = _argmax(x, 0, init)
    out.sum().backward()

    print("grad:", x.grad)
    print("finite:", torch.all(torch.isfinite(x.grad)))


def test_grad_forall():
    print("\n=== GRAD FORALL ===")
    init = {}
    _forall = logic.forall(0, init)

    x = torch.tensor([0.9,0.7,0.4], requires_grad=True)
    out, _ = _forall(x, 0, init)
    out.backward()

    print("grad:", x.grad)


# =============================================================================
# 8. NUMERIC STABILITY
# =============================================================================

def test_sigmoid_overflow():
    print("\n=== SIGMOID OVERFLOW ===")

    comp = SigmoidComparison(1e6)
    init = {}
    _ge = comp.greater_equal(0, init)

    x = torch.tensor([1e6])
    y = torch.tensor([-1e6])

    out, _ = _ge(x, y, init)
    print(out)


# =============================================================================
# 9. STRESS TESTS
# =============================================================================

def test_large_tensor():
    print("\n=== LARGE TENSOR ===")
    init = {}
    _and = logic.logical_and(0, init)

    x = torch.rand(100000)
    y = torch.rand(100000)

    out, _ = _and(x, y, init)
    print("shape:", out.shape)


# =============================================================================
# 10. END-TO-END PIPELINE
# =============================================================================

def test_pipeline():
    print("\n=== END TO END PIPELINE ===")

    init = {}
    _gt  = logic.greater(0, init)
    _and = logic.logical_and(1, init)
    _if  = logic.control_if(2, init)

    x = torch.tensor([0.9, 0.1, 2.0])
    y = torch.tensor([0.5, 0.5, 0.5])

    c1, _ = _gt(x, y, init)
    c2, _ = _gt(x, 0.0, init)
    both, _ = _and(c1, c2, init)

    out, _ = _if(both, x, y, init)
    print(out)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_logical()
    test_indexing()
    test_quantifiers()
    test_control()
    test_random()
    test_rounding()
    test_gradients()
    test_pipeline()
    test_logical_basic()
    test_and_monotonicity()
    test_demorgan()
    test_zero_weight_comparison()

    test_argmax_basic()
    test_argmax_axis1()
    test_argmax_ties()

    test_forall_exists()
    test_forall_one_bad()
    test_exists_one_good()

    test_control_flow()

    test_random_bernoulli_lln()
    test_random_discrete_uniform()
    test_random_geometric_mean()

    test_rounding()
    test_mod_negative()

    test_grad_and()
    test_grad_argmax()
    test_grad_forall()

    test_sigmoid_overflow()
    test_large_tensor()

    test_pipeline()
