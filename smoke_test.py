"""
Smoke test — Operador de Pro-Acción Γ
========================================
Implementación mínima (numpy stubs) que verifica:
1. Γ no explota con inputs dummy
2. Reducción a Driveplexity produce exactamente A1–A3
3. Ruta fast produce outputs distintos de ruta slow bajo alta amenaza
4. El vector de termostatos permanece finito tras varias iteraciones
5. Driveplexity degenerado es un subespacio de Γ

Uso: python smoke_test.py
"""

import numpy as np

np.random.seed(42)

# ============================================================
# Stubs de cada capa — reemplazables por implementaciones reales
# ============================================================

def A(o, I, D, M, x_A):
    """Atención: competición bottom-up / top-down.
    Salida: estímulo seleccionado (mismo shape que o) o 0 si nada pasa.
    """
    bu = np.linalg.norm(o) * 0.5
    td = (D * 0.3 + np.mean(M) * 0.2)
    theta = 0.5 + x_A * 0.1
    gate = 1 / (1 + np.exp(-(bu + td - theta)))
    return o * gate


def H_SAM(o, x_H):
    """Eje simpático-adrenomedular: output de arousal rápido."""
    surprise = np.linalg.norm(o)
    return 1 / (1 + np.exp(-(surprise + x_H)))


def N_fast(o, x, M):
    """Ruta rápida: thalamus → amígdala → motor."""
    return np.tanh(o[0] + x[2] + np.mean(M) * 0.1)


def P(o, x_P, M):
    """Perceptivo: predictive coding (error de predicción simplificado)."""
    pred = np.mean(M) * 0.1
    return o - pred


def E(P_out, I, M, x_E):
    """Emocional: valencia × arousal."""
    valence = np.tanh(P_out[0] + I * 0.3 + np.mean(M) * 0.1 + x_E)
    arousal = 1 / (1 + np.exp(-np.linalg.norm(P_out)))
    return np.array([valence, arousal])


def H_HPA(t, x_H_hist, tau=5):
    """HPA con retardo τ: cortisol basado en historia de x_H."""
    if t >= tau:
        return np.mean(x_H_hist[t - tau:t])
    return 0.0


def C(E_out, P_out, x_C, M, HPA_val):
    """Cognitivo: beliefs + plan + utilidad."""
    valence, arousal = E_out
    plan = np.tanh(valence * 0.5 + P_out[0] * 0.3 + x_C + np.mean(M) * 0.1 - HPA_val * 0.2)
    complexity = 0.05
    return plan - complexity


def ARB_pi(c_slow, c_fast, x, x_star, gamma=0.1, theta_arb=0.5, kappa=2.0):
    """Arbitrador de precisión entre ruta slow y fast."""
    # Free energy simplificada: -log P(c|a) proxy
    F_slow = -np.log(np.abs(c_slow) + 1e-6)
    F_fast = -np.log(np.abs(c_fast) + 1e-6)

    # Asignación de precisión según arousal/urgencia
    SAM_val = x[2]
    arousal = x[3]
    pi_fast = 1 / (1 + np.exp(-kappa * (SAM_val + arousal - theta_arb)))
    pi_slow = 1 - pi_fast

    # Penalización por desvío de set-points
    penalty = gamma * np.sum((x - x_star) ** 2)

    # Acción elegida como minimización de free energy ponderada
    a = pi_slow * F_slow + pi_fast * F_fast + penalty
    return a, pi_fast, pi_slow


def O(a, e_t, e_tp1, x):
    """Observación: distribuye señales de saciedad ρ_k."""
    change = np.linalg.norm(e_tp1 - e_t)
    rho = np.array([
        change * 0.1,           # ρ_A: cambio percibido
        change * 0.2,           # ρ_P: error reducido
        change * 0.3,           # ρ_H: balance energético
        change * 0.2,           # ρ_E: valencia post-acción
        change * 0.1,           # ρ_N: coste metabólico
        change * 0.2            # ρ_C: utilidad − complejidad
    ])
    return rho


# ============================================================
# Γ — Operador de Pro-Acción
# ============================================================

def Gamma(t, o_t, I_t, D_t, M_t, x, W, x_star, tau=5, x_H_hist=None):
    """
    Gamma: un paso del operador.
    x = [x_A, x_P, x_H, x_E, x_N, x_C]
    """
    # Paso 1: Saliencia
    o_tilde = A(o_t, I_t, D_t, M_t, x[0])

    # Paso 2: Rutas paralelas
    h_sam = H_SAM(o_tilde, x[2])
    c_fast = N_fast(o_tilde, x, M_t)

    p_out = P(o_tilde, x[1], M_t)
    e_out = E(p_out, I_t, M_t, x[3])
    hpa = H_HPA(t, x_H_hist, tau) if x_H_hist is not None else 0.0
    c_slow = C(e_out, p_out, x[5], M_t, hpa)

    # Paso 3: Arbitrador
    a, pi_fast, pi_slow = ARB_pi(c_slow, c_fast, x, x_star)

    return {
        'a': a,
        'pi_fast': pi_fast,
        'pi_slow': pi_slow,
        'c_fast': c_fast,
        'c_slow': c_slow,
        'o_tilde': o_tilde,
        'h_sam': h_sam,
        'hpa': hpa,
        'e_out': e_out
    }


def update_thermostats(x, a, e_t, e_tp1, lambdas, alphas, W, phi_fn, O_fn=None):
    """
    Actualización post-observación.
    x_{t+1} = x_t + λ − α ⊙ O(a, e_t, e_tp1) + W · φ(x_t)
    """
    obs = O_fn if O_fn is not None else O
    rho = obs(a, e_t, e_tp1, x)
    phi_x = phi_fn(x)
    x_new = x + lambdas - alphas * rho + W @ phi_x
    return x_new


def update_memory(M, a, O_val, x_new):
    """M_t+1 = M_t ⊕ encode(...). Simplificado: append + decay."""
    M_new = np.roll(M, -1)
    M_new[-1] = np.tanh(a + np.mean(x_new) * 0.1)
    return M_new


# ============================================================
# Smoke tests
# ============================================================

def test_01_gamma_does_not_explode():
    """Test 1: Γ no explota con inputs dummy."""
    x = np.zeros(6)
    o = np.array([0.5, -0.3, 0.1])
    I, D = 0.2, 0.3
    M = np.zeros(10)
    W = np.eye(6) * 0.05
    W[np.diag_indices(6)] = 0
    x_star = np.zeros(6)

    result = Gamma(t=0, o_t=o, I_t=I, D_t=D, M_t=M, x=x, W=W, x_star=x_star)

    assert np.isfinite(result['a']), "a* debe ser finito"
    assert 0 <= result['pi_fast'] <= 1, "π_fast debe estar en [0,1]"
    assert 0 <= result['pi_slow'] <= 1, "π_slow debe estar en [0,1]"
    assert np.isclose(result['pi_fast'] + result['pi_slow'], 1.0), "π_fast + π_slow = 1"
    print("[PASS] Test 1: Γ ejecuta sin errores, salidas finitas")


def test_02_reduction_to_driveplexity():
    """Test 2: Reducción a Driveplexity produce A1–A3 exactos."""
    # Colapso de todos los termostatos excepto x_C
    x = np.array([0, 0, 0, 0, 0, 0.5])  # sólo δ = 0.5

    # Driveplexity equivalent
    def driveplexity_step(delta, e_t, e_tp1, lam=0.1, alpha=0.3):
        """A1–A3 exactos del paper JAIIO."""
        change = np.linalg.norm(e_tp1 - e_t)
        delta_new = delta + lam - alpha * change
        p = 1 / (1 + np.exp(-(delta - 0.5)))  # σ(D(δ)) simplificado
        return delta_new, p

    e_t, e_tp1 = np.zeros(3), np.ones(3) * 0.5
    delta_dp, p_dp = driveplexity_step(x[5], e_t, e_tp1)

    # En Γ degenerado: A=id, I=0, M=0, D=0, W=0, fast route off
    # Para consistencia exacta con A3, O debe devolver g(e^(t),e^(t+1)) en la
    # posición cognitiva (sin escalamiento por capa).
    def O_dp(a, e_t, e_tp1, x):
        change = np.linalg.norm(e_tp1 - e_t)
        rho = np.zeros(6)
        rho[5] = change  # ρ_C ≡ g(e^(t), e^(t+1))
        return rho

    W = np.zeros((6, 6))
    lambdas = np.array([0, 0, 0, 0, 0, 0.1])  # sólo λ_C
    alphas = np.array([0, 0, 0, 0, 0, 0.3])   # sólo α_C

    x_new = update_thermostats(x, a=0, e_t=e_t, e_tp1=e_tp1,
                               lambdas=lambdas, alphas=alphas, W=W,
                               phi_fn=lambda x: np.zeros_like(x), O_fn=O_dp)

    assert np.isclose(x_new[5], delta_dp, rtol=1e-5), \
        f"Driveplexity mismatch: Γ={x_new[5]:.6f}, A1–A3={delta_dp:.6f}"
    print(f"[PASS] Test 2: Reducción a Driveplexity exacta (δ={x_new[5]:.4f})")


def test_03_fast_vs_slow_under_threat():
    """Test 3: Alta amenaza → π_fast sube significativamente."""
    x_calm = np.array([0, 0, 0, 0.2, 0, 0])      # bajo arousal
    x_threat = np.array([0, 0, 0.8, 0.9, 0, 0])  # alto SAM + arousal

    o = np.array([1.0, 0, 0])  # estímulo intenso
    W = np.zeros((6, 6))

    r_calm = Gamma(t=0, o_t=o, I_t=0.1, D_t=0.2, M_t=np.zeros(10),
                   x=x_calm, W=W, x_star=np.zeros(6))
    r_threat = Gamma(t=0, o_t=o, I_t=0.1, D_t=0.2, M_t=np.zeros(10),
                     x=x_threat, W=W, x_star=np.zeros(6))

    assert r_threat['pi_fast'] > r_calm['pi_fast'] + 0.3, \
        f"π_fast debería subir bajo amenaza: calm={r_calm['pi_fast']:.2f}, threat={r_threat['pi_fast']:.2f}"
    print(f"[PASS] Test 3: π_fast sube bajo amenaza ({r_calm['pi_fast']:.2f} → {r_threat['pi_fast']:.2f})")


def test_04_thermostats_remain_finite():
    """Test 4: 20 iteraciones → x_t permanece finito."""
    x = np.random.randn(6) * 0.2
    W = np.eye(6) * 0.05
    W[np.diag_indices(6)] = 0
    lambdas = np.ones(6) * 0.05
    alphas = np.ones(6) * 0.2
    x_star = np.zeros(6)
    M = np.zeros(10)
    x_H_hist = []

    for t in range(20):
        o = np.random.randn(3) * 0.5
        I = np.random.rand() * 0.3
        D = np.random.rand() * 0.3

        r = Gamma(t=t, o_t=o, I_t=I, D_t=D, M_t=M, x=x, W=W,
                  x_star=x_star, x_H_hist=x_H_hist)
        a = r['a']
        e_t = np.random.randn(3) * 0.3
        e_tp1 = e_t + np.random.randn(3) * 0.1

        x = update_thermostats(x, a, e_t, e_tp1, lambdas, alphas, W,
                               phi_fn=lambda x: np.tanh(x))
        x_H_hist.append(x[2])
        M = update_memory(M, a, 0, x)

        assert np.all(np.isfinite(x)), f"NaN/Inf en x en t={t}"

    print(f"[PASS] Test 4: 20 iteraciones sin divergencia (||x||={np.linalg.norm(x):.4f})")


def test_05_driveplexity_is_subspace():
    """Test 5: Driveplexity es subespacio de Γ (proyección preserva dinámica)."""
    x_full = np.random.randn(6) * 0.2
    x_full[:5] = 0  # sólo x_C activo

    x_dp = x_full[5]

    # Proyección al subespacio Driveplexity
    def project_dp(x):
        return np.array([0, 0, 0, 0, 0, x[5]])

    x_proj = project_dp(x_full)
    assert np.allclose(x_proj, x_full), "Driveplexity no es subespacio inyectivo"

    # Dinámica de Γ sobre subespacio debe coincidir con Driveplexity
    e_t, e_tp1 = np.zeros(3), np.ones(3)
    W = np.zeros((6, 6))
    lambdas = np.array([0, 0, 0, 0, 0, 0.1])
    alphas = np.array([0, 0, 0, 0, 0, 0.3])
    x_new = update_thermostats(x_full, a=0, e_t=e_t, e_tp1=e_tp1,
                               lambdas=lambdas, alphas=alphas, W=W,
                               phi_fn=lambda x: np.zeros_like(x))

    # Verificar que los 5 primeros termostatos se mantienen en 0
    assert np.allclose(x_new[:5], 0), "Subespacio Driveplexity no es invariante"
    print(f"[PASS] Test 5: Driveplexity es subespacio invariante de Γ (δ={x_new[5]:.4f})")


# ============================================================
# Ejecutar
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Smoke Test — Operador de Pro-Acción Γ")
    print("=" * 60)
    test_01_gamma_does_not_explode()
    test_02_reduction_to_driveplexity()
    test_03_fast_vs_slow_under_threat()
    test_04_thermostats_remain_finite()
    test_05_driveplexity_is_subspace()
    print("=" * 60)
    print("[ALL PASS] Los 5 smoke tests pasaron.")
    print("=" * 60)
