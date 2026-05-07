"""
simulator.py — Simulador de 2 agentes con Operador Γ para micro-experimento.
=============================================================================
Genera un JSON con la historia de la simulación para visualización.
No requiere matplotlib ni LLM real (usa mock).

Uso:
    python simulator.py --rounds 30 --output simulation.json
"""

import argparse
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List


# ============================================================
# Stubs simplificados del operador Γ (del smoke_test.py)
# ============================================================

def sigmoid(x, k=1.0):
    return 1.0 / (1.0 + np.exp(-k * x))


def attention_gate(o, I, D, M, x_A):
    bu = np.linalg.norm(o) * 0.5
    td = D * 0.3 + np.mean(M) * 0.2
    theta = 0.5 + x_A * 0.1
    gate = sigmoid(bu + td - theta)
    return o * gate


def hormone_SAM(o, x_H):
    surprise = np.linalg.norm(o)
    return sigmoid(surprise + x_H)


def neuro_fast(o, x, M):
    return np.tanh(o[0] + x[2] + np.mean(M) * 0.1)


def perception(o, x_P, M):
    pred = np.mean(M) * 0.1
    return o - pred


def emotion(P_out, I, M, x_E):
    valence = np.tanh(P_out[0] + I * 0.3 + np.mean(M) * 0.1 + x_E)
    arousal = sigmoid(np.linalg.norm(P_out))
    return np.array([valence, arousal])


def cognition(E_out, P_out, x_C, M, HPA_val):
    valence, arousal = E_out
    plan = np.tanh(valence * 0.5 + P_out[0] * 0.3 + x_C + np.mean(M) * 0.1 - HPA_val * 0.2)
    return plan - 0.05


def arbitrator(c_slow, c_fast, x, x_star, gamma=0.1):
    F_slow = -np.log(np.abs(c_slow) + 1e-6)
    F_fast = -np.log(np.abs(c_fast) + 1e-6)
    SAM_val = x[2]
    arousal = x[3]
    pi_fast = sigmoid(2.0 * (SAM_val + arousal - 0.5))
    pi_slow = 1 - pi_fast
    penalty = gamma * np.sum((x - x_star) ** 2)
    a = pi_slow * F_slow + pi_fast * F_fast + penalty
    return a, pi_fast, pi_slow


def observation(a, e_t, e_tp1, x, action_quality):
    """Saciedad proporcional a calidad de acción × acoplamiento con cada capa."""
    q = action_quality  # en [0, 1]
    return np.array([
        q * 0.3,                        # rho_A: saciedad atencional si acción fue relevante
        q * 0.5,                        # rho_P: reducción de error de predicción
        q * 0.4 + abs(x[2]) * 0.2,      # rho_H: recupera homeostasis energética
        q * 0.6,                        # rho_E: valencia post-acción
        q * 0.3,                        # rho_N: coste ruta justificado
        q * 0.8                         # rho_C: utilidad cognitiva conseguida
    ])


def update_x(x, a, e_t, e_tp1, lambdas, alphas, W, x_star, kappa, action_quality):
    """
    Dinámica regulatoria REAL:
    x_{t+1} = x_t - kappa * (x_t - x*) + lambda - alpha * rho + W * phi(x)
    El primer término es el RETORNO AL SET-POINT (regulación homeostática).
    """
    rho = observation(a, e_t, e_tp1, x, action_quality)
    phi = np.tanh(x)
    # Retorno elástico al set-point + drift + saciedad + acoplamiento
    return x - kappa * (x - x_star) + lambdas - alphas * rho + W @ phi


def update_memory(M, a, x_new):
    M_new = np.roll(M, -1)
    M_new[-1] = np.tanh(a + np.mean(x_new) * 0.1)
    return M_new


# ============================================================
# Agente
# ============================================================

@dataclass
class AgentState:
    name: str
    x: np.ndarray = field(default_factory=lambda: np.zeros(6))
    M: np.ndarray = field(default_factory=lambda: np.zeros(10))
    x_H_hist: List[float] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)
    total_payoff: float = 0.0

    def decide(self, t, o_t, I_t, D_t, e_t, e_tp1, lambdas, alphas, W, x_star,
               kappa, opponent_last_action=None):
        x = self.x
        M = self.M

        o_tilde = attention_gate(o_t, I_t, D_t, M, x[0])
        h_sam = hormone_SAM(o_tilde, x[2])
        c_fast = neuro_fast(o_tilde, x, M)

        p_out = perception(o_tilde, x[1], M)
        e_out = emotion(p_out, I_t, M, x[3])
        hpa = np.mean(self.x_H_hist[-5:]) if len(self.x_H_hist) >= 5 else 0.0
        c_slow = cognition(e_out, p_out, x[5], M, hpa)

        a, pi_fast, pi_slow = arbitrator(c_slow, c_fast, x, x_star)

        # Decide: Cooperate (a > 0) or Defect (a <= 0)
        action = "C" if a > 0 else "D"

        # Action quality: high if action matched context (tit-for-tat proxy)
        if opponent_last_action is None:
            quality = 0.5  # neutral en ronda 1
        elif opponent_last_action == "C" and action == "C":
            quality = 0.9  # mutua cooperación
        elif opponent_last_action == "D" and action == "D":
            quality = 0.6  # defección justificada
        elif opponent_last_action == "D" and action == "C":
            quality = 0.1  # traicionado mientras coopero - mala
        else:
            quality = 0.7  # defecto frente a cooperador - utilidad alta corto plazo

        # Update con regulación real
        x_new = update_x(x, a, e_t, e_tp1, lambdas, alphas, W, x_star, kappa, quality)
        M_new = update_memory(M, a, x_new)

        # Record
        record = {
            "round": t,
            "action": action,
            "a_value": float(a),
            "pi_fast": float(pi_fast),
            "pi_slow": float(pi_slow),
            "c_fast": float(c_fast),
            "c_slow": float(c_slow),
            "h_sam": float(h_sam),
            "hpa": float(hpa),
            "valence": float(e_out[0]),
            "arousal": float(e_out[1]),
            "x": x.tolist(),
            "x_new": x_new.tolist(),
            "o_tilde_norm": float(np.linalg.norm(o_tilde)),
        }
        self.history.append(record)
        self.x = x_new
        self.M = M_new
        self.x_H_hist.append(float(x_new[2]))

        return action


# ============================================================
# IPD payoff
# ============================================================

def payoff(a1, a2):
    """T=5, R=3, P=1, S=0"""
    if a1 == "C" and a2 == "C":
        return 3, 3
    elif a1 == "C" and a2 == "D":
        return 0, 5
    elif a1 == "D" and a2 == "C":
        return 5, 0
    else:
        return 1, 1


# ============================================================
# Simulación
# ============================================================

def run_simulation(rounds=30, seed=42, noise_prob=0.1):
    np.random.seed(seed)

    # Parámetros compartidos (balanceados para mostrar regulación visible sin overshoot)
    # lambdas: drift basal (empuja hacia arriba por inactividad)
    lambdas = np.array([0.08, 0.07, 0.10, 0.08, 0.06, 0.10])
    # alphas: ganancia de saciedad (qué tanto reduce el déficit una buena acción)
    alphas = np.array([0.15, 0.18, 0.22, 0.18, 0.12, 0.25])
    # kappa: elasticidad hacia set-point (REGULACIÓN HOMEOSTÁTICA) — dominante
    kappa = np.array([0.35, 0.30, 0.40, 0.35, 0.30, 0.25])
    # Acoplamiento entre capas
    W = np.zeros((6, 6))
    W[2, 3] = 0.08   # H -> E: hormonas sesgan valencia
    W[3, 4] = 0.06   # E -> N: afecto baja precisión PFC
    W[4, 5] = 0.05   # N -> C: modo rápido limita planificación
    W[5, 3] = -0.04  # C -> E: reappraisal cognitivo baja valencia negativa
    W[5, 2] = -0.03  # C -> H: reappraisal crónico modula alostasis
    W[0, 5] = 0.04   # A -> C: atención alimenta cognición

    # Set-points homeostáticos (equilibrio deseado, no cero)
    x_star = np.array([0.3, 0.2, 0.3, 0.1, 0.2, 0.4])

    # Inicialización diferenciada (lejos del set-point para ver regulación)
    agent1 = AgentState(name="Agent_A", x=np.array([0.8, 0.5, 0.9, 0.7, 0.6, 1.0]))
    agent2 = AgentState(name="Agent_B", x=np.array([0.1, 0.2, 0.0, -0.2, 0.1, 0.0]))

    e_t = np.zeros(3)

    for t in range(rounds):
        # Perturbaciones
        if t == 10:
            # Traición forzada: agente B defecta sin importar decisión
            forced_a2 = "D"
        else:
            forced_a2 = None

        # Observación: lo que cada agente percibe del otro
        # Inicialmente ambos ven un entorno neutro
        o1 = np.array([0.5, 0.0, 0.0]) if t == 0 else np.array([
            1.0 if (agent2.history[-1]["action"] == "D" if agent2.history else "C") == "D" else -0.5,
            agent2.history[-1]["pi_fast"] if agent2.history else 0.0,
            agent2.history[-1]["arousal"] if agent2.history else 0.0,
        ])
        o2 = np.array([0.5, 0.0, 0.0]) if t == 0 else np.array([
            1.0 if (agent1.history[-1]["action"] == "D" if agent1.history else "C") == "D" else -0.5,
            agent1.history[-1]["pi_fast"] if agent1.history else 0.0,
            agent1.history[-1]["arousal"] if agent1.history else 0.0,
        ])

        I1, D1 = 0.2 + np.random.rand() * 0.1, 0.3
        I2, D2 = 0.2 + np.random.rand() * 0.1, 0.3

        e_tp1 = np.array([float(t) / rounds, 0.0, 0.0])

        opp_last_a1 = agent2.history[-1]["action"] if agent2.history else None
        opp_last_a2 = agent1.history[-1]["action"] if agent1.history else None

        a1 = agent1.decide(t, o1, I1, D1, e_t, e_tp1, lambdas, alphas, W, x_star, kappa, opp_last_a1)
        a2 = agent2.decide(t, o2, I2, D2, e_t, e_tp1, lambdas, alphas, W, x_star, kappa, opp_last_a2)

        if forced_a2:
            a2 = forced_a2
            agent2.history[-1]["action"] = a2
            agent2.history[-1]["note"] = "forced_defect"

        # Ruido
        if np.random.rand() < noise_prob:
            a1 = "D" if a1 == "C" else "C"
        if np.random.rand() < noise_prob:
            a2 = "D" if a2 == "C" else "C"

        p1, p2 = payoff(a1, a2)
        agent1.total_payoff += p1
        agent2.total_payoff += p2

        agent1.history[-1]["payoff"] = p1
        agent1.history[-1]["opponent_action"] = a2
        agent2.history[-1]["payoff"] = p2
        agent2.history[-1]["opponent_action"] = a1

        e_t = e_tp1

    return {
        "rounds": rounds,
        "seed": seed,
        "noise_prob": noise_prob,
        "agent1": {
            "name": agent1.name,
            "total_payoff": agent1.total_payoff,
            "history": agent1.history,
        },
        "agent2": {
            "name": agent2.name,
            "total_payoff": agent2.total_payoff,
            "history": agent2.history,
        },
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="simulation.json")
    parser.add_argument("--noise", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Simulando {args.rounds} rondas (seed={args.seed}, noise={args.noise})...")
    data = run_simulation(args.rounds, args.seed, args.noise)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Guardado en {args.output}")
    print(f"Payoffs finales: {data['agent1']['name']}={data['agent1']['total_payoff']}, "
          f"{data['agent2']['name']}={data['agent2']['total_payoff']}")
