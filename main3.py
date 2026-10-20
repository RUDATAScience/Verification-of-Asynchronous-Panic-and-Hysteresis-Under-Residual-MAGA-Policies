# ==============================================================================
# 3状態Blume-Capel選挙ポテンシャルゲーム 大規模モンテカルロ検証スイート (1M - 10M Trials)
# - 2024.12〜2026.10 Gallup & 全国世論調査実数キャリブレーション + 50万人デモショック
# - 核心的メカニズム: 「MAGA政策の残存 × 議会交代(上院53-47, 下院220-215反転)」のパニック
# - シナリオA (泥沼開示先行) / シナリオB (MAGA関税パニック先行) / シナリオC (忘却 vs 共鳴)
# ==============================================================================

import os
import time
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

# 論文準拠の擬似乱数シード固定 (完全再現性の保証)
RANDOM_SEED = 20261103
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = "blume_capel_scenarios_ABC_verification"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 0. 大規模計算パラメータ設定 (1,000,000 〜 10,000,000 試行)
# ------------------------------------------------------------------------------
N_TRIALS = 1_000_000        # 10,000,000 に変更すると1000万回試行を実行します
BATCH_SIZE = 200_000        # ベクトル化バッチサイズ (省メモリ高速演算用)
N_CHAINS = 4                # Gelman-Rubin R-hat 検証用の独立チェイン数

print("==========================================================================")
print(" 3-State Blume-Capel Electoral Potential Game: Scenarios A, B, C Suite")
print(f" Total Monte Carlo Trials per Regime : {N_TRIALS:,} (Batch: {BATCH_SIZE:,})")
print(" Initial Condition Prior             : X_{i,0} ~ Uniform U(0, 0.2)")
print("==========================================================================")

# ------------------------------------------------------------------------------
# 1. 2024年12月〜2026年10月 Gallup & 主要世論調査の実数データと磁場較正
# ------------------------------------------------------------------------------
polling_records = [
    # YearMonth, Approve(%), Disapprove(%), Ind_Approve(%), Cong_Approve(%), Protest_Scale_Index, Note
    ("2024-12", 49.0, 46.0, 44.0, 19.0, 0.05, "Presidential Transition Avg (Dec 2024)"),
    ("2025-01", 47.0, 48.0, 46.0, 21.0, 0.10, "Gallup Initial Poll (Jan 21-27, 2025: Net -1%)"),
    ("2025-02", 45.0, 51.0, 41.0, 22.0, 0.15, "Gallup Monthly (Feb 2025)"),
    ("2025-03", 44.5, 52.0, 39.0, 21.0, 0.22, "Gallup / Aggregate (Mar 2025: Broad tariff announcement)"),
    ("2025-04", 44.0, 53.0, 37.0, 20.0, 0.28, "Gallup (Apr 1-14, 2025: Q1 GDP -0.5% contraction)"),
    ("2025-05", 43.0, 53.0, 36.0, 19.0, 0.32, "Gallup (May 1-18, 2025)"),
    ("2025-06", 40.0, 57.0, 33.0, 18.0, 0.55, "Gallup (Jun 3-19, 2025: Nationwide protest wave 1)"),
    ("2025-07", 37.0, 58.0, 29.0, 17.0, 0.48, "Gallup Summer Low (Jul 2025)"),
    ("2025-08", 40.0, 56.0, 32.0, 17.0, 0.42, "Gallup (Aug 1-15, 2025)"),
    ("2025-09", 40.0, 56.0, 33.0, 16.0, 0.45, "Gallup (Sep 2-16, 2025)"),
    ("2025-10", 41.0, 54.0, 33.0, 15.0, 0.52, "Gallup (Oct 1-16, 2025: Federal shutdown friction)"),
    ("2025-11", 36.0, 60.0, 25.0, 14.0, 0.68, "Gallup 2nd-Term Low (Nov 2025: 36% Approve, Ind 25%, Cong 14%)"),
    ("2025-12", 36.0, 60.0, 25.0, 14.0, 0.64, "Gallup Year-End (Dec 2025: 36% Approve)"),
    ("2026-01", 37.0, 59.0, 27.0, 15.0, 0.60, "Aggregate Poll (Jan 2026: Tariff court battle & USD drop)"),
    ("2026-02", 36.5, 60.0, 26.0, 15.0, 0.62, "Aggregate Poll (Feb 2026)"),
    ("2026-03", 36.5, 60.5, 26.0, 14.5, 0.65, "Aggregate Poll (Mar 2026)"),
    ("2026-04", 36.0, 61.0, 25.5, 14.0, 0.70, "Aggregate Poll (Apr 2026)"),
    ("2026-05", 35.0, 62.0, 24.0, 13.5, 0.80, "Aggregate Poll (May 2026: Inflation spike & consumer pessimism)"),
    ("2026-06", 35.5, 61.5, 25.0, 14.0, 0.76, "Aggregate Poll (Jun 2026)"),
    ("2026-07", 36.0, 61.0, 25.5, 14.0, 0.74, "Aggregate Poll (Jul 2026)"),
    ("2026-08", 36.0, 61.5, 25.0, 13.5, 0.78, "Aggregate Poll (Aug 2026)"),
    ("2026-09", 36.0, 62.0, 25.0, 13.5, 0.85, "Reuters/Ipsos, Pew Avg (Sep 2026: Net -26% to -32%)"),
    ("2026-10", 35.5, 64.5, 24.0, 13.0, 1.00, "Oct 2026 Pre-Midterm: Net -29% & 500,000 Anti-Trump Protest Shock")
]

df_polls = pd.DataFrame(
    polling_records,
    columns=[
        "YearMonth", "Approve_Pct", "Disapprove_Pct",
        "Ind_Approve_Pct", "Cong_Approve_Pct",
        "Protest_Scale_Index", "Note"
    ]
)
df_polls["Net_Approval_Pct"] = df_polls["Approve_Pct"] - df_polls["Disapprove_Pct"]
base_net_jan25 = df_polls.loc[df_polls["YearMonth"] == "2025-01", "Net_Approval_Pct"].values[0]

# 実数支持率の下落と50万人デモ動員から負の外部磁場ショック -Δh_anti を算出
df_polls["Delta_h_Anti_Shock"] = (
    0.0042 * (df_polls["Net_Approval_Pct"] - base_net_jan25)
    - 0.048 * df_polls["Protest_Scale_Index"]
)
df_polls.to_csv(f"{OUTPUT_DIR}/01_empirical_polling_gallup_2024_2026.csv", index=False)

DELTA_H_OCT2026 = float(df_polls.iloc[-1]["Delta_h_Anti_Shock"])  # -0.1656

# ------------------------------------------------------------------------------
# 2. 米国全50州ベースラインパラメータ (論文 Table 6 準拠)
# ------------------------------------------------------------------------------
states_data = [
    # State, EV, Tier, Emp_Rep_Share(%), b_k, s_swing, w_work, w_est, w_truth, Sen_Up_2026
    ("PA", 19, "Battleground", 48.6, -0.015, 0.58, 0.41, 0.31, 0.28, False),
    ("MI", 15, "Battleground", 47.6, -0.022, 0.58, 0.42, 0.30, 0.28, True),
    ("WI", 10, "Battleground", 48.5, -0.020, 0.58, 0.42, 0.30, 0.28, False),
    ("GA", 16, "Battleground", 49.3, -0.005, 0.56, 0.40, 0.32, 0.28, True),
    ("NC", 16, "Battleground", 50.6, +0.006, 0.56, 0.41, 0.31, 0.28, True),
    ("AZ", 11, "Battleground", 49.2, -0.008, 0.57, 0.39, 0.31, 0.30, False),
    ("NV",  6, "Battleground", 48.1, -0.014, 0.59, 0.40, 0.30, 0.30, False),
    ("MN", 10, "Lean",         41.0, -0.085, 0.55, 0.36, 0.33, 0.31, True),
    ("NH",  4, "Lean",         42.9, -0.070, 0.56, 0.35, 0.33, 0.32, True),
    ("VA", 13, "Lean",         39.0, -0.115, 0.54, 0.34, 0.36, 0.30, True),
    ("ME",  4, "Lean",         41.3, -0.078, 0.57, 0.36, 0.32, 0.32, True),
    ("NM",  5, "Lean",         40.0, -0.095, 0.55, 0.36, 0.33, 0.31, True),
    ("OH", 17, "Lean",         61.7, +0.115, 0.52, 0.46, 0.28, 0.26, True),
    ("IA",  6, "Lean",         62.5, +0.120, 0.52, 0.46, 0.28, 0.26, True),
    ("FL", 30, "Lean",         59.8, +0.090, 0.51, 0.44, 0.29, 0.27, True),
    ("TX", 40, "Lean",         60.7, +0.098, 0.51, 0.45, 0.28, 0.27, True),
    ("AL",  9, "Solid Red",    81.8, +0.340, 0.40, 0.50, 0.26, 0.24, True),
    ("AK",  3, "Solid Red",    65.7, +0.155, 0.50, 0.46, 0.27, 0.27, True),
    ("AR",  6, "Solid Red",    83.8, +0.370, 0.39, 0.51, 0.25, 0.24, True),
    ("ID",  4, "Solid Red",    84.6, +0.395, 0.38, 0.51, 0.25, 0.24, True),
    ("IN", 11, "Solid Red",    72.0, +0.225, 0.45, 0.48, 0.27, 0.25, False),
    ("KS",  6, "Solid Red",    69.7, +0.200, 0.46, 0.47, 0.28, 0.25, True),
    ("KY",  8, "Solid Red",    81.8, +0.345, 0.40, 0.50, 0.26, 0.24, True),
    ("LA",  8, "Solid Red",    73.8, +0.240, 0.44, 0.48, 0.27, 0.25, True),
    ("MS",  6, "Solid Red",    73.0, +0.232, 0.44, 0.48, 0.27, 0.25, True),
    ("MO", 10, "Solid Red",    70.2, +0.205, 0.46, 0.47, 0.28, 0.25, False),
    ("MT",  4, "Solid Red",    71.7, +0.225, 0.45, 0.48, 0.27, 0.25, True),
    ("NE",  5, "Solid Red",    73.8, +0.245, 0.44, 0.48, 0.27, 0.25, True),
    ("ND",  3, "Solid Red",    84.6, +0.402, 0.38, 0.52, 0.24, 0.24, False),
    ("OK",  7, "Solid Red",    84.6, +0.415, 0.37, 0.52, 0.24, 0.24, True),
    ("SC",  9, "Solid Red",    67.7, +0.180, 0.47, 0.47, 0.28, 0.25, True),
    ("SD",  3, "Solid Red",    81.8, +0.340, 0.40, 0.50, 0.26, 0.24, True),
    ("TN", 11, "Solid Red",    80.2, +0.335, 0.40, 0.50, 0.26, 0.24, True),
    ("UT",  6, "Solid Red",    75.8, +0.280, 0.43, 0.49, 0.26, 0.25, False),
    ("WV",  4, "Solid Red",    84.6, +0.460, 0.36, 0.53, 0.24, 0.23, True),
    ("WY",  3, "Solid Red",    84.6, +0.490, 0.35, 0.54, 0.23, 0.23, True),
    ("CA", 54, "Solid Blue",   16.0, -0.390, 0.42, 0.28, 0.38, 0.34, False),
    ("CO", 10, "Solid Blue",   34.4, -0.175, 0.52, 0.32, 0.35, 0.33, True),
    ("CT",  7, "Solid Blue",   26.0, -0.295, 0.46, 0.30, 0.37, 0.33, False),
    ("DE",  3, "Solid Blue",   28.0, -0.265, 0.47, 0.31, 0.36, 0.33, True),
    ("HI",  4, "Solid Blue",   16.0, -0.382, 0.42, 0.28, 0.38, 0.34, False),
    ("IL", 19, "Solid Blue",   28.0, -0.258, 0.47, 0.31, 0.36, 0.33, True),
    ("MD", 10, "Solid Blue",   16.0, -0.400, 0.41, 0.28, 0.39, 0.33, False),
    ("MA", 11, "Solid Blue",   15.0, -0.435, 0.40, 0.27, 0.39, 0.34, True),
    ("NJ", 14, "Solid Blue",   30.0, -0.245, 0.48, 0.31, 0.36, 0.33, True),
    ("NY", 28, "Solid Blue",   21.7, -0.345, 0.44, 0.29, 0.38, 0.33, False),
    ("OR",  8, "Solid Blue",   30.2, -0.225, 0.49, 0.31, 0.35, 0.34, True),
    ("RI",  4, "Solid Blue",   21.9, -0.335, 0.44, 0.29, 0.38, 0.33, True),
    ("VT",  3, "Solid Blue",   15.0, -0.450, 0.40, 0.27, 0.39, 0.34, False),
    ("WA", 12, "Solid Blue",   26.1, -0.282, 0.46, 0.30, 0.36, 0.34, False),
]

df_states = pd.DataFrame(
    states_data,
    columns=["State", "EV", "Tier", "Emp_Rep_Share", "b_k", "s_swing", "w_work", "w_est", "w_truth", "Sen_Up_2026"]
)
N_STATES = len(df_states)

b_vec = df_states["b_k"].values
s_swing_vec = df_states["s_swing"].values
w_work_vec = df_states["w_work"].values
w_est_vec = df_states["w_est"].values
w_truth_vec = df_states["w_truth"].values
ev_vec = df_states["EV"].values
emp_share_vec = df_states["Emp_Rep_Share"].values / 100.0
sen_up_mask = df_states["Sen_Up_2026"].values

# 州間ネットワーク指数類似度結合行列 W_inter,kl (論文 式26)
diff_b = np.abs(b_vec[:, None] - b_vec[None, :])
W_inter = np.exp(-diff_b / 0.15)
np.fill_diagonal(W_inter, 0.0)
W_inter /= W_inter.sum(axis=1, keepdims=True)

# ------------------------------------------------------------------------------
# 3. 6次Landau自由エネルギー展開と三重臨界点の厳密検証 (論文 式18-25)
# ------------------------------------------------------------------------------
def compute_exact_landau_6th(J_eff, beta, D, h_asym):
    u = np.exp(beta * D)
    h1_eff = (2.0 * u / (2.0 * u + 1.0)) * h_asym
    a2 = (J_eff / (2.0 * (2.0 * u + 1.0))) * (1.0 + 2.0 * u * (1.0 - beta * J_eff))
    a4 = (J_eff**4 * beta**3 * u * (4.0 * u - 1.0)) / (12.0 * (2.0 * u + 1.0)**2)
    a6 = -(J_eff**6 * beta**5 * u * (64.0 * u**2 - 26.0 * u + 1.0)) / (360.0 * (2.0 * u + 1.0)**3)
    return u, h1_eff, a2, a4, a6

landau_records = []
for scen_label, J_val, h_val, d_shift, b_ratio in [
    ("Baseline_2026_Midterm_Dem_Flip",           1.05, -0.055, +0.08, 0.990),
    ("Exact_Tricritical_Point (a2=a4=0)",        1.15,  0.000,  0.00, 1.000),
    ("Scenario_A_Epstein_to_War_to_Trilat",      1.15, +0.065, -0.08, 1.010),
    ("Scenario_B_MAGA_Tariff_Panic_First",       1.25, +0.080, -0.09, 1.012),
    ("Scenario_C1_Dispersed_Decay (No_Collapse)",1.05, +0.030, +0.04, 0.995),
    ("Scenario_C2_Clustered_Resonance_Burst",    1.25, +0.085, -0.10, 1.015),
]:
    beta_t = 3.0 / J_val
    D_t = -np.log(4.0) / beta_t
    D_act = D_t + d_shift
    beta_act = beta_t * b_ratio
    u_v, h1_v, a2_v, a4_v, a6_v = compute_exact_landau_6th(J_val, beta_act, D_act, h_val)
    landau_records.append({
        "Scenario": scen_label, "J_eff": J_val, "D_crit": D_t, "D_actual": D_act,
        "beta_actual": beta_act, "u": u_v, "h1_eff": h1_v, "a2": a2_v, "a4": a4_v, "a6": a6_v
    })
df_landau = pd.DataFrame(landau_records)
df_landau.to_csv(f"{OUTPUT_DIR}/02_tricritical_6th_order_landau_coefficients.csv", index=False)

# ------------------------------------------------------------------------------
# 4. 非同期パルス・ヒステリシス時間発展シミュレーション (シナリオA, B, C1, C2 の比較)
# ------------------------------------------------------------------------------
def simulate_asynchronous_pulse_dynamics(T_max=100):
    t_steps = np.arange(T_max)
    records = []

    # 各シナリオにおけるパルス発生時刻 (t_ep, t_war, t_tri, t_panic) と減衰緩和時定数 tau_decay
    # 論文の3ショック + MAGA政策×ねじれ議会のパニック項
    pulse_configs = {
        "Scenario_A (Epstein -> War -> Trilat)": {
            "pulses": [("ep", 12), ("war", 38), ("tri", 62), ("panic", 48)],
            "tau_relax": 45.0, "memory_retention": 0.88
        },
        "Scenario_B (MAGA Tariff/Policy Panic -> Trilat -> Epstein)": {
            "pulses": [("panic", 10), ("tri", 32), ("war", 52), ("ep", 68)],
            "tau_relax": 55.0, "memory_retention": 0.94
        },
        "Scenario_C1 (Dispersed: dt >> tau_relax, Shock Decay)": {
            "pulses": [("ep", 10), ("war", 48), ("tri", 85), ("panic", 65)],
            "tau_relax": 11.0, "memory_retention": 0.25
        },
        "Scenario_C2 (Clustered Resonance: dt < tau_relax, Late Burst)": {
            "pulses": [("panic", 52), ("ep", 56), ("war", 60), ("tri", 64)],
            "tau_relax": 40.0, "memory_retention": 0.92
        }
    }

    for scen_name, cfg in pulse_configs.items():
        # t=0: 2026年中間選挙で民主党が上下両院を奪還した直後の状態 (m < 0, 民主党リード)
        m_w, m_e, m_t = +0.36, -0.58, -0.46
        q_w, q_e, q_t =  0.84,  0.90,  0.86
        mem_est, mem_truth = 0.0, 0.0
        beta_d = 2.45
        tau = cfg["tau_relax"]
        ret_rate = cfg["memory_retention"]

        for t in t_steps:
            # 減衰記憶カーネル K(t - t_k) による各ショックの実効強度計算
            I_shocks = {"ep": 0.0, "war": 0.0, "tri": 0.0, "panic": 0.0}
            for s_type, t_onset in cfg["pulses"]:
                if t >= t_onset:
                    dt = t - t_onset
                    # 立ち上がり(時定数2.0) × 指数減衰カーネル exp(-dt / tau_relax)
                    env = (1.0 - np.exp(-dt / 2.0)) * np.exp(-dt / tau)
                    I_shocks[s_type] = env

            s_ep, s_war, s_tri, s_pan = I_shocks["ep"], I_shocks["war"], I_shocks["tri"], I_shocks["panic"]

            m_tot = 0.21 * (+1.0) + 0.21 * (-1.0) + 0.24 * m_w + 0.18 * m_e + 0.16 * m_t
            q_tot = 0.42 * 0.92 + 0.24 * q_w + 0.18 * q_e + 0.16 * q_t

            # 1. 労働者・同調企業層 (c = work): MAGA関税・減税の防衛的ロックイン (D_work > 0)
            h_w = 0.36 + 0.25 * s_pan + 0.16 * s_tri + 0.10 * s_war + 0.55 * m_tot
            D_w = 0.48 + 0.35 * s_pan + 0.12 * s_tri - 0.06 * s_ep

            # 2. エスタブリッシュメント層 (c = est): MAGA政策残存×ねじれ議会の麻痺で D_est < D_t へ崩壊
            h_e = -0.44 + 0.12 * s_pan + 0.10 * s_tri + 0.08 * s_war + 0.55 * m_tot
            D_e = (
                0.62
                - 0.62 * s_pan
                - 0.46 * s_tri
                - 0.35 * s_war
                - 0.28 * s_ep
                - 0.55 * mem_est
            )

            # 3. 真実追求・無党派層 (c = truth): スキャンダルと超党派機能不全で s=0 (棄権) へ雪崩
            h_t = -0.34 - 0.05 * s_ep + 0.06 * s_war + 0.55 * m_tot
            D_t_f = (
                0.40
                - 1.20 * s_ep
                - 0.48 * s_pan
                - 0.30 * s_war
                - 0.28 * s_tri
                - 0.60 * mem_truth
            )

            m_w_n = 2.0 * np.sinh(beta_d * h_w) / (np.exp(-beta_d * D_w) + 2.0 * np.cosh(beta_d * h_w))
            q_w_n = 2.0 * np.cosh(beta_d * h_w) / (np.exp(-beta_d * D_w) + 2.0 * np.cosh(beta_d * h_w))

            m_e_n = 2.0 * np.sinh(beta_d * h_e) / (np.exp(-beta_d * D_e) + 2.0 * np.cosh(beta_d * h_e))
            q_e_n = 2.0 * np.cosh(beta_d * h_e) / (np.exp(-beta_d * D_e) + 2.0 * np.cosh(beta_d * h_e))

            m_t_n = 2.0 * np.sinh(beta_d * h_t) / (np.exp(-beta_d * D_t_f) + 2.0 * np.cosh(beta_d * h_t))
            q_t_n = 2.0 * np.cosh(beta_d * h_t) / (np.exp(-beta_d * D_t_f) + 2.0 * np.cosh(beta_d * h_t))

            # 復帰点記憶 (Return-Point Memory) の破壊とヒステリシス蓄積
            mem_est = max(ret_rate * mem_est, 0.90 * (1.0 - q_e_n))
            mem_truth = max(ret_rate * mem_truth, 0.92 * (1.0 - q_t_n))

            m_w = 0.25 * m_w + 0.75 * m_w_n
            q_w = 0.25 * q_w + 0.75 * q_w_n
            m_e = 0.25 * m_e + 0.75 * m_e_n
            q_e = 0.25 * q_e + 0.75 * q_e_n
            m_t = 0.25 * m_t + 0.75 * m_t_n
            q_t = 0.25 * q_t + 0.75 * q_t_n

            R_eff = 0.5 * (1.0 + m_tot / q_tot) * 100.0
            records.append({
                "Scenario": scen_name, "Step": t,
                "Net_Magnetization_m": m_tot,
                "Total_Turnout_q_Pct": q_tot * 100.0,
                "Rep_Effective_Share_Pct": R_eff,
                "Turnout_Work_Pct": q_w * 100.0,
                "Turnout_Est_Pct": q_e * 100.0,
                "Turnout_Truth_Pct": q_t * 100.0,
                "D_work": D_w, "D_est": D_e, "D_truth": D_t_f
            })
    return pd.DataFrame(records)

df_pulse_traj = simulate_asynchronous_pulse_dynamics()
df_pulse_traj.to_csv(f"{OUTPUT_DIR}/03_scenarios_ABC_asynchronous_hysteresis_trajectories.csv", index=False)

# ------------------------------------------------------------------------------
# 5. 100万〜1,000万回試行 ベクトル化モンテカルロ・アンサンブルエンジン
# ------------------------------------------------------------------------------
def execute_massive_mc_ensemble(regime_key, n_trials=N_TRIALS, batch_size=BATCH_SIZE):
    n_batches = int(np.ceil(n_trials / batch_size))
    done_trials = 0

    # インクリメンタル統計量アキュムレータ (50州分)
    sum_m1 = np.zeros(N_STATES, dtype=np.float64)
    sum_m2 = np.zeros(N_STATES, dtype=np.float64)
    sum_m3 = np.zeros(N_STATES, dtype=np.float64)
    sum_m4 = np.zeros(N_STATES, dtype=np.float64)

    sum_q1 = np.zeros(N_STATES, dtype=np.float64)
    sum_R1 = np.zeros(N_STATES, dtype=np.float64)
    sum_R2 = np.zeros(N_STATES, dtype=np.float64)
    sum_win = np.zeros(N_STATES, dtype=np.float64)

    # 階層別結晶場 D_c と投票率 q_c の記録用
    sum_D_work = np.zeros(N_STATES, dtype=np.float64)
    sum_D_est  = np.zeros(N_STATES, dtype=np.float64)
    sum_D_truth = np.zeros(N_STATES, dtype=np.float64)
    sum_q_work = np.zeros(N_STATES, dtype=np.float64)
    sum_q_est  = np.zeros(N_STATES, dtype=np.float64)
    sum_q_truth = np.zeros(N_STATES, dtype=np.float64)

    # 初期値 X_{i,0} ~ U(0, 0.2) の1次Sobol感度 S_1 計算用
    sum_x0 = np.zeros(N_STATES, dtype=np.float64)
    sum_x0_2 = np.zeros(N_STATES, dtype=np.float64)
    sum_x0_m = np.zeros(N_STATES, dtype=np.float64)

    # Gelman-Rubin R-hat チェイン別集計
    ch_sum_m = np.zeros((N_CHAINS, N_STATES), dtype=np.float64)
    ch_sum_m2 = np.zeros((N_CHAINS, N_STATES), dtype=np.float64)
    ch_counts = np.zeros(N_CHAINS, dtype=np.int64)

    ev_samples_list = []
    sen_dem_list = []
    house_dem_list = []
    nat_q_list = []
    nat_R_list = []

    beta = 2.45
    J_intra = 0.62

    for _ in range(n_batches):
        B = min(batch_size, n_trials - done_trials)
        done_trials += B

        # 連続一様事前分布 X_{i,0} ~ U(0, 0.2) による独立初期化 (論文 式30)
        X_0 = np.random.uniform(0.0, 0.2, size=(B, N_STATES))
        m_curr = X_0.copy()
        q_curr = np.full((B, N_STATES), 0.85, dtype=np.float64)

        if regime_key == "0_Baseline_2026_Midterm_Dem_Flip":
            # 2026年10月: 支持率下落(35.5%) + 50万人反トランプデモ (-Δh_anti) による上下両院逆転
            xi_protest = np.random.gamma(shape=4.5, scale=0.22, size=(B, 1))
            econ_pain  = np.random.normal(loc=-0.12, scale=0.045, size=(B, 1))

            for _ in range(12):
                eta_h = np.random.normal(0.0, 0.028, size=(B, N_STATES))
                eta_D = np.random.normal(0.0, 0.030, size=(B, N_STATES))
                soc_m = J_intra * m_curr + 0.12 * (m_curr @ W_inter.T)

                h_work  = b_vec + 0.22 + 0.45 * econ_pain - 0.05 * xi_protest + eta_h
                D_work  = 0.15 - 0.38 * xi_protest + eta_D

                h_est   = b_vec - 0.52 + 0.55 * DELTA_H_OCT2026 - 0.22 * xi_protest + eta_h
                D_est   = 0.78 + 0.35 * xi_protest + eta_D

                h_truth = b_vec - 0.38 + 0.85 * DELTA_H_OCT2026 - 0.28 * xi_protest + eta_h
                D_truth = 0.52 + 0.20 * xi_protest + eta_D

                m_w = 2.0 * np.sinh(beta * (soc_m + h_work)) / (np.exp(-beta * D_work) + 2.0 * np.cosh(beta * (soc_m + h_work)))
                q_w = 2.0 * np.cosh(beta * (soc_m + h_work)) / (np.exp(-beta * D_work) + 2.0 * np.cosh(beta * (soc_m + h_work)))

                m_e = 2.0 * np.sinh(beta * (soc_m + h_est)) / (np.exp(-beta * D_est) + 2.0 * np.cosh(beta * (soc_m + h_est)))
                q_e = 2.0 * np.cosh(beta * (soc_m + h_est)) / (np.exp(-beta * D_est) + 2.0 * np.cosh(beta * (soc_m + h_est)))

                m_t = 2.0 * np.sinh(beta * (soc_m + h_truth)) / (np.exp(-beta * D_truth) + 2.0 * np.cosh(beta * (soc_m + h_truth)))
                q_t = 2.0 * np.cosh(beta * (soc_m + h_truth)) / (np.exp(-beta * D_truth) + 2.0 * np.cosh(beta * (soc_m + h_truth)))

                m_sw = w_work_vec * m_w + w_est_vec * m_e + w_truth_vec * m_t
                q_sw = w_work_vec * q_w + w_est_vec * q_e + w_truth_vec * q_t

                m_curr = 0.32 * m_curr + 0.68 * ((1.0 - s_swing_vec) * b_vec + s_swing_vec * m_sw)
                q_curr = 0.32 * q_curr + 0.68 * ((1.0 - s_swing_vec) * 0.92 + s_swing_vec * q_sw)

        else:
            # 議会反転後の3つの非対称ガンマショック (論文 式35-37) + MAGA政策×ねじれ議会パニック項
            xi_war     = np.random.gamma(shape=2.5, scale=0.22, size=(B, 1)) # E=0.550
            xi_epstein = np.random.gamma(shape=3.0, scale=0.28, size=(B, 1)) # E=0.840
            xi_trilat  = np.random.gamma(shape=2.8, scale=0.24, size=(B, 1)) # E=0.672
            xi_panic   = np.random.gamma(shape=3.5, scale=0.25, size=(B, 1)) # E=0.875

            # シナリオ別の非同期ステージ重み (w_war, w_epstein, w_trilat, w_panic) と記憶減衰率
            if regime_key == "1_Cell3_Simultaneous_Benchmark":
                stages = [(1.0, 1.0, 1.0, 0.25)] * 3
                mem_decay = 0.50
            elif regime_key == "2_Scenario_A_Epstein_War_Trilat":
                # 第1波: 議会召喚・スキャンダル開示 -> 第2波: 戦争終結ディール強行 -> 第3波: 三国間協定
                stages = [
                    (0.15, 1.35, 0.10, 0.35),
                    (1.25, 1.05, 0.35, 0.75),
                    (1.00, 1.00, 1.25, 1.05)
                ]
                mem_decay = 0.78
            elif regime_key == "3_Scenario_B_MAGA_Tariff_Panic_First":
                # 第1波: MAGA関税・規制のねじれパニック先行 -> 第2波: 三国間安保再編 -> 第3波: スキャンダル暴露
                stages = [
                    (0.35, 0.20, 0.45, 1.45),
                    (0.85, 0.55, 1.35, 1.35),
                    (1.10, 1.20, 1.15, 1.30)
                ]
                mem_decay = 0.90
            elif regime_key == "4_Scenario_C1_Dispersed_Forgetting":
                # 時間間隔 Δt >> τ_relax により先行ショックが減衰・忘却されるケース
                stages = [
                    (0.10, 0.95, 0.10, 0.20),
                    (0.85, 0.25, 0.15, 0.25),
                    (0.25, 0.15, 0.85, 0.30)
                ]
                mem_decay = 0.15
            elif regime_key == "5_Scenario_C2_Clustered_Resonance":
                # 選挙直前に Δt < τ_relax で集中投下され非線形共鳴を起こすケース
                stages = [
                    (0.20, 0.20, 0.20, 0.30),
                    (0.95, 1.15, 0.95, 1.20),
                    (1.20, 1.25, 1.30, 1.40)
                ]
                mem_decay = 0.92

            mem_est   = np.zeros((B, N_STATES), dtype=np.float64)
            mem_truth = np.zeros((B, N_STATES), dtype=np.float64)

            for (w_w, w_e, w_t, w_p) in stages:
                for _ in range(4):
                    eta_h = np.random.normal(0.0, 0.035, size=(B, N_STATES))
                    eta_D = np.random.normal(0.0, 0.040, size=(B, N_STATES))
                    soc_m = J_intra * m_curr + 0.14 * (m_curr @ W_inter.T)

                    # 労働者階級・同調企業層: MAGA経済利得 Δπ_work^econ > 0 の防衛的ロックイン
                    h_work = b_vec + 0.50 + 0.14 * (w_w * xi_war) + 0.22 * (w_t * xi_trilat) + 0.18 * (w_p * xi_panic) + eta_h
                    D_work = 0.65 + 0.26 * (w_p * xi_panic) - 0.08 * (w_e * xi_epstein) + eta_D

                    # エスタブリッシュメント層: 制度的パニックにより D_est < D_t = -0.4621 へ崩壊
                    h_est = b_vec - 0.35 + 0.10 * (w_w * xi_war) + 0.12 * (w_t * xi_trilat) + 0.08 * (w_p * xi_panic) + eta_h
                    D_est = (
                        0.44
                        - 0.40 * (w_w * xi_war)
                        - 0.34 * (w_e * xi_epstein)
                        - 0.46 * (w_t * xi_trilat)
                        - 0.50 * (w_p * xi_panic)
                        - 0.38 * mem_est
                        + eta_D
                    )

                    # 真実追求・無党派層: 全方位シニシズム効果により D_truth << D_t へ崩壊
                    h_truth = b_vec - 0.18 + 0.06 * (w_w * xi_war) - 0.05 * (w_e * xi_epstein) + eta_h
                    D_truth = (
                        0.32
                        - 0.28 * (w_w * xi_war)
                        - 1.15 * (w_e * xi_epstein)
                        - 0.25 * (w_t * xi_trilat)
                        - 0.38 * (w_p * xi_panic)
                        - 0.42 * mem_truth
                        + eta_D
                    )

                    m_w = 2.0 * np.sinh(beta * (soc_m + h_work)) / (np.exp(-beta * D_work) + 2.0 * np.cosh(beta * (soc_m + h_work)))
                    q_w = 2.0 * np.cosh(beta * (soc_m + h_work)) / (np.exp(-beta * D_work) + 2.0 * np.cosh(beta * (soc_m + h_work)))

                    m_e = 2.0 * np.sinh(beta * (soc_m + h_est)) / (np.exp(-beta * D_est) + 2.0 * np.cosh(beta * (soc_m + h_est)))
                    q_e = 2.0 * np.cosh(beta * (soc_m + h_est)) / (np.exp(-beta * D_est) + 2.0 * np.cosh(beta * (soc_m + h_est)))

                    m_t = 2.0 * np.sinh(beta * (soc_m + h_truth)) / (np.exp(-beta * D_truth) + 2.0 * np.cosh(beta * (soc_m + h_truth)))
                    q_t = 2.0 * np.cosh(beta * (soc_m + h_truth)) / (np.exp(-beta * D_truth) + 2.0 * np.cosh(beta * (soc_m + h_truth)))

                    # ヒステリシス記憶の更新 (mem_decay が大きいほど復帰点記憶が破壊され棄権に留まる)
                    mem_est   = mem_decay * mem_est   + (1.0 - mem_decay) * (1.0 - q_e)
                    mem_truth = mem_decay * mem_truth + (1.0 - mem_decay) * (1.0 - q_t)

                    m_sw = w_work_vec * m_w + w_est_vec * m_e + w_truth_vec * m_t
                    q_sw = w_work_vec * q_w + w_est_vec * q_e + w_truth_vec * q_t

                    m_curr = 0.30 * m_curr + 0.70 * ((1.0 - s_swing_vec) * b_vec + s_swing_vec * m_sw)
                    q_curr = 0.30 * q_curr + 0.70 * ((1.0 - s_swing_vec) * 0.90 + s_swing_vec * q_sw)

        R_curr = 0.5 * (1.0 + m_curr / np.clip(q_curr, 1e-5, 1.0))
        win_rep = (R_curr >= 0.50).astype(np.float64)

        sum_m1 += m_curr.sum(axis=0)
        sum_m2 += (m_curr**2).sum(axis=0)
        sum_m3 += (m_curr**3).sum(axis=0)
        sum_m4 += (m_curr**4).sum(axis=0)

        sum_q1 += q_curr.sum(axis=0)
        sum_R1 += R_curr.sum(axis=0)
        sum_R2 += (R_curr**2).sum(axis=0)
        sum_win += win_rep.sum(axis=0)

        sum_D_work  += D_work.sum(axis=0)
        sum_D_est   += D_est.sum(axis=0)
        sum_D_truth += D_truth.sum(axis=0)
        sum_q_work  += q_w.sum(axis=0)
        sum_q_est   += q_e.sum(axis=0)
        sum_q_truth += q_t.sum(axis=0)

        sum_x0 += X_0.sum(axis=0)
        sum_x0_2 += (X_0**2).sum(axis=0)
        sum_x0_m += (X_0 * m_curr).sum(axis=0)

        splits = np.array_split(m_curr, N_CHAINS, axis=0)
        for c_idx, arr_c in enumerate(splits):
            ch_sum_m[c_idx] += arr_c.sum(axis=0)
            ch_sum_m2[c_idx] += (arr_c**2).sum(axis=0)
            ch_counts[c_idx] += arr_c.shape[0]

        ev_b = win_rep @ ev_vec
        nat_q_b = (q_curr @ ev_vec) / 535.0
        nat_R_b = (R_curr @ ev_vec) / 535.0

        dem_sen_up = ((1.0 - win_rep)[:, sen_up_mask]).sum(axis=1)
        sen_dem_b = np.clip(35.0 + dem_sen_up, 40.0, 62.0)
        house_dem_b = np.clip(np.round(212.0 + 380.0 * ((1.0 - nat_R_b) - 0.50)), 175.0, 265.0)

        keep = min(40_000, B)
        ev_samples_list.append(ev_b[:keep])
        sen_dem_list.append(sen_dem_b[:keep])
        house_dem_list.append(house_dem_b[:keep])
        nat_q_list.append(nat_q_b[:keep])
        nat_R_list.append(nat_R_b[:keep])

    N = float(done_trials)
    mu = sum_m1 / N
    m2 = sum_m2 / N
    m3 = sum_m3 / N
    m4 = sum_m4 / N

    var = np.maximum(m2 - mu**2, 1e-12)
    sigma = np.sqrt(var)

    mu3_c = m3 - 3.0 * mu * m2 + 2.0 * (mu**3)
    mu4_c = m4 - 4.0 * mu * m3 + 6.0 * (mu**2) * m2 - 3.0 * (mu**4)
    gamma3 = mu3_c / (sigma**3)
    kappa4 = (mu4_c / (var**2)) - 3.0

    snr_lambda = (mu**2) / var
    delta_NG = 4.0 * (mu / sigma) * gamma3 + kappa4
    U4_exact = 1.0 - (m4 / (3.0 * (m2**2)))
    U4_master = (2.0 * (snr_lambda**2) - delta_NG) / (3.0 * ((1.0 + snr_lambda)**2))

    # 1次Sobol感度 S_1(X_{i,0})
    mean_x0 = sum_x0 / N
    var_x0 = np.maximum((sum_x0_2 / N) - mean_x0**2, 1e-12)
    cov_x0_m = (sum_x0_m / N) - mean_x0 * mu
    sobol_S1 = (cov_x0_m**2) / (var_x0 * var)

    # Gelman-Rubin R-hat
    n_c = float(ch_counts[0])
    c_means = ch_sum_m / n_c
    c_vars = np.maximum((ch_sum_m2 / n_c) - c_means**2, 1e-12)
    W_gr = c_vars.mean(axis=0)
    B_gr = n_c * np.var(c_means, axis=0, ddof=1)
    R_hat = np.sqrt((((n_c - 1.0) / n_c) * W_gr + (1.0 / n_c) * B_gr) / W_gr)

    q_mean = sum_q1 / N
    R_mean = sum_R1 / N
    R_var = np.maximum((sum_R2 / N) - R_mean**2, 1e-12)
    R_std = np.sqrt(R_var)
    win_prob = sum_win / N

    emp_win = (emp_share_vec >= 0.50).astype(np.float64)
    brier_i = (win_prob - emp_win)**2
    z_val = (emp_share_vec - R_mean) / R_std
    crps_i = R_std * (z_val * (2.0 * norm.cdf(z_val) - 1.0) + 2.0 * norm.pdf(z_val) - 1.0 / np.sqrt(np.pi))
    sigma_emp = 0.025
    kl_i = np.log(sigma_emp / R_std) + (R_var + (R_mean - emp_share_vec)**2) / (2.0 * sigma_emp**2) - 0.5

    df_out = df_states[["State", "EV", "Tier", "Emp_Rep_Share"]].copy()
    df_out["mu_m"] = mu
    df_out["sigma_m"] = sigma
    df_out["SNR_lambda"] = snr_lambda
    df_out["Skewness_gamma3"] = gamma3
    df_out["Kurtosis_kappa4"] = kappa4
    df_out["Delta_NG"] = delta_NG
    df_out["Binder_U4"] = U4_exact
    df_out["Binder_U4_Master"] = U4_master
    df_out["Mean_D_work"] = sum_D_work / N
    df_out["Mean_D_est"] = sum_D_est / N
    df_out["Mean_D_truth"] = sum_D_truth / N
    df_out["Turnout_work_Pct"] = (sum_q_work / N) * 100.0
    df_out["Turnout_est_Pct"] = (sum_q_est / N) * 100.0
    df_out["Turnout_truth_Pct"] = (sum_q_truth / N) * 100.0
    df_out["Turnout_Total_q_Pct"] = q_mean * 100.0
    df_out["Sim_Rep_Share_Pct"] = R_mean * 100.0
    df_out["Sim_Rep_Std_Pct"] = R_std * 100.0
    df_out["Rep_Win_Prob_Pct"] = win_prob * 100.0
    df_out["Brier_Score"] = brier_i
    df_out["CRPS"] = crps_i
    df_out["KL_Divergence"] = kl_i
    df_out["Sobol_S1_X0"] = sobol_S1
    df_out["Gelman_Rubin_Rhat"] = R_hat

    ev_all = np.concatenate(ev_samples_list)
    sen_all = np.concatenate(sen_dem_list)
    house_all = np.concatenate(house_dem_list)
    nat_q_all = np.concatenate(nat_q_list)
    nat_R_all = np.concatenate(nat_R_list)

    bg_mask = df_out["Tier"] == "Battleground"
    macro_info = {
        "Regime": regime_key,
        "N_Trials": int(N),
        "Prior_X0_Mean": float(mean_x0.mean()),
        "Mean_Sobol_S1": float(sobol_S1.mean()),
        "Max_Gelman_Rubin_Rhat": float(R_hat.max()),
        "Battleground_Rep_Share_Pct": float(df_out.loc[bg_mask, "Sim_Rep_Share_Pct"].mean()),
        "Battleground_Turnout_q_Pct": float(df_out.loc[bg_mask, "Turnout_Total_q_Pct"].mean()),
        "Battleground_D_est": float(df_out.loc[bg_mask, "Mean_D_est"].mean()),
        "Battleground_D_truth": float(df_out.loc[bg_mask, "Mean_D_truth"].mean()),
        "Expected_Rep_EV": float(ev_all.mean()),
        "Rep_Majority_Prob_Pct": float((ev_all >= 270).mean() * 100.0),
        "Expected_Dem_Senate": float(sen_all.mean()),
        "Expected_Rep_Senate": float(100.0 - sen_all.mean()),
        "Expected_Dem_House": float(house_all.mean()),
        "Expected_Rep_House": float(435.0 - house_all.mean()),
        "Mean_Brier": float(brier_i.mean()),
        "Mean_CRPS": float(crps_i.mean()),
        "Mean_KL_Div": float(kl_i.mean()),
        "Mean_Binder_U4": float(U4_exact.mean()),
        "EV_Samples": ev_all,
        "Nat_Turnout_Samples": nat_q_all * 100.0,
        "Nat_Rep_Share_Samples": nat_R_all * 100.0,
    }
    return df_out, macro_info

# ------------------------------------------------------------------------------
# 6. 全6レジーム（2026議会反転 + 同時ショック + シナリオA, B, C1, C2）の一括実行
# ------------------------------------------------------------------------------
t_start = time.time()
regime_keys = [
    "0_Baseline_2026_Midterm_Dem_Flip",
    "1_Cell3_Simultaneous_Benchmark",
    "2_Scenario_A_Epstein_War_Trilat",
    "3_Scenario_B_MAGA_Tariff_Panic_First",
    "4_Scenario_C1_Dispersed_Forgetting",
    "5_Scenario_C2_Clustered_Resonance",
]

ensemble_results = {}
summary_rows = []

for r_key in regime_keys:
    print(f"  [EXEC] Running {r_key} ({N_TRIALS:,} trials)...")
    df_st, mac_info = execute_massive_mc_ensemble(r_key, n_trials=N_TRIALS, batch_size=BATCH_SIZE)
    ensemble_results[r_key] = (df_st, mac_info)
    df_st.to_csv(f"{OUTPUT_DIR}/state_posterior_{r_key}.csv", index=False)
    summary_rows.append({k: v for k, v in mac_info.items() if not isinstance(v, np.ndarray)})

df_global_summary = pd.DataFrame(summary_rows)
df_global_summary.to_csv(f"{OUTPUT_DIR}/04_global_scenarios_ABC_verification_summary.csv", index=False)
print(f"[INFO] All 6 ensembles completed in {time.time() - t_start:.2f} seconds.")

# ------------------------------------------------------------------------------
# 7. 高解像度グラフ & 統合マスター・ダッシュボードの作成
# ------------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# --- Figure 1: シナリオA・B・C の非同期パルス応答・結晶場崩壊・投票率逆転トラジェクトリ ---
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10), dpi=180)
scen_palette = {
    "Scenario_A (Epstein -> War -> Trilat)": "#1f77b4",
    "Scenario_B (MAGA Tariff/Policy Panic -> Trilat -> Epstein)": "#d62728",
    "Scenario_C1 (Dispersed: dt >> tau_relax, Shock Decay)": "#2ca02c",
    "Scenario_C2 (Clustered Resonance: dt < tau_relax, Late Burst)": "#9467bd"
}

for s_name, grp in df_pulse_traj.groupby("Scenario"):
    col = scen_palette[s_name]
    axes1[0, 0].plot(grp["Step"], grp["Rep_Effective_Share_Pct"], lw=2.2, color=col, label=s_name)
    axes1[0, 1].plot(grp["Step"], grp["D_est"], lw=2.2, color=col, label=s_name)
    axes1[1, 0].plot(grp["Step"], grp["Turnout_Est_Pct"], lw=2.2, color=col, label=f"Est. ({s_name.split(' ')[0]})")
    axes1[1, 0].plot(grp["Step"], grp["Turnout_Truth_Pct"], lw=1.4, ls="--", color=col)
    axes1[1, 1].plot(grp["Total_Turnout_q_Pct"], grp["Rep_Effective_Share_Pct"], lw=2.0, color=col, label=s_name)

axes1[0, 0].axhline(50.0, color="black", ls="--", lw=1.2, label="50% Majority Line")
axes1[0, 0].set_title("(A) Effective Two-Party Share R(t) Across Scenarios A, B, C", fontweight="bold")
axes1[0, 0].set_xlabel("Time Step t (Post-2026 Midterm Congressional Flip)")
axes1[0, 0].set_ylabel("Republican Effective Vote Share (%)")
axes1[0, 0].grid(True, alpha=0.3)
axes1[0, 0].legend(fontsize=8)

axes1[0, 1].axhline(-np.log(4.0)/3.0, color="black", ls="--", lw=1.5, label="Tricritical Threshold D_t/J = -0.4621")
axes1[0, 1].set_title("(B) Establishment Crystal Field D_est(t) & Spinodal Crossing", fontweight="bold")
axes1[0, 1].set_xlabel("Time Step t")
axes1[0, 1].set_ylabel("Crystal Field D_est(t)")
axes1[0, 1].grid(True, alpha=0.3)
axes1[0, 1].legend(fontsize=8)

axes1[1, 0].set_title("(C) Stratum Turnout Collapse q_c*(t) (Solid: Est, Dashed: Truth)", fontweight="bold")
axes1[1, 0].set_xlabel("Time Step t")
axes1[1, 0].set_ylabel("Active Voter Turnout (%)")
axes1[1, 0].grid(True, alpha=0.3)
axes1[1, 0].legend(fontsize=8)

axes1[1, 1].axhline(50.0, color="black", ls="--", lw=1.2)
axes1[1, 1].set_title("(D) Phase-Space Hysteresis Loop: Total Turnout q* vs Vote Share R", fontweight="bold")
axes1[1, 1].set_xlabel("Total Active Voter Turnout q*(t) (%)")
axes1[1, 1].set_ylabel("Republican Effective Share R(t) (%)")
axes1[1, 1].grid(True, alpha=0.3)
axes1[1, 1].legend(fontsize=8)

plt.tight_layout()
fig1.savefig(f"{OUTPUT_DIR}/Fig1_Scenarios_ABC_Asynchronous_Hysteresis_Dynamics.png", dpi=180)
plt.close(fig1)

# --- Figure 2: 2024.12-2026.10 世論調査較正・上下両院反転・階層別ロックイン対比 ---
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.2), dpi=180)

# (2A) Gallup & Aggregate Polling Time Series
x_p = np.arange(len(df_polls))
axes2[0].plot(x_p, df_polls["Approve_Pct"], "o-", color="#1f77b4", lw=2, label="Approval (%) [Gallup/Agg]")
axes2[0].plot(x_p, df_polls["Disapprove_Pct"], "s-", color="#d62728", lw=2, label="Disapproval (%)")
axes2[0].plot(x_p, df_polls["Ind_Approve_Pct"], "^--", color="#2ca02c", lw=1.6, label="Independents Approval (%)")
axes2[0].plot(x_p, df_polls["Cong_Approve_Pct"], "d:", color="#7f7f7f", lw=1.5, label="Congress Approval (%)")
axes2[0].set_xticks(x_p[::3])
axes2[0].set_xticklabels(df_polls["YearMonth"].iloc[::3], rotation=35, fontsize=8)
axes2[0].set_title("(A) 2024.12-2026.10 Empirical Polling & 500k Protest Shock", fontweight="bold")
axes2[0].set_ylabel("Percentage (%)")
axes2[0].grid(True, alpha=0.3)
axes2[0].legend(fontsize=8)

# (2B) Stratum Crystal Fields D_c Comparison Across Regimes
reg_short = ["2026 Flip", "Cell3 Sim", "Scen A", "Scen B", "Scen C1", "Scen C2"]
d_work_vals = [ensemble_results[k][0]["Mean_D_work"].mean() for k in regime_keys]
d_est_vals  = [ensemble_results[k][0]["Mean_D_est"].mean() for k in regime_keys]
d_tru_vals  = [ensemble_results[k][0]["Mean_D_truth"].mean() for k in regime_keys]

x_r = np.arange(len(regime_keys))
w_b = 0.26
axes2[1].bar(x_r - w_b, d_work_vals, width=w_b, color="#d62728", label="Working/Corp D_work (Locked-in > 0)")
axes2[1].bar(x_r,       d_est_vals,  width=w_b, color="#1f77b4", label="Establishment D_est (Panic Collapse)")
axes2[1].bar(x_r + w_b, d_tru_vals,  width=w_b, color="#2ca02c", label="Truth-Seeking D_truth (Cynicism)")
axes2[1].axhline(-np.log(4.0)/3.0, color="purple", ls="--", lw=1.8, label="Tricritical D_t = -0.4621")
axes2[1].axhline(0.0, color="black", lw=0.8)
axes2[1].set_xticks(x_r)
axes2[1].set_xticklabels(reg_short, fontsize=8.5)
axes2[1].set_title("(B) Asymmetric Crystal Field Divergence (h_i vs D_i)", fontweight="bold")
axes2[1].set_ylabel("Mean Crystal Field D_c")
axes2[1].grid(True, axis="y", alpha=0.3)
axes2[1].legend(fontsize=7.8)

# (2C) Expected Electoral College & Win Probabilities Across Scenarios A, B, C
ev_means = [ensemble_results[k][1]["Expected_Rep_EV"] for k in regime_keys]
bar_cols = ["#2b5c8f", "#ff7f0e", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
bars = axes2[2].bar(x_r, ev_means, color=bar_cols, edgecolor="black", width=0.55)
axes2[2].axhline(270, color="black", ls="--", lw=1.8, label="270 Electoral College Majority")
for b_item, k_name in zip(bars, regime_keys):
    w_p = ensemble_results[k_name][1]["Rep_Majority_Prob_Pct"]
    axes2[2].text(b_item.get_x() + b_item.get_width()/2, b_item.get_height() + 5,
                  f"{b_item.get_height():.1f} EV\n({w_p:.1f}%)", ha="center", fontsize=8, fontweight="bold")
axes2[2].set_xticks(x_r)
axes2[2].set_xticklabels(reg_short, fontsize=8.5)
axes2[2].set_ylim(0, 380)
axes2[2].set_title("(C) Expected Electoral College Weight Across Scenarios", fontweight="bold")
axes2[2].set_ylabel("Expected Republican Electoral Votes (EV)")
axes2[2].grid(True, axis="y", alpha=0.3)
axes2[2].legend(fontsize=8.5)

plt.tight_layout()
fig2.savefig(f"{OUTPUT_DIR}/Fig2_Empirical_Calibration_and_Crystal_Field_Lockin.png", dpi=180)
plt.close(fig2)

# --- Figure 3: マスター・サマリー・ダッシュボード (全シナリオ・数理検証の統合図) ---
fig3 = plt.figure(figsize=(20, 12), dpi=180)
gs = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.32, wspace=0.25)

# Panel A: Scenarios A, B, C Asynchronous Trajectories
axA = fig3.add_subplot(gs[0, 0])
for s_name, grp in df_pulse_traj.groupby("Scenario"):
    axA.plot(grp["Step"], grp["Rep_Effective_Share_Pct"], lw=2, color=scen_palette[s_name], label=s_name)
axA.axhline(50.0, color="black", ls="--")
axA.set_title("(A) Asynchronous Shock Cascades (Scenarios A, B, C1, C2)", fontweight="bold")
axA.set_xlabel("Time Step t")
axA.set_ylabel("Battleground Rep Effective Share (%)")
axA.grid(True, alpha=0.3)
axA.legend(fontsize=7.2)

# Panel B: 6th-Order Landau Potential Landscapes
axB = fig3.add_subplot(gs[0, 1])
m_vec_grid = np.linspace(-0.95, 0.95, 300)
for _, r_l in df_landau.iterrows():
    F_curve = -r_l["h1_eff"] * m_vec_grid + r_l["a2"] * (m_vec_grid**2) + r_l["a4"] * (m_vec_grid**4) + r_l["a6"] * (m_vec_grid**6)
    axB.plot(m_vec_grid, F_curve, lw=1.9, label=r_l["Scenario"])
axB.axhline(0, color="black", ls=":", lw=0.8)
axB.set_title("(B) 6th-Order Landau Potentials F(m) = -Phi(m)", fontweight="bold")
axB.set_xlabel("Partisan Magnetization m")
axB.set_ylabel("Free Energy Density F(m)")
axB.grid(True, alpha=0.3)
axB.legend(fontsize=7.2)

# Panel C: Binder Cumulant U_4 vs Turnout q* (Scenario A vs B vs C1)
axC = fig3.add_subplot(gs[0, 2])
for r_k, lbl_k, col_k, mk_k in [
    ("2_Scenario_A_Epstein_War_Trilat", "Scenario A (Scandal First)", "#1f77b4", "o"),
    ("3_Scenario_B_MAGA_Tariff_Panic_First", "Scenario B (MAGA Panic First)", "#d62728", "^"),
    ("4_Scenario_C1_Dispersed_Forgetting", "Scenario C1 (Dispersed Decay)", "#2ca02c", "s"),
]:
    df_tmp = ensemble_results[r_k][0]
    axC.scatter(df_tmp["Turnout_Total_q_Pct"], df_tmp["Binder_U4"], c=col_k, marker=mk_k, s=42, alpha=0.8, edgecolors="k", label=lbl_k)
axC.axhline(2.0/3.0, color="darkred", ls="--", lw=1.2, label="Ordered Limit (2/3)")
axC.set_title("(C) 4th-Order Binder Cumulant U_4 vs Active Turnout q*", fontweight="bold")
axC.set_xlabel("Active Voter Turnout q* (%)")
axC.set_ylabel("Binder Cumulant U_4")
axC.grid(True, alpha=0.3)
axC.legend(fontsize=7.8)

# Panel D: Electoral College Posterior Histogram (2026 Flip vs Scenario A vs Scenario B)
axD = fig3.add_subplot(gs[1, 0])
for r_k, lbl_k, col_k in [
    ("0_Baseline_2026_Midterm_Dem_Flip", "2026 Midterm Flip (Dem Congress)", "#2b5c8f"),
    ("4_Scenario_C1_Dispersed_Forgetting", "Scenario C1 (Dispersed Forgetting)", "#2ca02c"),
    ("2_Scenario_A_Epstein_War_Trilat", "Scenario A (Scandal -> War -> Trilat)", "#1f77b4"),
    ("3_Scenario_B_MAGA_Tariff_Panic_First", "Scenario B (MAGA Panic First)", "#d62728"),
]:
    ev_arr = ensemble_results[r_k][1]["EV_Samples"]
    axD.hist(ev_arr, bins=28, density=True, alpha=0.52, color=col_k, label=f"{lbl_k} ({ev_arr.mean():.1f} EV)")
axD.axvline(270, color="black", ls="--", lw=1.8)
axD.set_title(f"(D) 50-State Electoral College Posterior ({N_TRIALS:,} Trials)", fontweight="bold")
axD.set_xlabel("Republican Electoral Weight (out of 535)")
axD.set_ylabel("Density")
axD.grid(True, alpha=0.3)
axD.legend(fontsize=7.2)

# Panel E: Initial Value Independence Sobol Sensitivity S_1(X_0)
axE = fig3.add_subplot(gs[1, 1])
df_scA = ensemble_results["2_Scenario_A_Epstein_War_Trilat"][0]
df_scB = ensemble_results["3_Scenario_B_MAGA_Tariff_Panic_First"][0]
axE.bar(np.arange(N_STATES) - 0.2, df_scA["Sobol_S1_X0"] * 100.0, width=0.4, color="#1f77b4", label="Scenario A")
axE.bar(np.arange(N_STATES) + 0.2, df_scB["Sobol_S1_X0"] * 100.0, width=0.4, color="#d62728", label="Scenario B")
axE.axhline(0.10, color="red", ls="--", lw=1.5, label="0.1% Negligible Threshold")
axE.set_ylim(0, 0.20)
axE.set_title("(E) Initial Prior X_0 ~ U(0,0.2) Sobol Sensitivity S_1 (%)", fontweight="bold")
axE.set_xlabel("50 US States")
axE.set_ylabel("Variance Explained by X_0 (%)")
axE.grid(True, alpha=0.3)
axE.legend(fontsize=8)

# Panel F: Verification Table Across Scenarios A, B, C
axF = fig3.add_subplot(gs[1, 2])
axF.axis("off")
tbl_rows = [
    ["Metric", "2026 Flip", "Scen A", "Scen B", "Scen C1", "Scen C2"],
    ["Senate (Dem/Rep)",
     f"{df_global_summary.iloc[0]['Expected_Dem_Senate']:.0f}/{df_global_summary.iloc[0]['Expected_Rep_Senate']:.0f}",
     f"{df_global_summary.iloc[2]['Expected_Dem_Senate']:.0f}/{df_global_summary.iloc[2]['Expected_Rep_Senate']:.0f}",
     f"{df_global_summary.iloc[3]['Expected_Dem_Senate']:.0f}/{df_global_summary.iloc[3]['Expected_Rep_Senate']:.0f}",
     f"{df_global_summary.iloc[4]['Expected_Dem_Senate']:.0f}/{df_global_summary.iloc[4]['Expected_Rep_Senate']:.0f}",
     f"{df_global_summary.iloc[5]['Expected_Dem_Senate']:.0f}/{df_global_summary.iloc[5]['Expected_Rep_Senate']:.0f}"],
    ["House (Dem/Rep)",
     f"{df_global_summary.iloc[0]['Expected_Dem_House']:.0f}/{df_global_summary.iloc[0]['Expected_Rep_House']:.0f}",
     f"{df_global_summary.iloc[2]['Expected_Dem_House']:.0f}/{df_global_summary.iloc[2]['Expected_Rep_House']:.0f}",
     f"{df_global_summary.iloc[3]['Expected_Dem_House']:.0f}/{df_global_summary.iloc[3]['Expected_Rep_House']:.0f}",
     f"{df_global_summary.iloc[4]['Expected_Dem_House']:.0f}/{df_global_summary.iloc[4]['Expected_Rep_House']:.0f}",
     f"{df_global_summary.iloc[5]['Expected_Dem_House']:.0f}/{df_global_summary.iloc[5]['Expected_Rep_House']:.0f}"],
    ["Battleground D_est",
     f"{df_global_summary.iloc[0]['Battleground_D_est']:+.3f}",
     f"{df_global_summary.iloc[2]['Battleground_D_est']:+.3f}",
     f"{df_global_summary.iloc[3]['Battleground_D_est']:+.3f}",
     f"{df_global_summary.iloc[4]['Battleground_D_est']:+.3f}",
     f"{df_global_summary.iloc[5]['Battleground_D_est']:+.3f}"],
    ["Swing Turnout q*",
     f"{df_global_summary.iloc[0]['Battleground_Turnout_q_Pct']:.1f}%",
     f"{df_global_summary.iloc[2]['Battleground_Turnout_q_Pct']:.1f}%",
     f"{df_global_summary.iloc[3]['Battleground_Turnout_q_Pct']:.1f}%",
     f"{df_global_summary.iloc[4]['Battleground_Turnout_q_Pct']:.1f}%",
     f"{df_global_summary.iloc[5]['Battleground_Turnout_q_Pct']:.1f}%"],
    ["Swing Rep Share",
     f"{df_global_summary.iloc[0]['Battleground_Rep_Share_Pct']:.1f}%",
     f"{df_global_summary.iloc[2]['Battleground_Rep_Share_Pct']:.1f}%",
     f"{df_global_summary.iloc[3]['Battleground_Rep_Share_Pct']:.1f}%",
     f"{df_global_summary.iloc[4]['Battleground_Rep_Share_Pct']:.1f}%",
     f"{df_global_summary.iloc[5]['Battleground_Rep_Share_Pct']:.1f}%"],
    ["Expected Rep EV",
     f"{df_global_summary.iloc[0]['Expected_Rep_EV']:.1f}",
     f"{df_global_summary.iloc[2]['Expected_Rep_EV']:.1f}",
     f"{df_global_summary.iloc[3]['Expected_Rep_EV']:.1f}",
     f"{df_global_summary.iloc[4]['Expected_Rep_EV']:.1f}",
     f"{df_global_summary.iloc[5]['Expected_Rep_EV']:.1f}"],
    ["Rep Win Prob (%)",
     f"{df_global_summary.iloc[0]['Rep_Majority_Prob_Pct']:.1f}%",
     f"{df_global_summary.iloc[2]['Rep_Majority_Prob_Pct']:.1f}%",
     f"{df_global_summary.iloc[3]['Rep_Majority_Prob_Pct']:.1f}%",
     f"{df_global_summary.iloc[4]['Rep_Majority_Prob_Pct']:.1f}%",
     f"{df_global_summary.iloc[5]['Rep_Majority_Prob_Pct']:.1f}%"],
    ["Mean Sobol S1",
     f"{df_global_summary.iloc[0]['Mean_Sobol_S1']:.1e}",
     f"{df_global_summary.iloc[2]['Mean_Sobol_S1']:.1e}",
     f"{df_global_summary.iloc[3]['Mean_Sobol_S1']:.1e}",
     f"{df_global_summary.iloc[4]['Mean_Sobol_S1']:.1e}",
     f"{df_global_summary.iloc[5]['Mean_Sobol_S1']:.1e}"],
    ["Max R-hat (<1.01)",
     f"{df_global_summary.iloc[0]['Max_Gelman_Rubin_Rhat']:.5f}",
     f"{df_global_summary.iloc[2]['Max_Gelman_Rubin_Rhat']:.5f}",
     f"{df_global_summary.iloc[3]['Max_Gelman_Rubin_Rhat']:.5f}",
     f"{df_global_summary.iloc[4]['Max_Gelman_Rubin_Rhat']:.5f}",
     f"{df_global_summary.iloc[5]['Max_Gelman_Rubin_Rhat']:.5f}"],
]
tbl = axF.table(cellText=tbl_rows, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.0)
tbl.scale(1.05, 1.52)
for (ri, ci), cell in tbl.get_celld().items():
    if ri == 0:
        cell.set_facecolor("#2b5c8f")
        cell.set_text_props(color="white", fontweight="bold")
    elif ri in [3, 5, 6, 7]:
        cell.set_facecolor("#f4f7fb")
axF.set_title("(F) Scenarios A, B, C Mathematical & Calibration Summary", fontweight="bold")

fig3.suptitle(
    "Master Verification Dashboard: MAGA Policy Lock-in vs. Post-Midterm Asynchronous Panic Cascades (Scenarios A, B, C)",
    fontsize=13, fontweight="bold", y=0.98
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig3.savefig(f"{OUTPUT_DIR}/Fig3_Master_Dashboard_Scenarios_ABC.png", dpi=180)
plt.close(fig3)

# ------------------------------------------------------------------------------
# 8. 全結果のZIP圧縮とGoogle Colab自動ダウンロード
# ------------------------------------------------------------------------------
zip_path = "Blume_Capel_Scenarios_ABC_MAGA_Panic_Verification.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            full_p = os.path.join(root, f)
            zf.write(full_p, arcname=os.path.join(OUTPUT_DIR, f))

print("\n==========================================================================")
print(f" [COMPLETE] ZIP Archive Created: {zip_path}")
print("==========================================================================\n")
print(df_global_summary[[
    "Regime", "Expected_Dem_Senate", "Expected_Dem_House",
    "Battleground_D_est", "Battleground_Turnout_q_Pct",
    "Battleground_Rep_Share_Pct", "Expected_Rep_EV", "Rep_Majority_Prob_Pct"
]].to_string(index=False))

try:
    from google.colab import files
    files.download(zip_path)
except ImportError:
    print(f"\n[INFO] Local execution detected. Archive saved at: ./{zip_path}")
