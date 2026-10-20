# ==============================================================================
# 3状態Blume-Capel選挙ポテンシャルゲーム 大規模モンテカルロ実証スイート (1M - 10M Trials)
# - 2024.12〜2026.10 Gallup & 全国世論調査実数キャリブレーション + 50万人デモショック
# - 検証テーマ: エスタブリッシュメント層(c=est)と真実追求層(c=truth)の「制度的パニックと棄権雪崩」
#   および「頑迷な党派(Stubborn Partisans)による法案改変・差し戻しループ(Legislative Ping-Pong)」
# ==============================================================================

import os
import time
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

# 論文準拠の擬似乱数シード固定 (完全な数値的再現性の保証)
RANDOM_SEED = 20261103
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = "blume_capel_institutional_panic_gridlock_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 0. 大規模計算パラメータ設定 (1,000,000 〜 10,000,000 試行)
# ------------------------------------------------------------------------------
N_TRIALS = 1_000_000        # 10,000,000 に変更することで1000万回試行を実行可能
BATCH_SIZE = 200_000        # 省メモリ・高速ベクトル化演算のためのバッチサイズ
N_CHAINS = 4                # Gelman-Rubin R-hat 収束診断用の独立マルコフ連鎖数

print("==========================================================================")
print(" 3-State Blume-Capel Electoral Potential Game: Institutional Panic &")
print(" Legislative Remand Loop (Nothing Moves Gridlock) Verification Suite")
print(f" Total Monte Carlo Trials per Regime : {N_TRIALS:,} (Batch: {BATCH_SIZE:,})")
print(" Initial Condition Prior             : X_{i,0} ~ Uniform U(0, 0.2)")
print("==========================================================================")

# ------------------------------------------------------------------------------
# 1. 2024年12月〜2026年10月 Gallup & 全国世論調査の実数データと磁場較正
# ------------------------------------------------------------------------------
polling_records = [
    # YearMonth, Approve(%), Disapprove(%), Ind_Approve(%), Cong_Approve(%), Protest_Scale_Index, Note
    ("2024-12", 49.0, 46.0, 44.0, 19.0, 0.05, "Presidential Transition Avg (Dec 2024)"),
    ("2025-01", 47.0, 48.0, 46.0, 21.0, 0.10, "Gallup Inauguration Baseline (Jan 21-27, 2025: Net -1%)"),
    ("2025-02", 45.0, 51.0, 41.0, 22.0, 0.15, "Gallup Monthly (Feb 2025)"),
    ("2025-03", 44.5, 52.0, 39.0, 21.0, 0.22, "Gallup / Aggregate (Mar 2025: Tariff announcement)"),
    ("2025-04", 44.0, 53.0, 37.0, 20.0, 0.28, "Gallup (Apr 1-14, 2025: Q1 GDP -0.5% contraction)"),
    ("2025-05", 43.0, 53.0, 36.0, 19.0, 0.32, "Gallup (May 1-18, 2025)"),
    ("2025-06", 40.0, 57.0, 33.0, 18.0, 0.55, "Gallup (Jun 3-19, 2025: First No Kings protests)"),
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

# 実数支持率の下落と50万人反トランプデモ動員から負の外部磁場ショック -Δh_anti を算出
df_polls["Delta_h_Anti_Shock"] = (
    0.0042 * (df_polls["Net_Approval_Pct"] - base_net_jan25)
    - 0.048 * df_polls["Protest_Scale_Index"]
)
# 議会不信スコア (100 - Cong_Approve) から「立法停滞・差し戻しループへの基礎感受性」を算出
df_polls["Legislative_Distrust_Index"] = (100.0 - df_polls["Cong_Approve_Pct"]) / 100.0
df_polls.to_csv(f"{OUTPUT_DIR}/01_empirical_polling_gallup_2024_2026.csv", index=False)

DELTA_H_OCT2026 = float(df_polls.iloc[-1]["Delta_h_Anti_Shock"])          # -0.1656
CONG_DISTRUST_OCT2026 = float(df_polls.iloc[-1]["Legislative_Distrust_Index"]) # 0.870

# ------------------------------------------------------------------------------
# 2. 米国全50州ベースライン構造パラメータ (論文 Table 6 準拠)
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
# 3. 厳密6次Landau展開と「法案差し戻しループによる結晶場侵食」の解析的評価
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
    ("0_Phase1_2026_Midterm_Dem_Flip (D > D_t)",         1.05, -0.055, +0.10, 0.990),
    ("1_Cell2_Single_Shock_Establishment_Active",        1.00, +0.020, +0.12, 0.960),
    ("2_Tricritical_Boundary_Onset (a2=a4=0)",           1.15, +0.045,  0.00, 1.000),
    ("3_Moderate_Legislative_Amendment_Delay",           1.15, +0.060, -0.05, 1.008),
    ("4_Severe_Remand_Loop_Gridlock (Nothing_Moves)",    1.25, +0.082, -0.11, 1.015),
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
df_landau.to_csv(f"{OUTPUT_DIR}/02_tricritical_landau_gridlock_coefficients.csv", index=False)

# ------------------------------------------------------------------------------
# 4. 「法案の改変・差し戻しループ (Ping-Pong Loop)」時間発展シミュレーション
# ------------------------------------------------------------------------------
def simulate_legislative_pingpong_loop_dynamics(T_max=100):
    """
    2026年中間選挙で民主党が上下両院を奪還した後、
    頑迷な党派(Stubborn Zealots)による法案改変・差し戻し・大統領拒否権の無限ループにより
    「何も進まない(Nothing Moves)」状態に陥り、エスタブリッシュメント層(c=est)と
    真実追求層(c=truth)の結晶場 D_c(t) が三重臨界点 D_t = -0.4621 を割ってスピノーダル崩壊する過程を追跡。
    """
    t_steps = np.arange(T_max)
    records = []

    loop_scenarios = {
        "Regime_1_Cell2_No_Gridlock (Single Scandal Only)": {
            "remand_amp": 0.05, "maga_friction": 0.10, "veto_loop_freq": 0.0, "mem_ret": 0.30
        },
        "Regime_2_Moderate_Amendment_Friction (Partial Compromise)": {
            "remand_amp": 0.42, "maga_friction": 0.48, "veto_loop_freq": 0.18, "mem_ret": 0.75
        },
        "Regime_3_Stubborn_Partisan_Remand_Loop (Nothing Moves Gridlock)": {
            "remand_amp": 0.95, "maga_friction": 0.92, "veto_loop_freq": 0.35, "mem_ret": 0.95
        }
    }

    for scen_name, cfg in loop_scenarios.items():
        # t=0: 2026年中間選挙で民主党が上下両院をひっくり返した直後の均衡 (m < 0, 民主党優位)
        m_w, m_e, m_t = +0.36, -0.58, -0.46
        q_w, q_e, q_t =  0.84,  0.91,  0.86
        mem_est, mem_truth = 0.0, 0.0
        cum_deadlock = 0.0
        beta_d = 2.45

        for t in t_steps:
            # 法案差し戻し(Remand / Ping-Pong)サイクルの振動と累積停滞度 L_deadlock(t)
            if t >= 10:
                dt = t - 10.0
                # 委員会修正 -> 本会議 -> 上下両院差し戻し -> 大統領拒否権の周期的ループ
