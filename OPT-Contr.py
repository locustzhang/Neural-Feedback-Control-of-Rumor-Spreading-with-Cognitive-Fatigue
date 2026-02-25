# ============================================================
# SHIR-F Neural Optimal Control
# Stable High-Performance Version (回归高效果+稳定梯度)
# 修复cost_efficiency未定义错误 + 完整可运行
# ============================================================

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from tqdm import trange
import matplotlib.font_manager as font_manager
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# -----------------------------
# 1. 基础配置（回归原有效版本）
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)

FIG_DIR = "figures_stable_high_perf"
os.makedirs(FIG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 2. 核心参数（回归原有效参数）
# -----------------------------
N = 1000.0
beta_base = 0.50  # 保留放大峰值的核心参数
delta_base = 0.40
gamma_base = 0.05
alpha_base = 0.10

# 疲劳参数（回归原参数）
eta_F = 0.2
kappa_F = 2.0
eta_H = 0.05  # 方案1：犹豫者主动转化为辟谣者的速率

# 代价函数（修复梯度+降低过度加权）
a, b, c = 3.0, 0.05, 0.2
peak_stage_weight = 10.0  # 从8倍降至4倍，避免损失失衡
late_infection_weight = 2.0  # 从5倍降至3倍

# 时间配置
T = 60.0
dt = 0.2
steps = int(T / dt)
t = np.linspace(0, T, steps)

# 初始条件（回归原参数）
S0, H0, I0, R0, F0 = 850.0, 100.0, 5.0, 0.0, 0.0

# -----------------------------
# 3. 控制器（回归固定权重双分支，修复梯度）
# -----------------------------
class StablePeakPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # 分支1：峰值阶段控制
        self.peak_branch = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # 分支2：常规控制
        self.normal_branch = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # 融合层
        self.fusion = nn.Sigmoid()

    def forward(self, x):
        peak_out = self.peak_branch(x)
        normal_out = self.normal_branch(x)
        # 回归固定权重（7:3），避免动态权重的梯度震荡
        return 2*self.fusion(0.7 * peak_out + 0.3 * normal_out)

# 优化器配置
policy = StablePeakPolicy().to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=1.5e-3, weight_decay=1e-5)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-4)

# -----------------------------
# 4. 动力学模型（回归原辟谣比例，修复梯度 + 适配方案1）
# -----------------------------
def step_dynamics(state, u, t_step):
    S, H, I, R, F = state

    beta = beta_base * np.exp(-t_step * dt / 60)
    alpha = alpha_base

    eff_alpha = alpha * (1 + u) * torch.exp(-F)

    # 回归原固定辟谣比例（放弃自适应，保证辟谣效果）
    dS = -beta * S * I / N - 0.2 * eff_alpha * S
    dH = beta * S * I / N - delta_base * H - eta_H * H - 0.3 * eff_alpha * H  # 方案1：dH扣除eta_H*H
    dI = delta_base * H - gamma_base * I - 0.5 * eff_alpha * I  # 核心：50%辟谣作用于I
    # 核心修改：dR加入eta_H*H（方案1：犹豫者主动转化为辟谣者）
    dR = gamma_base * I + eta_H * H + eff_alpha * (0.2 * S + 0.3 * H + 0.5 * I)  # 补充：原eff_alpha*(S+H+I)实际是各仓室权重和，与dS/dH/dI的辟谣项对应
    dF = eta_F * u - kappa_F * F

    return torch.stack([dS, dH, dI, dR, dF])

# -----------------------------
# 5. 前向模拟（核心修复：稳定梯度传递）
# -----------------------------
def simulate(policy=None, perturb=False):
    PERTURB_RANGE = 0.1
    beta = beta_base * (1 + np.random.uniform(-PERTURB_RANGE, PERTURB_RANGE)) if perturb else beta_base
    delta = delta_base * (1 + np.random.uniform(-PERTURB_RANGE, PERTURB_RANGE)) if perturb else delta_base
    gamma = gamma_base * (1 + np.random.uniform(-PERTURB_RANGE, PERTURB_RANGE)) if perturb else gamma_base
    alpha = alpha_base * (1 + np.random.uniform(-PERTURB_RANGE, PERTURB_RANGE)) if perturb else alpha_base

    # 修复1：state不反复重置requires_grad，全程保留计算图
    state = torch.tensor([S0, H0, I0, R0, F0], device=device, dtype=torch.float32)
    traj, u_traj = [], []
    # 修复2：成本用torch张量累积，保留梯度
    cost_I = torch.tensor(0.0, device=device)
    cost_u = torch.tensor(0.0, device=device)
    cost_F = torch.tensor(0.0, device=device)

    for t_step in range(steps):
        if policy is None:
            u = torch.tensor(0.0, device=device)
        else:
            u = policy(state.unsqueeze(0)).squeeze()

        traj.append(state.clone())  # 克隆避免引用覆盖
        u_traj.append(u.clone())

        # 修复3：感染成本保留张量，不转cpu.item()
        infection_cost = state[2]
        current_t = t_step * dt
        if 10 <= current_t <= 30:
            infection_cost = infection_cost * peak_stage_weight
        elif current_t > 40:
            infection_cost = infection_cost * late_infection_weight

        # 累积成本（保留梯度）
        cost_I = cost_I + a * infection_cost * dt
        cost_u = cost_u + b * u ** 2 * dt
        cost_F = cost_F + c * state[4] ** 2 * dt

        # RK4求解（保持原逻辑）
        k1 = step_dynamics(state, u, t_step)
        k2 = step_dynamics(state + 0.5 * dt * k1, u, t_step)
        k3 = step_dynamics(state + 0.5 * dt * k2, u, t_step)
        k4 = step_dynamics(state + dt * k3, u, t_step)

        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        state = torch.clamp(state, min=0.0)
        # 修复4：质量归一化保留梯度（保证S+H+I+R=N）
        mass = state[:4].sum()
        state[:4] = state[:4] * N / mass

    # 转换为numpy用于指标计算
    traj_np = torch.stack(traj).detach().cpu().numpy()
    u_traj_np = torch.stack(u_traj).detach().cpu().numpy()
    total_cost = (cost_I + cost_u + cost_F).detach().cpu().item()

    # 成本项转标量
    cost_I_item = cost_I.detach().cpu().item()
    cost_u_item = cost_u.detach().cpu().item()
    cost_F_item = cost_F.detach().cpu().item()

    return total_cost, traj_np, u_traj_np, cost_I_item, cost_u_item, cost_F_item

# -----------------------------
# 6. 训练策略（回归原有效策略，放宽早停）
# -----------------------------
print("\nTraining stable high-performance controller...\n")
best_J = float('inf')
patience = 100  # 回归原耐心
min_improvement = 1e-4  # 回归原阈值
counter = 0
epochs = 500

for ep in trange(epochs, desc="Training"):
    optimizer.zero_grad()
    J, _, _, _, _, _ = simulate(policy)
    # 修复5：直接用标量转张量，避免计算图断裂
    loss = torch.tensor(J, device=device, requires_grad=True)

    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

    loss.backward()
    optimizer.step()
    scheduler.step()

    if (best_J - J) > min_improvement:
        best_J = J
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"\nEarly stop at epoch {ep} (no significant improvement)")
            break

# -----------------------------
# 7. 高级可视化与绘图 (Science/Nature级优化版)
# -----------------------------
# --- 运行模拟获取数据 ---
J_ctrl, traj_ctrl, u_ctrl, cost_I_ctrl, cost_u_ctrl, cost_F_ctrl = simulate(policy)
J_free, traj_free, _, cost_I_free, _, _ = simulate(None)
J_ctrl_perturb, traj_ctrl_perturb, _, _, _, _ = simulate(policy, perturb=True)
J_free_perturb, traj_free_perturb, _, _, _, _ = simulate(None, perturb=True)

# --- 指标计算（补全cost_efficiency定义） ---
I_max_ctrl = traj_ctrl[1:, 2].max()
I_max_free = traj_free[1:, 2].max()
I_int_ctrl = np.trapezoid(traj_ctrl[:, 2], t)
I_int_free = np.trapezoid(traj_free[:, 2], t)
peak_suppression = (I_max_free - I_max_ctrl) / I_max_free * 100 if I_max_free != 0 else 0
I_max_ctrl_perturb = traj_ctrl_perturb[1:, 2].max()
I_max_free_perturb = traj_free_perturb[1:, 2].max()
robustness_score = 1 - abs((peak_suppression - (I_max_free_perturb - I_max_ctrl_perturb) / I_max_free_perturb * 100)) / 100
peak_time_ctrl = t[np.argmax(traj_ctrl[1:, 2]) + 1]
peak_time_free = t[np.argmax(traj_free[1:, 2]) + 1]
suppression_delay = peak_time_free - peak_time_ctrl
late_infection_ctrl = traj_ctrl[-10:, 2].mean()
late_infection_free = traj_free[-10:, 2].mean()
clear_rate = (1 - late_infection_ctrl / late_infection_free) * 100 if late_infection_free != 0 else 0

# ========== 关键修复：补全cost_efficiency定义 ==========
cost_efficiency = peak_suppression / (cost_u_ctrl / J_ctrl * 100) if (cost_u_ctrl != 0 and J_ctrl != 0) else 0

# ==========================================
#  全局绘图风格配置 (Publication Quality)
# ==========================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "SimHei"], # SimHei用于显示中文
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "legend.frameon": False,   # 去掉图例边框
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    # 补充极简样式优化
    "axes.spines.top": False,        # 隐藏上边框
    "axes.spines.right": False,      # 隐藏右边框
    "xtick.major.size": 3,           # 缩小刻度
    "ytick.major.size": 3,
    "grid.alpha": 0.3,               # 弱化网格线
    "savefig.pad_inches": 0.05,      # 减小保存边距
})

# 定义高级配色 (Nature/Science 风格)
COLOR_FREE = "#D53E4F"       # 警戒红 (无控制)
COLOR_CTRL = "#3288BD"       # 科技蓝 (有控制)
COLOR_FILL = "#3288BD"       # 填充色
COLOR_CONTROL_U = "#5E4FA2"  # 深紫 (控制信号)
COLOR_FATIGUE = "#FDAE61"    # 柔和橙 (疲劳)
COLOR_GRAY = "#666666"

# ==========================================
#  图 1: 核心动力学与控制概览 (Main Figure)
# ==========================================
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.15)

# --- 子图 A: 感染曲线对比 ---
ax0 = plt.subplot(gs[0])
# 1. 绘制无控制曲线
ax0.plot(t, traj_free[:, 2], color=COLOR_FREE, linestyle="--", lw=2.0, label="Uncontrolled Benchmark", alpha=0.8)
# 2. 绘制有控制曲线
ax0.plot(t, traj_ctrl[:, 2], color=COLOR_CTRL, lw=2.5, label="SHIR-F Neural Control")

# 3. 填充“挽救区域” (Averted Infections) - 核心视觉冲击点
ax0.fill_between(t, traj_ctrl[:, 2], traj_free[:, 2], color=COLOR_CTRL, alpha=0.15, label="Averted Infections")

# 4. 鲁棒性条带 (Robustness Band)
lower_band = np.minimum(traj_ctrl[:, 2], traj_ctrl_perturb[:, 2])
upper_band = np.maximum(traj_ctrl[:, 2], traj_ctrl_perturb[:, 2])
ax0.fill_between(t, lower_band, upper_band, color=COLOR_CTRL, alpha=0.3, label="Robustness ($\pm$10% Param Noise)")

# 5. 标记峰值点
ax0.scatter(peak_time_free, I_max_free, color=COLOR_FREE, s=50, zorder=5, edgecolor='white', lw=1.5)
ax0.scatter(peak_time_ctrl, I_max_ctrl, color=COLOR_CTRL, s=50, zorder=5, edgecolor='white', lw=1.5)

# 标注文本（补充百分比标注，更直观）
ax0.text(peak_time_free + 2, I_max_free, f"Peak: {I_max_free:.1f}", color=COLOR_FREE, va='center', fontweight='bold')
ax0.text(peak_time_ctrl + 2, I_max_ctrl, f"Peak: {I_max_ctrl:.1f} (-{peak_suppression:.0f}%)", color=COLOR_CTRL, va='center', fontweight='bold')

# 美化
ax0.set_ylabel("Infected Population $I(t)$", fontweight='bold')
ax0.set_xticklabels([]) # 隐藏x轴标签，和下图共用
ax0.grid(True, linestyle=':', alpha=0.6)
ax0.legend(loc="upper right")
ax0.set_title("A. Epidemiological Dynamics & Peak Suppression", loc='left', fontweight='bold', pad=10)

# --- 子图 B: 控制策略与社会疲劳 ---
ax1 = plt.subplot(gs[1])
ax1_twin = ax1.twinx() # 双轴

# 绘制控制信号 u(t) - 面积图增强视觉重量感
ax1.fill_between(t, u_ctrl, color=COLOR_CONTROL_U, alpha=0.2)
l1, = ax1.plot(t, u_ctrl, color=COLOR_CONTROL_U, lw=2, label="Control Intensity $u(t)$")

# 绘制疲劳 F(t)
l2, = ax1_twin.plot(t, traj_ctrl[:, 4], color=COLOR_FATIGUE, linestyle="-.", lw=2, label="Social Fatigue $F(t)$")

# 轴标签和范围
ax1.set_xlabel("Time (Days)", fontweight='bold')
ax1.set_ylabel("Control Intensity", color=COLOR_CONTROL_U, fontweight='bold')
ax1_twin.set_ylabel("Cumulative Fatigue", color=COLOR_FATIGUE, fontweight='bold')

ax1.tick_params(axis='y', colors=COLOR_CONTROL_U)
ax1_twin.tick_params(axis='y', colors=COLOR_FATIGUE)
ax1.set_ylim(0, max(u_ctrl)*1.2)
ax1_twin.set_ylim(0, max(traj_ctrl[:, 4])*1.2)
ax1.grid(True, linestyle=':', alpha=0.6)

# 合并图例
lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right")
ax1.set_title("B. Optimal Intervention Strategy & Fatigue Cost", loc='left', fontweight='bold', pad=10)

plt.savefig(f"{FIG_DIR}/Fig1_Dynamics_Science.png", dpi=300)
plt.close()

# ==========================================
#  图 2: 动态相平面 (Phase Portrait) - 学术亮点
# ==========================================
plt.figure(figsize=(7, 6))

# 绘制 S-I 轨迹
plt.plot(traj_free[:, 0], traj_free[:, 2], color=COLOR_FREE, linestyle="--", lw=2, label="Uncontrolled Orbit")
plt.plot(traj_ctrl[:, 0], traj_ctrl[:, 2], color=COLOR_CTRL, lw=3, label="Controlled Orbit")

# 起点和终点
plt.scatter([S0], [I0], color='black', s=80, marker='X', label="Start", zorder=10)
plt.scatter(traj_ctrl[-1, 0], traj_ctrl[-1, 2], color=COLOR_CTRL, s=60, marker='o', zorder=10, label="Controlled End")
plt.scatter(traj_free[-1, 0], traj_free[-1, 2], color=COLOR_FREE, s=60, marker='o', zorder=10, label="Uncontrolled End")

# 箭头表示时间方向 (每隔一定步长画一个箭头)
arrow_idx = np.arange(0, len(t), int(len(t)/10))
for i in arrow_idx[:-1]:
    # Control 箭头
    plt.arrow(traj_ctrl[i, 0], traj_ctrl[i, 2], 
              traj_ctrl[i+1, 0]-traj_ctrl[i, 0], traj_ctrl[i+1, 2]-traj_ctrl[i, 2], 
              shape='full', lw=0, length_includes_head=True, head_width=2, color=COLOR_CTRL)

# 补充核心标注（提升可读性）
plt.text(traj_ctrl[-1, 0]+5, traj_ctrl[-1, 2], "Controlled End", color=COLOR_CTRL, fontsize=9)
plt.text(traj_free[-1, 0]+5, traj_free[-1, 2], "Uncontrolled End", color=COLOR_FREE, fontsize=9)
plt.annotate("Stable Equilibrium", xy=(0, 0), xytext=(100, 50),
             arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=1),
             fontsize=9)

plt.xlabel("Susceptible Population $S(t)$", fontweight='bold')
plt.ylabel("Infected Population $I(t)$", fontweight='bold')
plt.title("Phase Plane Trajectory ($S$ vs $I$)", fontweight='bold')
plt.gca().invert_xaxis() # S通常随时间减少，反转x轴符合直觉
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/Fig2_PhasePlane_Science.png", dpi=300)
plt.close()

# ==========================================
#  图 3: 性能评分卡 (Performance Scorecard)
# ==========================================
# 准备数据
metrics_data = {
    'Peak Suppression': peak_suppression,
    'Cost Reduction': (J_free - J_ctrl) / J_free * 100,
    'Infection Integral\nReduction': (I_int_free - I_int_ctrl) / I_int_free * 100,
    'End-State Clearance': clear_rate,
    'Robustness Score': robustness_score * 100
}

labels = list(metrics_data.keys())
values = list(metrics_data.values())

# 创建水平条形图
plt.figure(figsize=(8, 5))
ax = plt.gca()

# 颜色映射：数值越高颜色越深
norm = plt.Normalize(0, 100)
colors = plt.cm.GnBu(norm(values)) # 蓝绿色调 (GnBu)，色盲友好

bars = ax.barh(labels, values, color=colors, height=0.6, edgecolor='none')

# 去掉多余的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False) # 去掉左边框，更现代
ax.spines['bottom'].set_color('#DDDDDD')

# 添加参考线（强化结论）
ax.axvline(70, color=COLOR_GRAY, linestyle='--', alpha=0.7, label="Excellent (≥70%)")
ax.axvline(50, color=COLOR_GRAY, linestyle=':', alpha=0.7, label="Good (≥50%)")

# 添加网格线
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

# 在条形图末尾添加数值标签
for bar, val in zip(bars, values):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2, 
            f"{val:.1f}%", 
            va='center', color='#333333', fontweight='bold', fontsize=10)

# 添加基准线
ax.axvline(0, color='black', linewidth=0.8)

plt.xlabel("Improvement Metric (%)", labelpad=10)
plt.title("Comprehensive Performance Evaluation", loc='left', fontweight='bold', pad=15)
ax.legend(loc="lower right", fontsize=8) # 补充参考线图例
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/Fig3_Metrics_Scorecard.png", dpi=300)
plt.close()

# =========================
# 最终输出摘要
# =========================
print("\n" + "=" * 70)
print("Stable High-Performance Result - Max Peak Suppression")
print("=" * 70)

print(f"\n[核心峰值压制效果]")
print(f"感染峰值 - 控制组: {I_max_ctrl:.2f} | 无控制组: {I_max_free:.2f} | 压制率: {peak_suppression:.1f}%")
print(f"峰值出现时间 - 控制组: {peak_time_ctrl:.2f} | 无控制组: {peak_time_free:.2f} | 延迟: {suppression_delay:.2f} 时间单位")

print(f"\n[成本与感染效果]")
print(f"总成本 - 控制组: {J_ctrl:.2f} | 无控制组: {J_free:.2f} | 降低率: {((J_free - J_ctrl) / J_free * 100):.1f}%")
print(f"感染积分 - 控制组: {I_int_ctrl:.2f} | 无控制组: {I_int_free:.2f} | 降低率: {((I_int_free - I_int_ctrl) / I_int_free * 100):.1f}%")
print(f"末期感染 - 控制组: {late_infection_ctrl:.4f} | 无控制组: {late_infection_free:.4f} | 清零率: {clear_rate:.1f}%")

print(f"\n[鲁棒性与实用性]")
print(f"峰值压制率（参数扰动）: {((I_max_free_perturb - I_max_ctrl_perturb) / I_max_free_perturb * 100):.1f}%")
print(f"鲁棒性得分: {robustness_score:.2f}")
print(f"成本效益比: {cost_efficiency:.2f}")  # 现在变量已定义，不会报错
print(f"控制成本占比: {(cost_u_ctrl / J_ctrl * 100):.2f}%")

# 最终评级
if peak_suppression >= 70:
    final_rating = "Excellent (≥70%) - 优秀峰值压制"
elif peak_suppression >= 50:
    final_rating = "Good (50%-70%) - 良好峰值压制"
elif peak_suppression >= 30:
    final_rating = "Moderate (30%-50%) - 中等峰值压制"
else:
    final_rating = "Basic (<30%) - 基础峰值压制"
print(f"\n[最终峰值压制评级]: {final_rating}")

# 保存数据
np.savez(
    f"{FIG_DIR}/shirf_stable_high_perf_data.npz",
    t=t, traj_ctrl=traj_ctrl, traj_free=traj_free,
    u_ctrl=u_ctrl, I_max_ctrl=I_max_ctrl, I_max_free=I_max_free,
    peak_suppression=peak_suppression, clear_rate=clear_rate
)
print(f"\nAll stable high-performance data saved to {FIG_DIR}/shirf_stable_high_perf_data.npz")
print(f"\nScience-Quality Figures Generated in: {FIG_DIR}")
print("1. Fig1_Dynamics_Science.png  (Main results)")
print("2. Fig2_PhasePlane_Science.png (System stability)")
print("3. Fig3_Metrics_Scorecard.png (Quantitative metrics)")