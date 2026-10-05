import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. 고해상도 폰트 및 스타일 세팅
# -------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.25, 1]})

# -------------------------------------------------------------------------
# Panel (A): Wu & Huang (2004) Energy-Period Significance Test Plane
# -------------------------------------------------------------------------
N = 512
ln_T = np.linspace(0.6, 5.2, 200)
T_vals = np.exp(ln_T)

# IMF 1 앵커링 오프셋 C 적용
C = -1.5
ln_E_mean = -ln_T + C
upper_bound_95 = ln_E_mean + 1.645 * np.sqrt(2 * T_vals / N)
lower_bound_95 = ln_E_mean - 1.645 * np.sqrt(2 * T_vals / N)

# 기준선 및 95% 신뢰대역 플롯 (Raw String r'...' 적용)
ax1.plot(ln_T, ln_E_mean, color='#555555', linestyle=':', lw=2, 
         label=r'Theoretical White Noise Mean ($\ln \bar{E} = -\ln \bar{T} + C$)')
ax1.plot(ln_T, upper_bound_95, color='#d63031', linestyle='--', lw=2.5, 
         label=r'95% Upper Bound ($\ln \bar{E} = -\ln \bar{T} + C + 1.645\sqrt{2\bar{T}/N}$)')
ax1.plot(ln_T, lower_bound_95, color='#74b9ff', linestyle=':', lw=1.5, alpha=0.7)
ax1.fill_between(ln_T, lower_bound_95, upper_bound_95, color='#dfe6e9', alpha=0.45, 
                 label='95% White Noise Confidence Band')

# IMF 샘플 데이터 매핑
imfs = [
    {'name': 'IMF 1', 'ln_T': 1.0, 'ln_E': -2.5, 'type': 'noise', 'desc': 'Anchored Noise Floor'},
    {'name': 'IMF 2', 'ln_T': 1.8, 'ln_E': -3.1, 'type': 'noise', 'desc': 'Stochastic Dyadic Mode'},
    {'name': 'IMF 3', 'ln_T': 2.7, 'ln_E': -2.0, 'type': 'signal', 'desc': 'Transient Shock Peak'},
    {'name': 'IMF 4', 'ln_T': 3.7, 'ln_E': -1.4, 'type': 'signal', 'desc': 'Intervention Dynamic Mode'},
]

for pt in imfs:
    if pt['type'] == 'noise':
        ax1.scatter(pt['ln_T'], pt['ln_E'], color='#0984e3', s=160, zorder=6, edgecolors='black', lw=1.5)
        # ★ \mathrm 제거로 EMD 이탤릭체 적용 및 수식 밖 일반 공백 분리
        ax1.annotate(f"{pt['name']}\n" + r"$\rightarrow$ Included in $R^{EMD}$",
                     xy=(pt['ln_T'], pt['ln_E']), xytext=(pt['ln_T'] - 0.2, pt['ln_E'] - 1.05),
                     fontsize=9.5, fontweight='bold', color='#0984e3',
                     arrowprops=dict(arrowstyle="->", color='#0984e3', lw=1.5))
    else:
        ax1.scatter(pt['ln_T'], pt['ln_E'], color='#d63031', marker='^', s=200, zorder=6, edgecolors='black', lw=1.5)
        # ★ 수식 밖 일반 공백 분리
        ax1.annotate(f"{pt['name']}\n" + r"$\rightarrow$ Included in $I_t$",
                     xy=(pt['ln_T'], pt['ln_E']), xytext=(pt['ln_T'] - 0.1, pt['ln_E'] + 0.75),
                     fontsize=9.5, fontweight='bold', color='#d63031',
                     arrowprops=dict(arrowstyle="->", color='#d63031', lw=1.5))

# bbox 딕셔너리
bbox_signal = dict(boxstyle='round,pad=0.5', facecolor='#fadbd8', edgecolor='#e74c3c', alpha=0.9)
bbox_noise  = dict(boxstyle='round,pad=0.5', facecolor='#d4e6f1', edgecolor='#3498db', alpha=0.9)

ax1.text(1.1, 0.3, "Signal Region ($I_t$)\n[White Noise Hypothesis Rejected]",
         fontsize=10.5, fontweight='bold', color='#c0392b', bbox=bbox_signal)

# ★ 하단 텍스트 박스도 동일하게 $R^{EMD}$로 이탤릭체 통일
ax1.text(3.1, -5.3, "Noise Region ($R^{EMD}$)\n[White Noise Hypothesis Accepted]",
         fontsize=10.5, fontweight='bold', color='#2980b9', bbox=bbox_noise)

ax1.set_xlim(0.5, 5.0)
ax1.set_ylim(-6.5, 1.5)
ax1.set_xlabel(r"Logarithm of Mean Period: $\ln(\bar{T}_k)$", fontsize=11.5, fontweight='bold')
ax1.set_ylabel(r"Logarithm of Energy Density: $\ln(\bar{E}_k)$", fontsize=11.5, fontweight='bold')
ax1.set_title("Statistical Significance Test proposed by Wu & Huang", fontsize=12.5, fontweight='bold', pad=12)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', fontsize=9.5, framealpha=0.95)

# -------------------------------------------------------------------------
# Panel (B): Hybrid STL-EMD Dynamic Mode Allocation Architecture
# -------------------------------------------------------------------------
ax2.axis('off')
ax2.set_title("(B) Hybrid STL-EMD Dynamic Allocation Architecture", fontsize=12.5, fontweight='bold', pad=12)

# 아키텍처 박스 서식 사전 정의
box_stl   = dict(boxstyle='round,pad=0.5', facecolor='#dfe6e9', edgecolor='#2d3436', lw=1.6)
box_test  = dict(boxstyle='round,pad=0.5', facecolor='#ffeaa7', edgecolor='#d35400', lw=2)
box_sig   = dict(boxstyle='round,pad=0.5', facecolor='#fadbd8', edgecolor='#d63031', lw=1.8)
box_noise = dict(boxstyle='round,pad=0.5', facecolor='#d4e6f1', edgecolor='#0984e3', lw=1.8)
box_out   = dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#2d3436', lw=2)
box_final = dict(boxstyle='round,pad=0.5', facecolor='#f1f2f6', edgecolor='#2ed573', lw=2)

ax2.text(0.5, 0.94, "Contaminated Residual $R_{\mathrm{stl}}(t)$\nfrom 1st-stage STL ($y = T + S + R_{\mathrm{stl}}$)", 
         ha='center', va='center', fontsize=10, fontweight='bold', bbox=box_stl)

ax2.annotate('', xy=(0.5, 0.85), xytext=(0.5, 0.89), arrowprops=dict(arrowstyle="->", lw=1.8, color='#2d3436'))

ax2.text(0.5, 0.80, r"EMD (Empirical Mode Decomposition)" + "\n" + r"$\rightarrow$ Extract IMF$_1, \dots,$ IMF$_K$ + Residue $r_n(t)$", 
         ha='center', va='center', fontsize=9.5, fontweight='bold', bbox=box_stl)

ax2.annotate('', xy=(0.5, 0.70), xytext=(0.5, 0.75), arrowprops=dict(arrowstyle="->", lw=1.8, color='#2d3436'))

ax2.text(0.5, 0.65, r"Wu & Huang (2004) Statistical Test" + "\n" + r"$\ln \bar{E}_k > -\ln \bar{T}_k + C + 1.645\sqrt{2\bar{T}_k / N}$", 
         ha='center', va='center', fontsize=9.5, fontweight='bold', bbox=box_test)

# 분기 화살표
ax2.annotate('', xy=(0.23, 0.49), xytext=(0.43, 0.59), arrowprops=dict(arrowstyle="->", lw=1.8, color='#d63031'))
ax2.annotate('', xy=(0.77, 0.49), xytext=(0.57, 0.59), arrowprops=dict(arrowstyle="->", lw=1.8, color='#0984e3'))

ax2.text(0.23, 0.43, "Statistically Significant ($p < 0.05$)\n" + r"$\mathcal{K}_{\mathrm{signal}} = \{k \mid \ln \bar{E}_k > \mathrm{Upper}\}$" + "\n+ Final Residue $r_n(t)$ (Mean Shift)", 
         ha='center', va='center', fontsize=8.5, fontweight='bold', bbox=box_sig)

ax2.text(0.77, 0.43, r"White Noise Hypothesis ($p \geq 0.05$)" + "\n" + r"$\mathcal{K}_{\mathrm{noise}} = \{k \mid \ln \bar{E}_k \leq \mathrm{Upper}\}$" + "\n(IMF$_1$ Anchored Baseline)", 
         ha='center', va='center', fontsize=8.5, fontweight='bold', bbox=box_noise)

ax2.annotate('', xy=(0.23, 0.27), xytext=(0.23, 0.35), arrowprops=dict(arrowstyle="->", lw=1.8, color='#d63031'))
ax2.annotate('', xy=(0.77, 0.27), xytext=(0.77, 0.35), arrowprops=dict(arrowstyle="->", lw=1.8, color='#0984e3'))

ax2.text(0.23, 0.20, "Intervention Signal ($I_t$)\n" + r"$I_t = \sum_{k \in \mathcal{K}_{\mathrm{signal}}} \mathrm{IMF}_k + r_n$" + "\n(Captures Transient Shocks & Shifts)", 
         ha='center', va='center', fontsize=8.5, fontweight='bold', color='#c0392b', bbox=box_out)

ax2.text(0.77, 0.20, "Purified Noise ($R_{\mathrm{pure}}$)\n" + r"$R_{\mathrm{pure}} = \sum_{k \in \mathcal{K}_{\mathrm{noise}}} \mathrm{IMF}_k$" + "\n(Recovers Clean $i.i.d.$ Noise)", 
         ha='center', va='center', fontsize=8.5, fontweight='bold', color='#2980b9', bbox=box_out)

ax2.annotate('', xy=(0.5, 0.09), xytext=(0.33, 0.13), arrowprops=dict(arrowstyle="->", lw=1.8, color='#2d3436'))
ax2.annotate('', xy=(0.5, 0.09), xytext=(0.67, 0.13), arrowprops=dict(arrowstyle="->", lw=1.8, color='#2d3436'))

ax2.text(0.5, 0.04, r"Final Strength Index: $F_I = \max\left(0, 1 - \frac{\mathrm{Var}(R_{\mathrm{pure}})}{\mathrm{Var}(I_t + R_{\mathrm{pure}})}\right)$",
         ha='center', va='center', fontsize=10, fontweight='bold', color='#2d3436', bbox=box_final)

# -------------------------------------------------------------------------
# 플롯 저장 및 출력
# -------------------------------------------------------------------------
save_path = './results/plot/wu_huang_significance_test.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 플롯이 성공적으로 저장되었습니다: {save_path}")
plt.show()