import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. 캔버스 및 스타일 설정
plt.figure(figsize=(11, 10))
ax = plt.gca()

# 2. 각 Sector별 배경 색상 영역 (Patches) 설정
# (1) Composite Sector: F_T >= 0.6 and F_S >= 0.4
rect_composite = patches.Rectangle(
    (0.6, 0.4), 0.4, 0.6,
    linewidth=0, facecolor='#d63031', alpha=0.18, label='Composite Sector'
)
# (2) Seasonal Sector: F_T < 0.6 and F_S >= 0.4
rect_seasonal = patches.Rectangle(
    (0.0, 0.4), 0.6, 0.6,
    linewidth=0, facecolor='#0984e3', alpha=0.18, label='Seasonal Sector'
)
# (3) Trending Sector: F_T >= 0.6 and F_S < 0.4
rect_trending = patches.Rectangle(
    (0.6, 0.0), 0.4, 0.4,
    linewidth=0, facecolor='#e67e22', alpha=0.18, label='Trending Sector'
)
# (4) Stationary (Noising) Sector: F_T <= 0.25 and F_S <= 0.25
rect_stationary = patches.Rectangle(
    (0.0, 0.0), 0.25, 0.25,
    linewidth=0, facecolor='#2ed573', alpha=0.25, label='Noising Sector'
)
# (5) Buffer / Transition Zone: 0.25 < F_T < 0.6 or 0.25 < F_S < 0.4
rect_buffer = patches.Rectangle(
    (0.0, 0.0), 0.6, 0.4,
    linewidth=0, facecolor='#b2bec3', alpha=0.10, label='Buffer / Weak Pattern Zone'
)

# 배경 패치 추가
ax.add_patch(rect_buffer)
ax.add_patch(rect_composite)
ax.add_patch(rect_seasonal)
ax.add_patch(rect_trending)
ax.add_patch(rect_stationary)

# 3. 임계치(Threshold) 기준 점선 렌더링
plt.axvline(x=0.6, color='#2d3436', linestyle='--', linewidth=2.0, label=r'Major Threshold ($F_T=0.6, F_S=0.4$)')
plt.axhline(y=0.4, color='#2d3436', linestyle='--', linewidth=2.0)

plt.plot([0.25, 0.25], [0.0, 0.25], color='#009432', linestyle=':', linewidth=2.2, label=r'Stationary Threshold ($0.25$)')
plt.plot([0.0, 0.25], [0.25, 0.25], color='#009432', linestyle=':', linewidth=2.2)

# 4. 각 Sector 영역 텍스트 표기 (\ge -> \geq, \le -> \leq, Raw String 사용)
plt.text(0.80, 0.70, 
         r"""Composite Sector
($F_T \geq 0.6, F_S \geq 0.4$)""", 
         fontsize=12, fontweight='bold', ha='center', va='center', color='#b71540',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#b71540', alpha=0.85))

plt.text(0.30, 0.70, 
         r"""Seasonal Sector
($F_T < 0.6, F_S \geq 0.4$)""", 
         fontsize=12, fontweight='bold', ha='center', va='center', color='#0a3d62',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#0a3d62', alpha=0.85))

plt.text(0.80, 0.20, 
         r"""Trending Sector
($F_T \geq 0.6, F_S < 0.4$)""", 
         fontsize=12, fontweight='bold', ha='center', va='center', color='#d35400',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#d35400', alpha=0.85))

plt.text(0.125, 0.125, 
         r"""Noising Sector
($F_T \leq 0.25, F_S \leq 0.25$)""", 
         fontsize=10.5, fontweight='bold', ha='center', va='center', color='#006266',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#006266', alpha=0.85))

plt.text(0.42, 0.15, 'Weak Pattern\n(Buffer Zone)', 
         fontsize=10, fontstyle='italic', ha='center', va='center', color='#636e72')

# 5. 축 서식 및 레이아웃 설정
plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)
plt.xlabel(r'Trend Strength ($F_T$)', fontsize=13, fontweight='bold', labelpad=10)
plt.ylabel(r'Seasonal Strength ($F_S$)', fontsize=13, fontweight='bold', labelpad=10)
plt.title('Synthetic Time-series Data Sector Mapping', fontsize=15, fontweight='bold', pad=15)

# 눈금에 임계치 위치 명시 (r'...' 적용으로 \tau의 \t 탭 변환 방지)
plt.xticks([0.0, 0.25, 0.6, 0.8, 1.0], ['0.0', r'0.25 ($\tau_{noise}$)', r'0.6 ($\tau_T$)', '0.8', '1.0'], fontsize=10)
plt.yticks([0.0, 0.25, 0.4, 0.6, 0.8, 1.0], ['0.0', r'0.25 ($\tau_{noise}$)', r'0.4 ($\tau_S$)', '0.6', '0.8', '1.0'], fontsize=10)

plt.grid(True, linestyle=':', alpha=0.4)
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10.5, framealpha=0.95)
plt.tight_layout()
plt.savefig("2D_Sector_Decision_Map.png", dpi=300, bbox_inches='tight')
plt.show()