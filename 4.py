"""
杨氏双缝干涉实验仿真
=====================
仿真内容：
1. 双缝干涉强度分布（含单缝衍射包络）
2. 屏幕上的干涉条纹图案（2D可视化）
3. 波前叠加动画式静态图（可选）
4. 参数可调：波长、缝间距、缝宽、屏幕距离

依赖库：numpy, matplotlib
安装：pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib

# 自动适配中文字体（Windows/Mac/Linux 常见字体依次尝试）
for _font in ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC',
              'WenQuanYi Zen Hei', 'Heiti TC']:
    if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        matplotlib.rcParams['font.sans-serif'] = [_font]
        break
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ----------------------------
# 1. 物理参数设置（单位：米）
# ----------------------------
wavelength = 632.8e-9      # 波长，默认氦氖激光 632.8 nm
d = 0.3e-3                 # 双缝间距 (slit separation)
a = 0.05e-3                # 单缝宽度 (slit width)
L = 1.5                    # 缝到屏幕的距离

# 屏幕上的观测范围（y方向，单位：米）
y = np.linspace(-0.02, 0.02, 4000)

k = 2 * np.pi / wavelength

# ----------------------------
# 2. 强度分布计算
# ----------------------------
def single_slit_envelope(y, a, L, wavelength):
    """单缝衍射包络 (sinc^2 函数)"""
    beta = (np.pi * a * y) / (wavelength * L)
    # 避免除零
    beta = np.where(beta == 0, 1e-12, beta)
    return (np.sin(beta) / beta) ** 2

def double_slit_interference(y, d, L, wavelength):
    """双缝干涉因子 (cos^2 函数)"""
    delta = (np.pi * d * y) / (wavelength * L)
    return np.cos(delta) ** 2

envelope = single_slit_envelope(y, a, L, wavelength)
interference = double_slit_interference(y, d, L, wavelength)
intensity_total = envelope * interference   # 总强度 = 衍射包络 × 干涉项
intensity_pure = interference               # 理想点缝双缝干涉（无衍射包络）

# 归一化
intensity_total /= intensity_total.max()
intensity_pure /= intensity_pure.max()

# ----------------------------
# 3. 可视化
# ----------------------------
fig = plt.figure(figsize=(11, 8))
gs = GridSpec(3, 1, height_ratios=[2, 1, 2], hspace=0.5)

# (a) 强度曲线图
ax1 = fig.add_subplot(gs[0])
ax1.plot(y * 1000, intensity_pure, label='理想双缝干涉 (忽略衍射)', color='tab:blue', alpha=0.6)
ax1.plot(y * 1000, intensity_total, label='双缝干涉 × 单缝衍射包络', color='tab:red', linewidth=1.5)
ax1.plot(y * 1000, envelope, '--', label='单缝衍射包络', color='gray', linewidth=1)
ax1.set_xlabel('屏幕位置 y (mm)')
ax1.set_ylabel('归一化强度')
ax1.set_title(f'杨氏双缝干涉强度分布\n'
              f'λ={wavelength*1e9:.1f} nm, d={d*1e3:.2f} mm, a={a*1e3:.3f} mm, L={L} m')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(alpha=0.3)

# (b) 屏幕条纹（将强度映射为灰度图像，模拟真实屏幕上看到的条纹）
ax2 = fig.add_subplot(gs[1])
# 构造二维条纹图案：把一维强度沿竖直方向复制
fringe_image = np.tile(intensity_total, (100, 1))
ax2.imshow(fringe_image, extent=[y.min()*1000, y.max()*1000, 0, 1],
           cmap='inferno', aspect='auto')
ax2.set_yticks([])
ax2.set_xlabel('屏幕位置 y (mm)')
ax2.set_title('屏幕上观察到的干涉条纹（模拟真实亮暗分布）')

# (c) 波场叠加图：模拟双缝出射球面波在空间中的叠加（俯视图）
ax3 = fig.add_subplot(gs[2])

x_field = np.linspace(0, L, 600)         # 传播方向
y_field = np.linspace(-0.02, 0.02, 600)  # 横向
X, Y = np.meshgrid(x_field, y_field)

slit1_y = d / 2
slit2_y = -d / 2

# 两个缝到场点的距离
r1 = np.sqrt((X - 0) ** 2 + (Y - slit1_y) ** 2) + 1e-9
r2 = np.sqrt((X - 0) ** 2 + (Y - slit2_y) ** 2) + 1e-9

# 球面波（简化为2D柱面波）叠加，实部表示振幅
wave1 = np.cos(k * r1) / np.sqrt(r1)
wave2 = np.cos(k * r2) / np.sqrt(r2)
field = wave1 + wave2
field_intensity = field ** 2

im = ax3.imshow(field_intensity, extent=[x_field.min(), x_field.max(),
                                          y_field.min()*1000, y_field.max()*1000],
                 cmap='RdBu_r', aspect='auto', origin='lower',
                 vmax=np.percentile(field_intensity, 99))
ax3.set_xlabel('传播方向 x (m)')
ax3.set_ylabel('横向位置 y (mm)')
ax3.set_title('双缝波场叠加示意图（俯视，展示波前干涉结构）')
ax3.plot([0, 0], [slit1_y*1000, slit2_y*1000], 'yo', markersize=4)

plt.tight_layout()
plt.savefig('double_slit_result.png', dpi=150)  # 保存到脚本所在的当前工作目录
plt.show()

print("仿真完成！结果图像已保存为 double_slit_result.png")
print(f"参数: 波长={wavelength*1e9:.1f}nm, 缝间距={d*1e3:.2f}mm, "
      f"缝宽={a*1e3:.3f}mm, 屏距={L}m")

# ----------------------------
# 4. 条纹间距理论计算（供验证）
# ----------------------------
fringe_spacing = wavelength * L / d
print(f"理论条纹间距 Δy = λL/d = {fringe_spacing*1000:.4f} mm")