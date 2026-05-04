import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def run_alchemy_audit():
    print("====================================================")
    print("   N.E.A. 炼金术账本：全球核素资产稳定性大普查 (v1.8)   ")
    print("====================================================\n")

    # 1. 继承 N.E.A. 核心基因 (GTUOCE 序列)
    U_weak = 10 * np.sqrt(3)          # 17.3205 ZY
    R = 1.0 / (1.0 + np.pi)           # 0.2415
    ZY_to_MeV = 0.406640              # ZY-SI 转换率
    stiffness = 1.0526                # 1.05 硬度比

    # 2. 导出 N.E.A. 核力常数 (MeV)
    # 纯几何推导，严禁凑数
    a_v = (9.0 / 4.0) * U_weak / stiffness * ZY_to_MeV   # 体积能
    a_s = (10.0 / 4.0) * U_weak * ZY_to_MeV              # 表面能
    a_c = np.sqrt(3.0) * stiffness * ZY_to_MeV           # 库仑能
    a_a = (np.pi + 1.0) * (U_weak / np.sqrt(2.0)) * ZY_to_MeV # 对称能
    a_p = 12.0 * ZY_to_MeV # 配对能基准 (25位机器的偶数对齐红利，暂用经验值)

    print(f"[审计配置] 正在使用 25位机器逻辑定标...")
    print(f"  > a_v: {a_v:.2f}, a_s: {a_s:.2f}, a_c: {a_c:.4f}, a_a: {a_a:.2f}\n")

    # 3. 构造同位素矩阵 (Z: 1-118, N: 1-180)
    Z_range = np.arange(1, 119)
    N_range = np.arange(1, 181)
    stability_map = np.zeros((len(Z_range), len(N_range)))

    # 4. 执行全量审计
    for i, Z in enumerate(Z_range):
        for j, N in enumerate(N_range):
            A = Z + N
            # 核子结合能结算公式 (Liquid Drop Topology)
            vol = a_v * A
            sur = -a_s * (A**(2/3.0))
            cou = -a_c * (Z**2) / (A**(1/3.0))
            sym = -a_a * ((N - Z)**2) / A
            
            # 配对红利审计 (偶偶核最稳)
            if Z % 2 == 0 and N % 2 == 0:
                pair = a_p / np.sqrt(A)
            elif Z % 2 != 0 and N % 2 != 0:
                pair = -a_p / np.sqrt(A)
            else:
                pair = 0
                
            be_total = vol + sur + cou + sym + pair
            be_per_nucleon = be_total / A
            
            # 记录资产价值，若破产(BE < 0)则标记为 0
            stability_map[i, j] = max(0, be_per_nucleon)

    # 5. 寻找全表“铁锚点”
    max_be = np.max(stability_map)
    loc = np.where(stability_map == max_be)
    best_z = Z_range[loc[0][0]]
    best_n = N_range[loc[1][0]]

    print(f"[审计结果]")
    print(f"  > 账本扫描完成。")
    print(f"  > 全宇宙最稳健资产锚点: Z={best_z}, N={best_n} (A={best_z+best_n})")
    print(f"  > 最高单位资产净值: {max_be:.4f} MeV/核子")
    
    # 6. 可视化：核素稳定谷
    plt.figure(figsize=(12, 8))
    plt.imshow(stability_map, origin='lower', extent=[1, 180, 1, 118], 
               cmap='magma', aspect='auto')
    plt.colorbar(label='Binding Energy per Nucleon (MeV)')
    plt.contour(stability_map, levels=[7.0, 8.0, 8.5], colors='white', alpha=0.3)
    plt.title("N.E.A. Alchemy Ledger: The Global Isotope Stability Map")
    plt.xlabel("Number of Neutrons (N)")
    plt.ylabel("Number of Protons (Z)")
    
    # 标注重要节点
    plt.scatter([30], [26], color='cyan', marker='*', s=100, label='Fe-56 Anchor')
    plt.scatter([best_n], [best_z], color='lime', marker='x', s=100, label='N.E.A. Optimal')
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_alchemy_audit()