import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

def nea_refined_shell_audit(L=25, Z_eff=15.0):
    """
    Z_eff: 有效电荷强度。强度越高，能层分得越开。
    """
    N = L**3
    center = L // 2
    
    # 1. 构建更加精确的 C8 算子 (考虑 1D 骨架的单位步长)
    # 我们使用标准 3D 拉普拉斯
    def get_idx(x, y, z): return x + y*L + z*L**2
    
    print(f"正在配置 3D C8 硬件环境 (L={L})...")
    adj_data, rows, cols = [], [], []
    for z in range(L):
        for y in range(L):
            for x in range(L):
                u = get_idx(x, y, z)
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    nx, ny, nz = (x+dx)%L, (y+dy)%L, (z+dz)%L # 周期性边界减少边缘效应
                    v = get_idx(nx, ny, nz)
                    adj_data.append(1.0); rows.append(u); cols.append(v)
    
    A = sp.csr_matrix((adj_data, (rows, cols)), shape=(N, N))
    D = sp.diags(np.array(A.sum(axis=1)).flatten())
    L_mat = D - A # 离散动能项

    # 2. 增强势阱吸引力 (模拟高租金的 K4 核)
    V_diag = np.zeros(N)
    for i in range(N):
        z, y, x = i // L**2, (i % L**2) // L, i % L
        dist = np.sqrt((x-center)**2 + (y-center)**2 + (z-center)**2)
        if dist < 0.5: dist = 0.5 # 防止中心奇异点过深
        V_diag[i] = -Z_eff / dist # 增加 Z_eff 使能层分离
    
    H_mat = L_mat + sp.diags(V_diag)

    # 3. 求解更多特征值
    k_modes = 20
    evals, _ = eigsh(H_mat, k=k_modes, which='SA')
    
    # 4. 自动审计归类 (将差异小于 0.05 的能级视为同一层简并)
    shells = []
    current_shell = [evals[0]]
    for i in range(1, len(evals)):
        if abs(evals[i] - evals[i-1]) < 0.05:
            current_shell.append(evals[i])
        else:
            shells.append(current_shell)
            current_shell = [evals[i]]
    shells.append(current_shell)

    # 5. 打印数值报告
    print("\n" + "="*40)
    print("      N.E.A. 原子能层资产审计报告")
    print("="*40)
    for n, shell in enumerate(shells):
        avg_energy = np.mean(shell)
        degeneracy = len(shell)
        print(f"能层 n={n+1} | 平均租金: {avg_energy:.4f} ZY | 简并度(电子位): {degeneracy}")
    print("="*40)

    # 6. 可视化
    plt.figure(figsize=(8, 10))
    for n, shell in enumerate(shells):
        label = f"Shell n={n+1} (deg:{len(shell)})"
        plt.hlines(shell, 0.2, 0.7, colors=f"C{n}", linewidth=2, label=label)
        plt.text(0.72, np.mean(shell), f"n={n+1}", fontweight='bold')

    plt.title(f"N.E.A. Energy Shell Split (Z_eff={Z_eff})")
    plt.ylabel("Enthalpy (Eigenvalues)")
    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=0.2)
    plt.show()

nea_refined_shell_audit(L=25, Z_eff=25.0)