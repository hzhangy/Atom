import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

def audit_and_tune_levels(L=31, Z_eff=60.0):
    """
    通过增大 Z_eff 强制压低基态，拉开能级间距。
    """
    N = L**3
    center = L // 2
    def get_idx(x, y, z): return x + y*L + z*L**2

    # 1. 构建 C8 支架 (增加周期性边界以减少边缘反射)
    adj_data, rows, cols = [], [], []
    for z in range(L):
        for y in range(L):
            for x in range(L):
                u = get_idx(x, y, z)
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    nx, ny, nz = (x+dx)%L, (y+dy)%L, (z+dz)%L
                    v = get_idx(nx, ny, nz)
                    adj_data.append(1.0); rows.append(u); cols.append(v)
    
    A = sp.csr_matrix((adj_data, (rows, cols)), shape=(N, N))
    L_mat = sp.diags(np.array(A.sum(axis=1)).flatten()) - A

    # 2. 构造核势阱 (增加有效电荷 Z_eff 以拉开能级)
    V_diag = np.zeros(N)
    for i in range(N):
        z, y, x = i // L**2, (i % L**2) // L, i % L
        dist = np.sqrt((x-center)**2 + (y-center)**2 + (z-center)**2)
        # 核心修正：使用 Stride-10 物理截断 (0.5 晶格单位)
        V_diag[i] = -Z_eff / max(dist, 0.5)

    H_mat = L_mat + sp.diags(V_diag)

    # 3. 求解前 25 个本征态
    print(f"正在审计原子能级 (Z_eff={Z_eff}, L={L})...")
    evals, _ = eigsh(H_mat, k=25, which='SA')
    evals = np.sort(evals)

    # 4. 自动聚类审计 (阈值 0.5 ZY 用于区分不同能层)
    print("\n" + "="*55)
    print(f"{'能级索引':<10} | {'特征值 (租金)':<15} | {'初步分层建议'}")
    print("-" * 55)
    
    last_val = evals[0]
    shell_id = 1
    for i, val in enumerate(evals):
        if abs(val - last_val) > 0.15: # 如果跳跃超过 0.15 ZY，视为新的一层
            shell_id += 1
            print("-" * 55)
        
        # 标注简并度含义
        note = f"Shell n={shell_id}"
        print(f"Mode {i:<7} | {val:<15.6f} | {note}")
        last_val = val
    print("="*55)

    # 5. 可视化
    plt.figure(figsize=(7, 10))
    plt.scatter(np.ones_like(evals), evals, color='red', marker='_')
    plt.title(f"N.E.A. Energy Shell Audit (Z_eff={Z_eff})")
    plt.ylabel("Enthalpy Rent (ZY)")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # 建议先用 Z_eff=60 观察。
    # 如果 1s 和 2s 还是分不开，就把 Z_eff 提到 100。
    audit_and_tune_levels(L=31, Z_eff=100.0)