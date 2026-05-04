import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

def nea_orbital_eigenmode_audit(L=21, num_modes=10):
    """
    N.E.A. 电子轨道本征模审计
    L: 立方体边长 (建议奇数以便中心对称)
    num_modes: 求解的特征向量数量
    """
    N = L**3
    print(f"正在构建 L={L} 的 C8 离散格点 (节点总数: {N})...")
    
    # 1. 构建标准 C8 (3D 立方) 图拉普拉斯矩阵
    # 我们使用标准 7 点差分格式（中心节点 + 6 个邻居）
    dx = dy = dz = 1
    
    def get_idx(x, y, z):
        return x + y*L + z*L**2

    # 构建邻接矩阵 A
    data = []
    rows = []
    cols = []
    
    for z in range(L):
        for y in range(L):
            for x in range(L):
                u = get_idx(x, y, z)
                # 检查 6 个邻居 (x±1, y±1, z±1)
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if 0 <= nx < L and 0 <= ny < L and 0 <= nz < L:
                        v = get_idx(nx, ny, nz)
                        data.append(1.0)
                        rows.append(u)
                        cols.append(v)
    
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    D = sp.diags(np.array(A.sum(axis=1)).flatten())
    L_mat = D - A # 基础图拉普拉斯

    # 2. 加入 K4 原子核扰动 (中心点势阱)
    # 在 N.E.A 中，原子核是高租金区域，吸引 1D 链在此刷新
    center = L // 2
    nuclear_idx = get_idx(center, center, center)
    
    # 模拟势阱: V(r) = -1/r (离散化)
    V_diag = np.zeros(N)
    for z in range(L):
        for y in range(L):
            for x in range(L):
                dist = np.sqrt((x-center)**2 + (y-center)**2 + (z-center)**2)
                if dist == 0:
                    V_diag[get_idx(x,y,z)] = -5.0 # 中心点极强吸引
                else:
                    V_diag[get_idx(x,y,z)] = -1.0 / dist
    
    H_mat = L_mat + sp.diags(V_diag) # N.E.A 离散哈密顿算子

    # 3. 求解特征值和特征向量 (最小的几个)
    print("正在计算 C8 支架上的拓扑本征模...")
    evals, evecs = eigsh(H_mat, k=num_modes, which='SA')

    # 4. 可视化
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"N.E.A. $C_8$ Scaffold Eigenmode Audit (Central Slice)", fontsize=16)
    
    orbitals = ['1s', '2p_x', '2p_y', '2p_z', '2s?', '3d_1', '3d_2', '3d_3', '3d_4', '3d_5']
    
    for i in range(num_modes):
        ax = axes[i//5, i%5]
        # 提取中心切面 (z = center)
        mode_3d = evecs[:, i].reshape((L, L, L))
        slice_2d = mode_3d[:, :, center]
        
        im = ax.imshow(slice_2d, cmap='RdBu', interpolation='bilinear')
        ax.set_title(f"Mode {i}\nVal: {evals[i]:.4f}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

    # 5. 打印简并度审计
    print("\n--- 简并度审计报告 ---")
    for i in range(num_modes):
        print(f"特征模式 {i}: 能量(特征值) = {evals[i]:.6f}")

if __name__ == "__main__":
    nea_orbital_eigenmode_audit()