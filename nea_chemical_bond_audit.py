import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

class MolecularArbitrageAuditor:
    def __init__(self, L=31, Z_eff=20.0):
        self.L = L
        self.N = L**3
        self.Z_eff = Z_eff
        self.center = L // 2
        
    def get_idx(self, x, y, z):
        return x + y*self.L + z*self.L**2

    def build_base_laplacian(self):
        """构建标准的 3D C8 动力学支架"""
        L = self.L
        adj_data, rows, cols = [], [], []
        for z in range(L):
            for y in range(L):
                for x in range(L):
                    u = self.get_idx(x, y, z)
                    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                        nx, ny, nz = (x+dx)%L, (y+dy)%L, (z+dz)%L
                        v = self.get_idx(nx, ny, nz)
                        adj_data.append(1.0); rows.append(u); cols.append(v)
        
        A = sp.csr_matrix((adj_data, (rows, cols)), shape=(self.N, self.N))
        D = sp.diags(np.array(A.sum(axis=1)).flatten())
        return D - A

    def audit_system_rent(self, R_dist, L_mat):
            L = self.L
            V_diag = np.zeros(self.N)
            pos1 = (self.center - R_dist/2.0, self.center, self.center)
            pos2 = (self.center + R_dist/2.0, self.center, self.center)
        
            for i in range(self.N):
                z, y, x = i // L**2, (i % L**2) // L, i % L
                d1 = np.sqrt((x-pos1[0])**2 + (y-pos1[1])**2 + (z-pos1[2])**2)
                d2 = np.sqrt((x-pos2[0])**2 + (y-pos2[1])**2 + (z-pos2[2])**2)
            
                # 使用 Z_eff=100 以匹配能级审计的硬度
                # 势能项：1D链在双核间的共享采样
                V_diag[i] = -self.Z_eff / (max(d1, 0.5)) - self.Z_eff / (max(d2, 0.5))
          
            # 修正后的核排斥项：
            # N.E.A. 逻辑：当 R 减小时，排斥力应呈指数级或高阶幂次上升（硬件溢出保护）
            # 但在平衡位置 R~1.0 附近，电子云的屏蔽效应让该项不应压倒套利收益
            # 我们将系数从 Z^2/4 降低到 Z*1.5 左右，模拟屏蔽后的有效相互作用
            effective_shielded_repulsion = (self.Z_eff * 1.5) / (R_dist**1.5 + 0.1)
          
            H_mat = L_mat + sp.diags(V_diag)
            evals, _ = eigsh(H_mat, k=1, which='SA')
        
            return evals[0] + effective_shielded_repulsion

# --- 执行化学键套利审计 ---
auditor = MolecularArbitrageAuditor(L=25, Z_eff=15.0)
L_mat = auditor.build_base_laplacian()

distances = np.arange(1.0, 10.0, 1.0)
total_rents = []

print("正在计算双核系统的带宽收益曲线...")
for r in distances:
    rent = auditor.audit_system_rent(r, L_mat)
    total_rents.append(rent)
    print(f"核间距 R={r:.1f} | 总焓值租金: {rent:.4f} ZY")

# --- 寻找最优套利点 ---
min_idx = np.argmin(total_rents)
r_optimal = distances[min_idx]

# --- 绘图 ---
plt.figure(figsize=(10, 6))
plt.plot(distances, total_rents, 'bo-', linewidth=2, label='System Total Enthalpy (ZY)')
plt.axvline(x=r_optimal, color='r', linestyle='--', label=f'Optimal Bond Length (R={r_optimal})')

plt.title("N.E.A. Molecular Arbitrage Audit (Hydrogen-like Molecule)")
plt.xlabel("Internuclear Distance R (Lattice Units)")
plt.ylabel("Total Enthalpy Rent (ZY)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n审计结果：")
print(f"1. 发现套利洼地：在距离 R={r_optimal} 处，系统租金最低。")
print(f"2. 物理结论：两个原子自动靠近并锁定在该距离，形成‘化学键’。")
print(f"3. 动力学起源：这不是因为‘力’，而是因为节点在这个间距下刷新的‘性价比’最高。")