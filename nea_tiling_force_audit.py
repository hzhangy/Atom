import numpy as np
import networkx as nx
import scipy.linalg as la
import matplotlib.pyplot as plt

def calculate_nea_rent(G):
    """计算图的总焓租金：基于拉普拉斯特征值的平方根和"""
    L = nx.laplacian_matrix(G).toarray()
    eigenvals = la.eigvalsh(L)
    # 过滤掉由于浮点误差产生的负值或接近0的值
    eigenvals = np.maximum(eigenvals, 1e-10)
    # 物理刷新频率正比于特征值的平方根
    rent = np.sum(np.sqrt(eigenvals))
    return rent, eigenvals

def calculate_ipr(G):
    """计算逆参与比 (IPR)：衡量模态的定位程度。IPR越大，定位越强"""
    L = nx.laplacian_matrix(G).toarray()
    _, vecs = la.eigh(L)
    # IPR = sum(psi_i^4) / (sum(psi_i^2))^2
    ipr_list = np.sum(vecs**4, axis=0)
    return np.mean(ipr_list)

def run_audit():
    print("--- N.E.A. 密铺-力强度 深度审计开始 ---")
    
    # 1. 构造密铺载体：C8 (Cube)
    # 模拟一个无限延展空间的一部分，使用周期性边界条件
    c8 = nx.hypercube_graph(3) # 8节点, 12边
    rent_c8, ev_c8 = calculate_nea_rent(c8)
    ipr_c8 = calculate_ipr(c8)
    
    # 2. 构造非密铺载体：K4 (Simplex/Strong Force)
    k4 = nx.complete_graph(4) # 4节点, 6边
    rent_k4, ev_k4 = calculate_nea_rent(k4)
    ipr_k4 = calculate_ipr(k4)
    
    # 3. 构造大尺度背景空间 (10x10x10 C8格点)
    # 用于计算引力背景下的租金稀释系数
    N_bg = 1000
    # 在 25位寻址机器假设下，有效稀释体积系数
    dilution_factor = np.exp(10 * np.sqrt(3)) / N_bg 
    
    # 4. 核心计算
    # 单位节点的租金成本
    unit_rent_c8 = rent_c8 / 8
    unit_rent_k4 = rent_k4 / 4
    
    # 5. 模拟力强度 (力 = -grad H)
    # 强力：由于不能密铺，租金完全堆积在 4个节点内
    strong_intensity = unit_rent_k4 * ipr_k4
    
    # 引力：由于能密铺，租金平摊到整个背景空间（被 N_max 稀释）
    gravity_intensity = unit_rent_c8 * (1.0 / dilution_factor)
    
    ratio = strong_intensity / gravity_intensity

    print(f"\n[数据结算]")
    print(f"C8 (密铺) 单位租金: {unit_rent_c8:.4f} ZY")
    print(f"K4 (非密铺) 单位租金: {unit_rent_k4:.4f} ZY")
    print(f"K4 定位比 (IPR): {ipr_k4:.4f} (极高，指示能量锁死)")
    print(f"C8 定位比 (IPR): {ipr_c8:.4f} (指示能量离域)")
    
    print(f"\n[力的强度对比]")
    print(f"强力 (堆积效应): {strong_intensity:.4e}")
    print(f"引力 (稀释效应): {gravity_intensity:.4e}")
    print(f"推导强度比 (Strong/Gravity): 10^{np.log10(ratio):.2f}")
    
    # 6. 可视化特征值分布
    plt.figure(figsize=(10, 6))
    plt.step(range(len(ev_c8)), np.sort(ev_c8), label='C8 (Space/Gravity) Spectrum', where='mid')
    plt.step(range(len(ev_k4)), np.sort(ev_k4), label='K4 (Matter/Strong) Spectrum', where='mid')
    plt.title("Spectral Signature of Tiling vs Non-Tiling")
    plt.xlabel("Mode Index")
    plt.ylabel("Eigenvalue (Energy Mode)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_audit()