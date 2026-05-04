import numpy as np
import matplotlib.pyplot as plt

def run_final_audit():
    # 1. 继承核心资产 (GTUOCE 序列)
    U_weak = 10 * np.sqrt(3)          # 17.3205 ZY (寻址总开户费)
    R = 1.0 / (1.0 + np.pi)           # 0.2415 (投影残差)
    ZY_to_MeV = 0.406640              # 结算汇率 (由 me 锚定)
    
    # 来自 Volume II 'nea_force_range_stiffness_audit.py' 的硬度实验结果
    # K4 (物质) 与 C8 (空间) 的物理响应比
    stiffness_ratio = 1.0526          
    
    # 2. 导出核力常数 (无量纲几何商)
    # 体积能 a_v: 寻址比 / 硬度比
    # 9/4 代表 C8(8+1) 与 K4(4) 的有效寻址槽位比
    a_v_zy = (9.0 / 4.0) * U_weak / stiffness_ratio     
    
    # 表面能 a_s: 寻址步数 / 节点
    # 10/4 代表 Stride-10 寻址包平摊到 K4 的 4 个顶点
    a_s_zy = (10.0 / 4.0) * U_weak    
    
    # 库仑能 a_c: 空间对角线投影 * 硬度比
    # 代表两个 K4 争夺 C8 槽位时的“对角线挤兑阻抗”
    a_c_zy = np.sqrt(3.0) * stiffness_ratio
    
    # 对称能 a_a: 寻址深度在代际混合态下的投影
    # (pi + 1) 是相位环闭合能，sqrt(2) 是 45度代际混合
    a_a_zy = (np.pi + 1.0) * (U_weak / np.sqrt(2.0))

    # 3. 开始全自动审计结算
    results = []
    for A in range(1, 240):
        best_be = -999.0
        # 沿着 Beta 稳定线搜索每个 A 对应的最稳 Z
        for Z in range(1, A + 1):
            N = A - Z
            # N.E.A. 记账公式 (ZY 为单位)
            # 总减损 = 体积收益 - 表面税 - 库仑债 - 对称罚金
            total_reduction = (a_v_zy * A 
                               - a_s_zy * (A**(2/3.0)) 
                               - a_c_zy * (Z**2 / (A**(1/3.0))) 
                               - a_a_zy * ((N - Z)**2 / A))
            be_per_nucleon = (total_reduction / A) * ZY_to_MeV
            if be_per_nucleon > best_be:
                best_be = be_per_nucleon
        results.append(best_be)

    # 4. 打印最终资产评估报告
    print("====================================================")
    print("   N.E.A. v1.7: 19参数全链路核力清算 (终审结果)   ")
    print("====================================================")
    print(f"【核力基因组结算 (MeV)】")
    print(f"  > 体积能 a_v: {a_v_zy * ZY_to_MeV:8.4f} (观测: 15.7)")
    print(f"  > 表面能 a_s: {a_s_zy * ZY_to_MeV:8.4f} (观测: 17.8)")
    print(f"  > 库仑能 a_c: {a_c_zy * ZY_to_MeV:8.4f} (观测: 0.71)")
    print(f"  > 对称能 a_a: {a_a_zy * ZY_to_MeV:8.4f} (观测: 23.2)")
    print("-" * 52)
    
    iron_idx = 55 # A=56
    pb_idx = 207  # A=208
    max_idx = np.argmax(results)
    
    print(f"【宏观锚点判定】")
    print(f"  > 宇宙最优资产锚点: A = {max_idx + 1}")
    print(f"  > 铁 (Fe-56) 结合能: {results[iron_idx]:8.4f} MeV")
    print(f"  > 铅 (Pb-208) 结合能: {results[pb_idx]:8.4f} MeV")
    print(f"  > 铁锚点误差 (vs 8.79): {abs(results[iron_idx]-8.79)/8.79*100:8.4f}%")
        # 5. 绘制结合能曲线
    A_values = np.arange(1, 240)
    plt.figure(figsize=(10, 6))
    plt.plot(A_values, results, 'b-', linewidth=1.5, label='N.E.A. B/A')
    plt.axvline(x=62, color='red', linestyle='--', alpha=0.8, label='Nickel anchor A=62')
    plt.axhline(y=results[55], color='gray', linestyle=':', alpha=0.5, label=f'Fe-56: {results[55]:.2f} MeV')
    plt.xlabel('Mass Number A')
    plt.ylabel('Binding Energy per Nucleon (MeV)')
    plt.title('N.E.A. Nuclear Binding Energy Audit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/binding_energy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[Figure] Binding energy curve saved to figures/binding_energy.png")
    print("====================================================")

if __name__ == "__main__":
    run_final_audit()