import numpy as np

def simulate_nucleus(protons=2, neutrons=2):
    """
    N.E.A. 原子核“顶点占用”与“租金”模拟器 v1.0
    """
    # 1. 硬件常数 (基于 GTUOCE 序列结算)
    C8_VERTICES = 8
    C8_BANDWIDTH = 1.0  # ZY，单个C8胞元的单位带宽池
    
    # K4 资产属性
    # 质子占用4个点，中子占用4个点
    POINTS_PER_K4 = 4
    
    # 租金成本
    H_BASE = 1.3333  # 基线租金
    K4_HOLDING_COST = 1.50  # ZY，K4 单位节点持有成本
    BANKRUPTCY_LINE = 2.0  # ZY，拓扑清算红线
    
    # 2. 结算计算
    total_baryons = protons + neutrons
    
    # 顶点占用计算
    occupied_vertices = protons * POINTS_PER_K4 + neutrons * POINTS_PER_K4
    # 注意：这是一个极简模型。
    # 在真实的红蓝 K4 互补模型中，1个质子和1个中子完美互补，只占8个点。
    # 这里为了通用性，先计算总占用数，后面再判断冲突。
    
    # 如果是互补对 (1p+1n 完美填充一个 C8)
    pairs = min(protons, neutrons)
    remaining_protons = protons - pairs
    remaining_neutrons = neutrons - pairs
    
    # 完美配对的占用
    paired_occupancy = pairs * 8
    
    # 剩余核子的占用 (剩下的没有互补伙伴，会竞争槽位)
    unpaired_occupancy = (remaining_protons + remaining_neutrons) * POINTS_PER_K4
    # 但实际上剩余的同类核子会争夺同一个颜色的顶点，产生“电荷排斥”
    
    total_occupancy = paired_occupancy + unpaired_occupancy
    
    # 3. 租金与稳定性审计
    # 每单位带宽能支持的 K4 数量
    supported_k4 = C8_BANDWIDTH / K4_HOLDING_COST
    
    print(f"\n{'='*50}")
    print(f"   N.E.A. 原子核结算报告 (Z={protons}, N={neutrons})")
    print(f"{'='*50}")
    
    print(f"\n【顶点占用审计】")
    print(f"  总重子数: {total_baryons}")
    print(f"  所需顶点数: {total_occupancy} (若全部分散)")
    print(f"  C8 胞元槽位: {C8_VERTICES}")
    
    if protons == neutrons and protons > 0:
        print(f"\n  ✅ 完美互补对 ({pairs}对 p-n)")
        print(f"  红蓝 K4 完美填充 C8 的 8 个顶点，无冲突。")
        avg_vertex_load = total_occupancy / C8_VERTICES
    else:
        print(f"\n  ⚠️  存在未配对核子")
        print(f"  配对: {pairs}对, 剩余质子: {remaining_protons}, 剩余中子: {remaining_neutrons}")
        if remaining_protons > 0:
            print(f"  🔴 警告: 存在 {remaining_protons} 个未配对质子")
            print(f"     这些质子会争夺相同的红色槽位，产生“电荷排斥” (顶点占用冲突)")
        avg_vertex_load = total_occupancy / C8_VERTICES
    
    # 4. 债务压力审计 (简化版)
    # 完美配对时，每个C8顶点负担一个K4的边
    # 真实节点压力 = (K4数量 * 每条边的租金) / 共享顶点数
    # 这里简化为单胞元模型
    
    if pairs > 0 and remaining_protons == 0 and remaining_neutrons == 0:
        # 纯互补对
        node_pressure = K4_HOLDING_COST * 2 / C8_VERTICES  # 两个K4共享8个点
        print(f"\n【租金压力审计】")
        print(f"  完美配对模式: 节点压强 = {node_pressure:.4f} ZY")
    else:
        # 有冲突时，压强升高
        conflict_penalty = 1.0 + 0.5 * (remaining_protons + remaining_neutrons)
        node_pressure = K4_HOLDING_COST * conflict_penalty / C8_VERTICES
        print(f"\n【租金压力审计】")
        print(f"  冲突模式: 节点压强 = {node_pressure:.4f} ZY (含冲突惩罚)")
    
    print(f"\n【审计判定】")
    print(f"  当前节点压强: {node_pressure:.4f} ZY")
    print(f"  破产线: {BANKRUPTCY_LINE:.2f} ZY")
    
    if node_pressure < 1.5:
        print(f"  ✅ 稳定资产 (压强安全边际充足)")
    elif node_pressure < BANKRUPTCY_LINE:
        print(f"  ⚠️  承压资产 (压强较高，但未破产)")
    else:
        print(f"  ❌ 拓扑破产 (压强超过清算线，该组合不稳定)")
    
    print(f"\n{'='*50}\n")
    return node_pressure

# 试试不同原子核
print("🔬 探索不同原子核在 N.E.A. 因果图里的稳定性...\n")
simulate_nucleus(protons=1, neutrons=1)  # 氘核 (稳定)
simulate_nucleus(protons=2, neutrons=2)  # 氦-4 (非常稳定)
simulate_nucleus(protons=3, neutrons=3)  # 锂-6 (稳定)
simulate_nucleus(protons=3, neutrons=4)  # 锂-7 (稳定)
simulate_nucleus(protons=2, neutrons=0)  # 氦-2 (极不稳定，现实中不存在)
simulate_nucleus(protons=3, neutrons=1)  # 锂-4 (极不稳定)