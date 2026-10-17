import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import signal
from scipy import ndimage

def jd_to_datetime(jd):
    """儒略日转datetime"""
    from datetime import datetime, timedelta
    # 假设jd是修正儒略日（MJD）或普通儒略日，这里需要根据实际情况调整
    # 这里简单处理：假设jd是相对于某个起始点的小数天
    # 实际使用时可能需要根据你的数据格式调整
    return datetime(2000, 1, 1) + timedelta(days=jd)

def mva_wavelet_scales(values0, start, end, 
                       scales='linear',
                       n_scales=32,
                       custom_scales=None,
                       overlap=0.75,
                       r12_threshold=10, 
                       r23_threshold=2,
                       wavelet='morl'):
    """
    多尺度MVA分析，类似小波变换的尺度-时间分析
    
    参数:
    - values0: 包含time_data和b_data的字典
    - start, end: 起始和结束时间（儒略日）
    - scales: 尺度类型 ('linear', 'log', 'dyadic', 'custom')
    - n_scales: 尺度数量（默认32）
    - custom_scales: 自定义尺度列表（当scales='custom'时使用）
    - overlap: 窗口重叠率
    - r12_threshold, r23_threshold: MVA阈值
    """
    
    # 尺度定义 - 从10秒到300秒，32个线性尺度
    if scales == 'linear':
        min_win, max_win = 10, 300  # 最小10秒，最大300秒
        window_sizes = np.linspace(min_win, max_win, n_scales).astype(int)
        
    elif scales == 'dyadic':
        base_scale = 10
        scale_factors = [2**i for i in range(n_scales)]
        window_sizes = [int(base_scale * f) for f in scale_factors]
        # 限制最大尺度不超过300秒
        window_sizes = [w for w in window_sizes if w <= 300]
        
    elif scales == 'log':
        min_win, max_win = 10, 300
        window_sizes = np.logspace(np.log10(min_win), np.log10(max_win), n_scales).astype(int)
    
    elif scales == 'custom' and custom_scales is not None:
        window_sizes = np.array(custom_scales).astype(int)
        n_scales = len(window_sizes)
    
    print(f"分析尺度 (窗口大小): {window_sizes} 秒")
    print(f"尺度数量: {len(window_sizes)}")
    
    # 扩展数据（最大尺度的2倍，确保边界效应可控）
    max_window = max(window_sizes)
    extend_seconds = max_window * 2
    extend_days = extend_seconds / 86400.0
    
    data_start = start - extend_days
    data_end = end + extend_days
    
    time = values0['time_data']
    data_mask = (time >= data_start) & (time <= data_end)
    time_ext = time[data_mask]
    Br_ext = values0['b_data'][:, 0][data_mask]
    Bt_ext = values0['b_data'][:, 1][data_mask]
    Bn_ext = values0['b_data'][:, 2][data_mask]
    
    # 构建尺度-时间图
    all_results = {}
    scaleogram_r12 = []  # 类似小波尺度图
    scaleogram_r23 = []
    scaleogram_quality = []  # 质量指标
    
    # 统一的时间轴（最细尺度的时间分辨率）
    finest_step = min(window_sizes) * (1 - overlap) / 86400.0
    unified_time = np.arange(start, end + finest_step, finest_step)
    
    for idx, window_sec in enumerate(window_sizes):
        window_days = window_sec / 86400.0
        step_days = window_days * (1 - overlap)
        
        # 该尺度下的时间中心点
        n_intervals = max(int(np.ceil((end - start) / step_days)), 1)
        centers = np.linspace(start, end, n_intervals + 1)
        
        r12_list, r23_list, quality_list = [], [], []
        
        for center in centers:
            ws = center - window_days/2
            we = center + window_days/2
            
            mask = (time_ext >= ws) & (time_ext < we)
            if np.sum(mask) < 3:
                r12_list.append(np.nan)
                r23_list.append(np.nan)
                quality_list.append(0)
                continue
            
            # MVA计算
            Brs, Bts, Bns = Br_ext[mask], Bt_ext[mask], Bn_ext[mask]
            Br0, Bt0, Bn0 = np.mean(Brs), np.mean(Bts), np.mean(Bns)
            dB = np.column_stack((Brs-Br0, Bts-Bt0, Bns-Bn0))
            M = (dB.T @ dB) / len(dB)
            eigvals, eigvecs = np.linalg.eigh(M)
            eigvals = np.sort(eigvals)[::-1]
            
            eps = 1e-12
            r12 = eigvals[0] / max(eigvals[1], eps)
            r23 = eigvals[1] / max(eigvals[2], eps)
            
            # 质量评分（类似小波系数强度）
            quality = 1.0 if (r12 > r12_threshold and r23 > r23_threshold) else \
                     0.5 if (r12 > r12_threshold/2 and r23 > r23_threshold/2) else 0.1
            
            r12_list.append(r12)
            r23_list.append(r23)
            quality_list.append(quality)
        
        # 插值到统一时间轴（便于绘制尺度-时间图）
        times_jd = np.array(centers)
        r12_arr = np.array(r12_list)
        r23_arr = np.array(r23_list)
        
        # 线性插值到统一时间网格
        # 处理NaN值
        r12_clean = np.where(np.isnan(r12_arr), np.nanmedian(r12_arr), r12_arr)
        r23_clean = np.where(np.isnan(r23_arr), np.nanmedian(r23_arr), r23_arr)
        
        r12_interp = np.interp(unified_time, times_jd, r12_clean)
        r23_interp = np.interp(unified_time, times_jd, r23_clean)
        
        all_results[window_sec] = {
            'centers_jd': times_jd,
            'centers_dt': [jd_to_datetime(t) for t in times_jd],
            'r12': r12_arr,
            'r23': r23_arr,
            'quality': np.array(quality_list),
            'eigenvectors': []  # 可存储特征向量用于方向分析
        }
        
        scaleogram_r12.append(r12_interp)
        scaleogram_r23.append(r23_interp)
        scaleogram_quality.append(np.interp(unified_time, times_jd, quality_list))
    
    return all_results, np.array(scaleogram_r12), np.array(scaleogram_r23), \
           np.array(scaleogram_quality), unified_time, window_sizes

def plot_mva_scalogram(values0, start, end, overlap=0.75):
    """
    绘制类似小波尺度图的MVA分析结果
    使用32个从10秒到300秒的线性尺度
    """
    import matplotlib.dates as mdates
    
    # 使用32个线性尺度，从10秒到300秒
    results, r12_scalo, r23_scalo, quality_scalo, time_jd, scales = \
        mva_wavelet_scales(values0, start, end, scales='linear', 
                          n_scales=32, overlap=overlap)
    
    time_dt = [jd_to_datetime(t) for t in time_jd]
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    # === 子图1: r12 的尺度-时间图 ===
    ax1 = fig.add_subplot(gs[0, :])
    
    # 使用对数尺度
    X, Y = np.meshgrid(mdates.date2num(time_dt), np.arange(len(scales)))
    
    # 绘制热力图 - r12
    im1 = ax1.pcolormesh(X, Y, np.log10(r12_scalo), 
                        shading='gouraud', cmap='hot', vmin=0, vmax=3)
    
    # 设置y轴刻度 - 显示部分刻度以避免拥挤
    step = max(1, len(scales) // 8)
    yticks = range(0, len(scales), step)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([f'{scales[i]}s\n({scales[i]/60:.1f}min)' for i in yticks])
    ax1.set_ylabel('尺度 (窗口大小)', fontsize=11)
    ax1.set_title('MVA r12 尺度-时间分布 (log10)', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='log10(r12)')
    
    # 添加几条参考尺度线
    ref_scales = [30, 60, 120, 240]  # 参考尺度（秒）
    for ref_s in ref_scales:
        scale_idx = np.argmin(np.abs(np.array(scales) - ref_s))
        ax1.axhline(y=scale_idx, color='cyan', linestyle='--', alpha=0.5, 
                   label=f'{ref_s}s' if ref_s == ref_scales[0] else '')
    if ref_scales:
        ax1.legend(loc='upper right', fontsize=9)
    
    # === 子图2: r23 的尺度-时间图 ===
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    im2 = ax2.pcolormesh(X, Y, np.log10(r23_scalo), 
                        shading='gouraud', cmap='viridis', vmin=0, vmax=2)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f'{scales[i]}s' for i in yticks])
    ax2.set_ylabel('尺度 (窗口大小)', fontsize=11)
    ax2.set_title('MVA r23 尺度-时间分布 (log10)', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='log10(r23)')
    
    # === 子图3: 质量因子 ===
    ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
    im3 = ax3.pcolormesh(X, Y, quality_scalo, 
                        shading='gouraud', cmap='RdYlGn', vmin=0, vmax=1)
    ax3.set_yticks(yticks)
    ax3.set_yticklabels([f'{scales[i]}s' for i in yticks])
    ax3.set_ylabel('尺度 (窗口大小)', fontsize=11)
    ax3.set_xlabel('时间', fontsize=12)
    ax3.set_title('MVA 质量因子 (1=高质量, 0.5=中等, 0.1=低质量)', 
                 fontsize=13, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='质量')
    
    # 标记高质量区域
    high_quality_mask = quality_scalo >= 1.0
    if np.any(high_quality_mask):
        # 找到连通区域
        from scipy import ndimage
        labeled, num_features = ndimage.label(high_quality_mask)
        ax3.contour(X, Y, labeled, levels=range(1, num_features+1), 
                   colors='white', linewidths=1.5, alpha=0.7)
    
    # 格式化x轴
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax3.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    plt.setp(ax1.xaxis.get_majorticklabels(), visible=False)
    plt.setp(ax2.xaxis.get_majorticklabels(), visible=False)
    
    # 旋转x轴标签避免重叠
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig, results, scales  # 返回scales以便后续使用

def detect_mva_across_scales(values0, start, end, 
                            r12_threshold=10, r23_threshold=2,
                            min_scale=10, max_scale=300,
                            scale_ratio=1.2):
    """
    跨尺度事件检测
    """
    import matplotlib.dates as mdates
    
    # 生成几何级数尺度（但限制在10-300秒范围内）
    scales = []
    s = min_scale
    while s <= max_scale:
        scales.append(int(s))
        s *= scale_ratio
    
    print(f"检测尺度: {scales}")
    
    # 使用 custom_scales 参数
    results, r12_scalo, r23_scalo, quality_scalo, time_jd, _ = \
        mva_wavelet_scales(values0, start, end, 
                          scales='custom', 
                          custom_scales=scales,
                          overlap=0.75,
                          r12_threshold=r12_threshold,
                          r23_threshold=r23_threshold)
    
    n_scales, n_times = r12_scalo.shape
    
    # === 跨尺度一致性检测 ===
    # 1. 在每个时间点上，检查哪些尺度满足阈值
    r12_mask = r12_scalo > r12_threshold
    r23_mask = r23_scalo > r23_threshold
    combined_mask = r12_mask & r23_mask
    
    # 2. 寻找"尺度锥"
    persistent_events = []
    
    for t_idx in range(n_times):
        # 检查该时间点在哪些尺度上是高质量的
        quality_at_t = combined_mask[:, t_idx]
        if not np.any(quality_at_t):
            continue
            
        # 找到连续的尺度范围
        scale_indices = np.where(quality_at_t)[0]
        if len(scale_indices) < 2:
            continue  # 需要至少2个尺度支持
        
        # 检查是否是连续的尺度
        if np.max(np.diff(scale_indices)) > 2:
            continue  # 尺度跳跃太大，可能不是同一结构
        
        # 计算该时间点的尺度指数
        valid_scales = [scales[i] for i in scale_indices]
        valid_r12 = [r12_scalo[i, t_idx] for i in scale_indices]
        
        if len(valid_scales) >= 2:
            # 拟合幂律：log(r12) = alpha * log(scale) + C
            log_s = np.log(valid_scales)
            log_r = np.log(valid_r12)
            alpha = np.polyfit(log_s, log_r, 1)[0]
            
            persistent_events.append({
                'time_jd': time_jd[t_idx],
                'time_dt': jd_to_datetime(time_jd[t_idx]),
                'scales': valid_scales,
                'r12_values': valid_r12,
                'scale_exponent': alpha,
                'max_r12': np.max(valid_r12),
                'mean_r23': np.mean([r23_scalo[i, t_idx] for i in scale_indices])
            })
    
    # === 聚类相邻时间点的事件 ===
    if len(persistent_events) == 0:
        return []
    
    # 按时间排序
    persistent_events.sort(key=lambda x: x['time_jd'])
    
    # 合并时间相邻的事件
    merged_events = []
    current_cluster = [persistent_events[0]]
    min_gap = (min_scale / 2) / 86400.0  # 转换为JD
    
    for event in persistent_events[1:]:
        if event['time_jd'] - current_cluster[-1]['time_jd'] < min_gap:
            current_cluster.append(event)
        else:
            # 处理当前簇
            merged_events.append(merge_event_cluster(current_cluster))
            current_cluster = [event]
    
    if current_cluster:
        merged_events.append(merge_event_cluster(current_cluster))
    
    return merged_events

def merge_event_cluster(cluster):
    """合并时间簇内的多个时间点检测"""
    return {
        'start_jd': cluster[0]['time_jd'],
        'end_jd': cluster[-1]['time_jd'],
        'center_jd': np.mean([e['time_jd'] for e in cluster]),
        'duration_days': cluster[-1]['time_jd'] - cluster[0]['time_jd'],
        'n_points': len(cluster),
        'best_scale_exponent': np.mean([e['scale_exponent'] for e in cluster]),
        'max_r12': max([e['max_r12'] for e in cluster]),
        'scales_covered': list(set([s for e in cluster for s in e['scales']])),
        'start_dt': cluster[0]['time_dt'],
        'end_dt': cluster[-1]['time_dt']
    }

def comprehensive_wavelet_mva_analysis(values0, start, end):
    """
    综合多尺度MVA分析 - 结合小波思想与MVA
    使用32个从10秒到300秒的线性尺度
    """
    import matplotlib.dates as mdates
    from scipy import ndimage
    
    print("开始多尺度MVA分析...")
    print(f"时间范围: {jd_to_datetime(start)} 到 {jd_to_datetime(end)}")
    print("使用32个线性尺度，从10秒到300秒")
    
    # 1. 生成尺度-时间图
    fig_scalo, results, scales = plot_mva_scalogram(values0, start, end, overlap=0.75)
    
    # 2. 跨尺度事件检测
    events = detect_mva_across_scales(values0, start, end,
                                     min_scale=10, max_scale=300)
    
    print(f"\n{'='*60}")
    print(f"跨尺度MVA分析完成")
    print(f"发现 {len(events)} 个跨尺度持续事件")
    print(f"{'='*60}")
    
    for i, evt in enumerate(events[:10]):  # 显示前10个
        duration_sec = evt['duration_days'] * 86400
        print(f"\n事件 {i+1}:")
        print(f"  时间: {evt['start_dt'].strftime('%H:%M:%S')} - {evt['end_dt'].strftime('%H:%M:%S')}")
        print(f"  持续时间: {duration_sec:.1f} 秒")
        print(f"  覆盖尺度: {sorted(evt['scales_covered'])}")
        print(f"  尺度指数: {evt['best_scale_exponent']:.2f}")
        print(f"  最大r12: {evt['max_r12']:.1f}")
    
    # 3. 绘制事件标记在尺度图上
    if events:
        # 计算每个事件的中心时间（数值格式）
        center_times = []
        y_positions = []
        
        for evt in events:
            # 计算中心时间
            center = evt['start_dt'] + (evt['end_dt'] - evt['start_dt'])/2
            center_times.append(mdates.date2num(center))
            # 根据覆盖的尺度数量确定y轴位置
            # 找到这些尺度对应的索引位置的平均值
            scale_indices = [np.argmin(np.abs(np.array(scales) - s)) for s in evt['scales_covered']]
            y_positions.append(np.mean(scale_indices))
        
        fig_scalo.axes[2].scatter(
            center_times,
            y_positions,
            c='blue', s=150, marker='*', edgecolors='white', linewidths=2,
            label='跨尺度事件', zorder=10
        )
        fig_scalo.axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    return fig_scalo, results, events

comprehensive_wavelet_mva_analysis(data, front, rear)

# 使用示例：
"""
# 假设你的数据格式如下：
data = {
    'time_data': np.array([...]),  # 儒略日时间数组
    'b_data': np.array([...])      # 形状为 (n, 3) 的磁场数据 [Br, Bt, Bn]
}

# 设置分析时间范围
start_time = data['time_data'][0]  # 起始时间
end_time = data['time_data'][-1]   # 结束时间

# 运行综合分析
fig, results, events = comprehensive_wavelet_mva_analysis(data, start_time, end_time)
plt.show()
"""