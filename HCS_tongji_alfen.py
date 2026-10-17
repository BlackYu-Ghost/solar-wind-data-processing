"""
使用说明：
1、数据提取时注意更改文件地址名和文件名
2、数据进行窗口有效性判断时注意函数变量数据分辨率修改，有效才会进行后续操作
3、所有涉及计算的时间都是儒略时，转换成秒要乘以86400
4、毛刺去除使用突变斜率角度判断，用质子密度异常突变来检测
5、所用的小波变换是复数小波变换，尺度使用线性分段
6、
"""
# ============================================================================
#数据预处理文件
# ============================================================================
from matplotlib.ticker import ScalarFormatter, FixedLocator, LogLocator, LogFormatterMathtext
from astropy.timeseries import LombScargle
from datetime import datetime, timedelta 
from scipy.signal import fftconvolve
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from astropy.time import Time
from spacepy.pycdf import CDF
from scipy import ndimage
from scipy import linalg
from scipy import stats
from tqdm import tqdm
import numpy.ma as ma
import numpy as np
import fnmatch
import pathlib
import julian
import pickle
import pywt
import os
plt.rcParams['font.family'] = ['DejaVu Sans','SimHei'] 
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.unicode_minus'] = False
BadVal = -1.e+31
# ============================================================================
#磁场数据函数
# ============================================================================
def mag_data_done(t_start: str, t_end: str,
                            src_dir: str = r"E:/science/tongji/",
                            save_pkl: bool = False):

    fmt_day = "%Y-%m-%d" #定义格式日期
    start = datetime.strptime(t_start, fmt_day) #将字符串转化成datetime对象
    end   = datetime.strptime(t_end, fmt_day) #将字符串转化成datetime对象
    days  = (end - start).days + 1 #包含结束日期
    time_slots = [start + timedelta(days=d) for d in range(days)] #生成日期列表

    time_list, bx_list, by_list, bz_list = [], [], [], [] #初始化空列表储存目的信息
    total_days = len(time_slots) #计算总天数
    day_with_data = 0 #天数计数器

    for slot in tqdm(time_slots, desc="处理", unit="天"): #使用tqdm创建进度循环条
        yyyymmdd = slot.strftime("%Y%m%d") #将日期转换成特定形式字符串
        pattern  = f"psp_fld_l2_mag_rtn_4_sa_per_cyc_{yyyymmdd}_v*.cdf" #构建文件匹配模式
        files    = sorted(fnmatch.filter(os.listdir(src_dir), pattern)) #查找匹配的文件
        if not files:
            continue #未发现目标文件，进入下一天
        day_with_data += 1  #发现目标文件，天数计数器增一
        for fname in files:
            cdf_path = os.path.join(src_dir, fname) #合成文件路径
            with CDF(cdf_path) as Data:

                t_cdf = Data['epoch_mag_RTN_4_Sa_per_Cyc'][:] #读取时间数据
                jd    = np.array([julian.to_jd(ts, fmt='jd') for ts in t_cdf],
                                 dtype='float64') #将时间转换成儒略时

                #epoch_qf = Data['epoch_quality_flags'][:]       
                #quality_values = Data['psp_fld_l2_quality_flags'][:]  
                
                #epoch_qf_jd = np.array([julian.to_jd(ts, fmt='jd') 
                #       for ts in epoch_qf], dtype='float64')
                
                good = np.zeros(len(jd), dtype = int)
                '''
                for i in range(len(epoch_qf_jd) - 1):
                    flag = (jd >= epoch_qf_jd[i]) & (jd < epoch_qf_jd[i+1])
                    good[flag] = quality_values[i]
                    
                if len(epoch_qf_jd) > 0:
                    last_flag = jd >= epoch_qf_jd[-1]
                    good[last_flag] = quality_values[-1]
                '''
                good_flag = good == 0
                
                if good_flag.sum() == 0:
                    continue
                time = jd[good_flag]
                Brtn = Data['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'][:][good_flag] #读磁场数据   
                
                mask = (Brtn[:, 0] != BadVal) & (Brtn[:, 1] != BadVal)\
                    & (Brtn[:, 2] != BadVal)
                if mask.sum() == 0:
                    continue

                time_list.append(time[mask])
                bx_list.append(Brtn[:, 0][mask])
                by_list.append(Brtn[:, 1][mask])
                bz_list.append(Brtn[:, 2][mask])

    if not time_list:
        print("⚠  该时段无数据，返回空字典")
        return {"time_B": np.empty(0), "B3_rtn": np.empty((0, 3))}

    time_B = np.concatenate(time_list)
    B3_rtn = np.column_stack((np.concatenate(bx_list),
                               np.concatenate(by_list),
                               np.concatenate(bz_list))) #合并磁场数据并重新组合成（N,3）格式

    data = {"time_B": time_B, "B3_rtn": B3_rtn} #生成字典
    print(f"共 {total_days} 个日历天，其中 {day_with_data} 天有数据 "
          f"（占比 {day_with_data/total_days*100:.1f}%）")

    return data
# ============================================================================
#等离子体数据读取函数
# ============================================================================
def plasma_data_done(t_start: str, t_end: str,
                          src_dir: str = r"E:/science/tongji/",
                          save_pkl: bool = True):
    fmt_day = "%Y-%m-%d"
    start = datetime.strptime(t_start, fmt_day)
    end   = datetime.strptime(t_end, fmt_day)
    days = (end - start).days + 1
    time_slots = [start + timedelta(days=d) for d in range(days)]
    total_days = len(time_slots)
    day_with_data = 0
    
    time_p, nproton, tp = [], [], []
    v_r, v_t, v_n = [], [], []
  
    for slot in tqdm(time_slots, desc="L3i 处理进度", unit="天"):
        yyyymmdd = slot.strftime("%Y%m%d")
        pattern = f"psp_swp_spi_sf00_l3_mom_{yyyymmdd}_v*.cdf"
        files = sorted(fnmatch.filter(os.listdir(src_dir), pattern))
        if not files:
            continue
        day_with_data += 1
        for fname in files:
            cdf_path = os.path.join(src_dir, fname)
            with CDF(cdf_path) as Data:
                epoch = Data['Epoch'][:]
                jd = np.array([julian.to_jd(ts, fmt='jd') for ts in epoch],
                              dtype='float64')
                vp_rtn = Data['VEL_RTN_SUN'][:]
                nep    = Data['DENS'][:]
                tep    = Data['TEMP'][:]

                mask = (nep != BadVal) & (tep != BadVal) & \
                       (vp_rtn[:, 0] != BadVal) & (vp_rtn[:, 1] != BadVal) & (vp_rtn[:, 2] != BadVal)
                if mask.sum() == 0:
                    continue
                time_p.append(jd[mask])
                nproton.append(nep[mask])
                tp.append(tep[mask])
                v_r.append(vp_rtn[mask, 0])
                v_t.append(vp_rtn[mask, 1])
                v_n.append(vp_rtn[mask, 2])

                # 在处理每个文件时添加
                time_diff = np.diff(jd[mask])
                if len(time_diff) > 0:
                    print(f"文件 {fname}:")
                    print(f"  数据点数: {len(jd[mask])}")
                    print(f"  时间范围: {jd[mask][0]:.6f} 到 {jd[mask][-1]:.6f}")
                    print("  时间间隔统计:")
                    print(f"    最小: {time_diff.min()*86400:.3f} 秒")  # 转换为秒
                    print(f"    最大: {time_diff.max()*86400:.3f} 秒")
                    print(f"    平均: {time_diff.mean()*86400:.3f} 秒")
 
    if not time_p:
        print("⚠  该时段无有效数据，返回空字典")
        return {
            "time_p": np.empty(0),
            "v3_rtn": np.empty((0, 3)),
            "nproton": np.empty(0),
            "tp": np.empty(0)
        }
    time_p = np.concatenate(time_p)
    v3_rtn = np.column_stack((np.concatenate(v_r),
                              np.concatenate(v_t),
                              np.concatenate(v_n)))
    data = {
        "time_p": time_p,
        "v3_rtn": v3_rtn,
        "nproton": np.concatenate(nproton),
        "tp": np.concatenate(tp)
    }

    print(f"共 {total_days} 个日历天，其中 {day_with_data} 天有数据 "
             f"（占比 {day_with_data/total_days*100:.1f}%）")
    
    return data
# ============================================================================
#超热电子数据函数
# ============================================================================
def strahl_data_done(t_start: str, t_end: str, target_eV = 314,
                            src_dir: str = r"E:/science/tongji/",
                            save_pkl: bool = False):

    fmt_day = "%Y-%m-%d" #定义格式日期
    start = datetime.strptime(t_start, fmt_day) #将字符串转化成datetime对象
    end   = datetime.strptime(t_end, fmt_day) #将字符串转化成datetime对象
    days  = (end - start).days + 1 #包含结束日期
    time_slots = [start + timedelta(days=d) for d in range(days)] #生成日期列表

    time_list, strahl3_list, info_list = [], [], []#初始化空列表储存目的信息
    total_days = len(time_slots) #计算总天数
    day_with_data = 0 #天数计数器

    for slot in tqdm(time_slots, desc="处理", unit="天"): #使用tqdm创建进度循环条
        yyyymmdd = slot.strftime("%Y%m%d") #将日期转换成特定形式字符串
        pattern  = f"psp_swp_spe_sf0_l3_pad_{yyyymmdd}_v*.cdf" #构建文件匹配模式
        files    = sorted(fnmatch.filter(os.listdir(src_dir), pattern)) #查找匹配的文件
        if not files:
            continue #未发现目标文件，进入下一天
        day_with_data += 1  #发现目标文件，天数计数器增一
        for fname in files:
            cdf_path = os.path.join(src_dir, fname) #合成文件路径
            with CDF(cdf_path) as Data:

                t_cdf = Data['Epoch'][:] #读取时间数据
                jd    = np.array([julian.to_jd(ts, fmt='jd') for ts in t_cdf],
                                 dtype='float64') #将时间转换成儒略时
                energy_vals = Data['ENERGY_VALS'][:] #读磁场数据   
                eflux_vs_pa_e = Data['EFLUX_VS_PA_E'][:]  # 电子通量 vs 俯仰角和能量 (3090, 12, 32)
                pitchangle = Data['PITCHANGLE'][:]  # 俯仰角 (3090, 12)

                idx0 = int(np.argmin(np.abs(energy_vals - target_eV)))
                idx3 = np.clip([idx0-1, idx0, idx0 + 1], 0, energy_vals.size-1)                
                strahl3ch = eflux_vs_pa_e[:, :, idx3]           
               
                time_list.append(jd)
                strahl3_list.append(strahl3ch)
                info_list.append(energy_vals[0, idx3])
                if slot == time_slots[0] and fname == files[0]:
                    print("-----全部能量通道/角度中心-----")
                    for i, ev in enumerate(energy_vals[0]):
                        print(f"  通道 {i:2d}: {ev:10.2f} eV")
                    print('-----------------------------')
                    print(np.array2string(pitchangle[0],separator=','))
                    print('-----------------------------')
                    print(f'目标能量{target_eV}eV -> 选中的通道下标/能量：'
                          f'{int(idx3[0])}/{float(energy_vals[0, idx3[0]]):.2f}eV,'
                          f'{int(idx3[1])}/{float(energy_vals[0, idx3[1]]):.2f}eV,'
                          f'{int(idx3[2])}/{float(energy_vals[0, idx3[2]]):.2f}eV,')
                    print('-----------------------------')
    
    if not time_list:
        print("⚠  该时段无数据，返回空字典")
        return {"time_s": np.empty(0), "strahl3": np.empty((0, 3)), "energy_name": np.empty(0)}

    time_s = np.concatenate(time_list)
    strahl_target = np.concatenate(strahl3_list, axis=0) 
    energy_name = np.concatenate(info_list)
    data = {"time_s": time_s, "strahl3": strahl_target, "energy_name": energy_name} #生成字典
    print(f"共 {total_days} 个日历天，其中 {day_with_data} 天有数据 "
          f"（占比 {day_with_data/total_days*100:.1f}%）")

    return data
# ============================================================================
#数据文件保存函数
# ============================================================================
def pkl_save(data,dst_dir: str = r"E:/science/tongji/",save_pkl = True):
    if save_pkl:
        pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)
        out_file = os.path.join(dst_dir,
                                "data_yuchuli.pkl")
        with open(out_file, "wb") as f:
            pickle.dump(data, f)
        print(f"\n✅ 已保存  →  {out_file}")     
# ============================================================================
#数据处理函数-数据窗口有效性判断
# ============================================================================
def data_judge(data,window_start, window_end, 
               revolution_B = 0.2185, revolution_p = 3.5, fanwei=1):    
    yvliu = (window_end - window_start) / fanwei
    window_start-=yvliu
    window_end  +=yvliu
    window_start = max(window_start, data['time_B'][0])  
    window_end = min(window_end, data['time_B'][-1])
    mag_start_idx = np.searchsorted(data['time_B'], window_start)
    mag_end_idx = np.searchsorted(data['time_B'], window_end, side='right')
    plasma_start_idx = np.searchsorted(data['time_p'], window_start)  
    plasma_end_idx = np.searchsorted(data['time_p'], window_end, side='right')
    plasma_mask = (data['time_p'] >= window_start) & (data['time_p'] <= window_end)
    
    mag_count = (mag_end_idx - mag_start_idx)
    plasma_count = (plasma_end_idx - plasma_start_idx)
    window_size = (window_end - window_start) * 86400
    expected_mag = int(window_size / revolution_B * 0.9)
    expected_plasma = int(window_size / revolution_p * 0.9)
    if mag_count < expected_mag:
        return False, plasma_mask, print("磁场数据点数不足")
    if plasma_count < expected_plasma:
        return False, plasma_mask, print("等离子体数据点数不足")
    
    max_gap_ratio = 0.02  
    max_allowed_gap = window_size * max_gap_ratio
    mag_window_ts = data['time_B'][mag_start_idx:mag_end_idx]
    if len(mag_window_ts) > 1:
        mag_time_diffs = np.diff(mag_window_ts) * 86400
        max_mag_gap = np.max(mag_time_diffs) if len(mag_time_diffs) > 0 else 0
        if max_mag_gap > max_allowed_gap:
            return False, plasma_mask, f"磁场数据时间间隔过大: {max_mag_gap:.1f}秒 > {max_allowed_gap}秒"
    
    plasma_window_ts = data['time_p'][plasma_start_idx:plasma_end_idx]
    if len(plasma_window_ts) > 1:
        plasma_time_diffs = np.diff(plasma_window_ts) * 86400
        max_plasma_gap = np.max(plasma_time_diffs) if len(plasma_time_diffs) > 0 else 0
        if max_plasma_gap > max_allowed_gap:
            return False, plasma_mask, f"等离子体数据时间间隔过大: {max_plasma_gap:.1f}秒 > {max_allowed_gap}秒"
    return True, plasma_mask, print("数据有效")    
# ============================================================================
#数据插值函数
# ============================================================================
def insert_data(data, start_ts, end_ts, resolution=3.5, fanwei=1):
    yvliu = max((end_ts - start_ts) / fanwei, 1/24)
    start_ts -= yvliu
    end_ts += yvliu
    start_ts = max(start_ts, data['time_B'][0], data['time_p'][0])
    end_ts = min(end_ts, data['time_B'][-1], data['time_p'][-1])
    num_points = int((end_ts - start_ts) * 86400 / resolution) + 1
    resample_time_data = start_ts + np.arange(num_points) * resolution / 86400 
    print("数据插值:")
    print(f"  原始磁场数据: {len(data['time_B'])} 点")
    print(f"  原始等离子体数据: {len(data['time_p'])} 点")
    print(f"  插值后数据: {num_points} 点 (分辨率: {resolution:.3f} 秒)")
    def clean_and_interp(target_time, source_time, source_data):
        valid = ~np.isnan(source_data)
        if valid.sum() < 2:
            print(f"警告: 仅 {valid.sum()} 个有效点，插值可能不可靠")
            return np.zeros(len(target_time))
        return np.interp(target_time, source_time[valid], source_data[valid])
    b_resampled = np.zeros((len(resample_time_data), 3))
    for i in range(3):
        b_resampled[:, i] = clean_and_interp(
            resample_time_data, data['time_B'], data['B3_rtn'][:, i])
    v_resampled = np.zeros((len(resample_time_data), 3))    
    for i in range(3):
        v_resampled[:, i] = clean_and_interp(
            resample_time_data, data['time_p'], data['v3_rtn'][:, i])
    nep_resampled = clean_and_interp(
        resample_time_data, data['time_p'], data['nproton'])
    tp_resampled = clean_and_interp(
        resample_time_data, data['time_p'], data['tp'])
    data['time_data'] = resample_time_data
    data['b_data'] = b_resampled
    data['v_data'] = v_resampled
    data['nep_data'] = nep_resampled
    data['tp_data'] = tp_resampled
    data['resolution'] = resolution
    return data
# ============================================================================
#图像绘制
# ============================================================================
def plot5figures(time, Br, Bt, Bn, vr, vt, vn, 
                 sigma_m, sigma_c, sigma_r, 
                 scales, periods,
                 icme_start=None, icme_end=None,
                 sfr_start=None, sfr_end=None,
                 title='Figure 2 Style Plot'):
    """绘制磁场和等离子体速度波动以及三σ小波变换"""
    time = Time(time, format='jd').datetime
    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.2, 1.2, 1.5, 1.5, 1.5], 
                          hspace=0.1, left=0.07, right=0.91, top=0.95, bottom=0.07)

    #-----------------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time, Br, 'r-', linewidth=1.2, label='$B_R$')
    ax1.plot(time, Bt, 'g-', linewidth=1.2, label='$B_T$')
    ax1.plot(time, Bn, 'b-', linewidth=1.2, label='$B_N$')
    ax1.set_ylabel('$B$ [nT]', fontsize=12)
    ax1.legend(loc='upper right', ncol=3, fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)
    #-----------------------------------
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(time, vr, 'r-', linewidth=1.2, label='$V_R$')
    ax2.plot(time, vt, 'g-', linewidth=1.2, label='$V_T$')
    ax2.plot(time, vn, 'b-', linewidth=1.2, label='$V_N$')
    ax2.set_ylabel('$V$ [km/s]', fontsize=12)
    ax2.legend(loc='upper right', ncol=3, fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelbottom=False)
    #-----------------------------------   
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_yscale('log')
    ax3.set_ylim(periods[-1], periods[0])
    ax3.set_ylabel('scale(min)',fontsize = 12)
    ax3.set_title('$\sigma_m$',fontsize = 14, pad = 1)

    ax3.yaxis.set_major_locator(LogLocator(base=2, numticks=10))
    ax3.yaxis.set_major_formatter(LogFormatterMathtext(base=2))
    ax3.tick_params(labelbottom=False)
    
    im1 = ax3.imshow(sigma_m, aspect = 'auto', cmap = 'jet', 
                    extent = [time[0], time[-1], periods[-1], periods[0]],
                    vmin = -1, vmax = 1, origin='upper')

    cbar_ax = fig.add_axes([0.92, 0.47, 0.01, 0.15]) 
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('$\sigma_m$', fontsize=12)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    
    '''
    ax3.contour(time, periods, np.abs(sigma_m), levels=[0.7],
                colors='black', linewidths=1.2, linestyles='-')
    
    ax3.contour(time, periods, np.abs(sigma_m), 
                levels=[-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9],
                colors='white', linewidths=0.2,  linestyles='-')
    '''
    #-----------------------------------
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    im2 = ax4.imshow(sigma_c, aspect = 'auto', cmap = 'jet', 
                    extent = [time[0], time[-1], periods[-1], periods[0]],
                    vmin = -1, vmax = 1, origin='upper')
    
    cbar_ax = fig.add_axes([0.92, 0.27, 0.01, 0.15]) 
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('$\sigma_c$', fontsize=12)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    ax4.set_ylabel('scale(min)',fontsize = 12)
    ax4.set_title('$\sigma_c$',fontsize = 14, pad = 1)

    ax4.set_yscale('log')
    ax4.set_ylim(periods[-1], periods[0])
    ax4.yaxis.set_major_locator(LogLocator(base=2, numticks=10))
    ax4.yaxis.set_major_formatter(LogFormatterMathtext(base=2))
    ax4.tick_params(labelbottom=False)
    #-----------------------------------
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    im3 = ax5.imshow(sigma_r, aspect = 'auto', cmap = 'jet', 
                    extent = [time[0], time[-1], periods[-1], periods[0]],
                    vmin = -1, vmax = 1, origin='upper')
    
    cbar_ax = fig.add_axes([0.92, 0.08, 0.01, 0.15]) 
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label('$\sigma_r$', fontsize=12)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    ax5.set_ylabel('scale(min)',fontsize = 12)
    ax5.set_title('$\sigma_r$',fontsize = 14, pad = 1)

    ax5.set_yscale('log')
    ax5.set_ylim(periods[-1], periods[0])
    ax5.yaxis.set_major_locator(LogLocator(base=2, numticks=10))
    ax5.yaxis.set_major_formatter(LogFormatterMathtext(base=2))
# ============================================================================
#数据插值函数
# ============================================================================
def calculate_num(data):
    Br, Bt, Bn = data['b_data'][:, 0], data['b_data'][:, 1], data['b_data'][:, 2]
    vr, vt, vn = data['v_data'][:, 0], data['v_data'][:, 1], data['v_data'][:, 2]
    time, nep, tp = data['time_data'], data['nep_data'], data['tp_data']
    
    B_total, v_total = np.linalg.norm(data['b_data'], axis=1),\
                       np.linalg.norm(data['v_data'], axis=1)
    B_total0, v_total0 = np.mean(B_total), np.mean(v_total)
    
    Br0, Bt0, Bn0 = np.mean(Br), np.mean(Bt), np.mean(Bn)
    δBr, δBt, δBn = Br - Br0, Bt - Bt0, Bn - Bn0
    vr0, vt0, vn0 = np.mean(vr), np.mean(vt), np.mean(vn)
    δvr, δvt, δvn = vr - vr0, vt - vt0, vn - vn0
    
    Ca = 1e-9 * 1e-3 / np.sqrt(4 * np.pi * 1e-7 * 1.6726e-27 *\
                               np.mean(nep) * 1e+6)
    δBr2v, δBt2v, δBn2v = Ca * δBr, Ca * δBt, Ca * δBn
    zrplus, ztplus, znplus = δvr + δBr2v, δvt + δBt2v, δvn + δBn2v
    zrminus, ztminus, znminus = δvr - δBr2v, δvt - δBt2v, δvn - δBn2v
    
    #-----------------------------------小波变换
    wavelet = 'cmor'
    min_time, max_time = 30, 960
    f_c = pywt.scale2frequency(wavelet, 1)
    scales_min = min_time * 60 * f_c / data['resolution']
    if scales_min < 2 : scales_min = 2
    scales_max = max_time * 60 * f_c / data['resolution']
    scales = np.linspace(scales_min, scales_max, 32)
    periods = scales * data['resolution'] / f_c / 60
    
    WBr, _ = pywt.cwt(Br, scales, wavelet, data['resolution'])
    WBt, _ = pywt.cwt(Bt, scales, wavelet, data['resolution'])
    WBn, _ = pywt.cwt(Bn, scales, wavelet, data['resolution'])
    WB_total = (np.abs(WBr)**2 + np.abs(WBt)**2 + np.abs(WBn)**2)

    Wzrplus, _ = pywt.cwt(zrplus, scales, wavelet, data['resolution'])
    Wztplus, _ = pywt.cwt(ztplus, scales, wavelet, data['resolution'])
    Wznplus, _ = pywt.cwt(znplus, scales, wavelet, data['resolution'])
    Wzplus_total = (np.abs(Wzrplus)**2 + np.abs(Wztplus)**2 + np.abs(Wznplus)**2)
    
    Wzrminus, _ = pywt.cwt(zrminus, scales, wavelet, data['resolution'])
    Wztminus, _ = pywt.cwt(ztminus, scales, wavelet, data['resolution'])
    Wznminus, _ = pywt.cwt(znminus, scales, wavelet, data['resolution'])
    Wzminus_total = (np.abs(Wzrminus)**2 + np.abs(Wztminus)**2 + np.abs(Wznminus)**2)
    
    #-----------------------------------σ求解
    fenmu1 = np.where(WB_total > 1e-30, WB_total, 1e-30)
    fenmu2 = np.where((Wzplus_total + Wzminus_total)>1e-30, (Wzplus_total + Wzminus_total), 1e-30)
    fenmu3 = np.where((Wzplus_total + Wzminus_total)>1e-30, (Wzplus_total + Wzminus_total), 1e-30)
    σm, σc, σr = -2 * np.imag((np.conj(WBt) * WBn)) / fenmu1,\
                 (Wzplus_total - Wzminus_total) / fenmu2,\
                 2 * np.real((np.conj(Wzrplus) * Wzrminus + np.conj(Wztplus) * Wztminus\
                             +np.conj(Wznplus) * Wznminus)) / fenmu3 
    #-----------------------------------绘图
    plot5figures(time, Br, Bt, Bn, δvr, δvt, δvn, σm, σc, σr, scales, periods)
# ============================================================================
#原始数据绘制函数
# ============================================================================
def gap_mask(time, max_gap = 10 * 3.5 / 86400):
    if len(time) < 2:
        return ma.array(time)
    mask = np.zeros(len(time), dtype = bool)
    for i in range(1, len(time)):
        if (time[i] - time[i-1] > max_gap):
            mask[i - 1 : i + 1] = True
    return ma.array(time, mask = mask)
def draw3vectors(time0, values0, start, end, 
                 labels, ylabel, xlabel, title, colors=['k','r','g','b']):
    time_mask = (time0 >= start) & (time0 <= end)
    value = values0[time_mask]
    time = time0[time_mask]
    time_draw = gap_mask(time)
    
    time_dt = Time(time, format='jd').datetime
    fig, ax = plt.subplots(figsize=(12, 4))
    value_total = np.linalg.norm(value, axis = 1)
    value_total_draw = ma.array(value_total, mask = time_draw.mask) 
    ax.plot(time_dt, value_total_draw, '-', linewidth = 0.8, 
               color = colors[0], label = labels[0])
    for i in range(3):
        ax.plot(time_dt, ma.array(value[:, i], mask = time_draw.mask),
                '-', linewidth = 0.8, color = colors[i+1], label = labels[i+1])
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    start_str = julian.from_jd(start, fmt='jd').strftime('%Y-%m-%d %H:%M')
    end_str = julian.from_jd(end, fmt='jd').strftime('%H:%M')
    ax.set_title(f'{title}: {start_str} to {end_str}', 
                fontsize=12, fontweight='bold', pad=10)  
    plt.tight_layout()
# ============================================================================
#八图（基本量）
# ============================================================================
def draw_8yuchuli(values0, start, end):
    """绘制插值并去除等离子体毛刺的数据图（磁场、速度、密度、温度、
    目标超热电子时序图、超热电子颜色图，超热电子角度分布图，磁场幂律拟合）"""
    time_mask = (values0['time_data'] >= start) & (values0['time_data'] <= end)
    time = values0['time_data'][time_mask]
    time_dt = Time(time, format='jd').datetime
    Br, Bt, Bn = values0['b_data'][time_mask][:, 0], \
        values0['b_data'][:, 1][time_mask], values0['b_data'][:, 2][time_mask]
    Br0, Bt0, Bn0 = np.mean(Br), np.mean(Bt), np.mean(Bn)
    δBr, δBt, δBn = Br - Br0, Bt - Bt0, Bn - Bn0
    b_total = np.linalg.norm(values0['b_data'][time_mask], axis = 1)
    v_total = np.linalg.norm(values0['v_data'][time_mask], axis = 1)
    
    fig, axes = plt.subplots(6, 1, figsize = (8, 6))
    plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.15, left=0.1,right=0.5)
    #-----------------------------------
    colors, labels = ['k','r','g','b'], ['|B|', 'Br', 'Bt', 'Bn'] 
    axes[0].set_title(f'PSP Unified Analysis: {time_dt[0]} to {time_dt[-1]}', 
                fontsize=12, fontweight='bold',loc = 'center',  pad=10) 
    axes[0].plot(time_dt, b_total, '-', linewidth = 1.5, alpha = 1.0,
               color = colors[0], label = labels[0])
    for i in range(3):
        axes[0].plot(time_dt, values0['b_data'][:, i][time_mask], '-', linewidth = 1.5, 
                     alpha = 1.0, color = colors[i+1], label = labels[i+1])
    axes[0].set_ylabel('Magnetic Field\n[nT]', fontsize=8)
    axes[0].legend(loc='upper right', fontsize=6)
    #-----------------------------------
    labels = ['|V|', 'Vr', 'Vt', 'Vn'] 
    axes[1].plot(time_dt, v_total, '-', linewidth = 1.5, alpha = 1.0,
               color = colors[0], label = labels[0])
    for i in range(3):
        axes[1].plot(time_dt, values0['v_data'][:, i][time_mask], '-', linewidth = 1.5, 
                     alpha = 1.0, color = colors[i+1], label = labels[i+1])
    axes[1].set_ylabel('Velocity\n[km/s]', fontsize=8)
    axes[1].legend(loc='upper right', fontsize=6)
    
    axes[2].plot(time_dt, values0['nep_data'][time_mask], '-', linewidth = 1.5, alpha = 1.0,
               color = 'darkorange', label = 'Np')
    axes[2].set_ylabel('Density\n[cm$^{-3}$]', fontsize=8)
    axes[2].legend(loc='upper right', fontsize=6)
    #-----------------------------------
    axes[3].plot(time_dt, values0['tp_data'][time_mask], '-', linewidth = 1.5, alpha = 1.0,
               color = 'purple', label = 'Tem')
    axes[3].set_ylabel('Temperature\n[eV]', fontsize=8)
    axes[3].legend(loc='upper right', fontsize=6)
    #-----------------------------------
    label3 = ['l', 'o', 'r']
    time_mask = (values0['time_s'] >= start) & (values0['time_s'] <= end)
    time_num = Time(values0['time_s'][time_mask], format='jd').datetime
    strahl3_total = np.sum(values0['strahl3'][time_mask], axis = 1)
    for i in range(3):
        axes[4].plot(time_num, strahl3_total[:, i], 
                     '-', linewidth = 1.5, alpha = 1.0,
                     color = colors[i+1], label = label3[i])
    axes[4].set_ylabel('Strahl Flux\n(cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$)', fontsize=8)
    axes[4].set_yscale('log')
    axes[4].legend(loc='upper right', fontsize=6)
    #-----------------------------------
    '''
    energy_target = (values0['strahl3'][time_mask, :, 1]).T
    min_vals = np.nanmin(energy_target, axis=0) 
    max_vals = np.nanmax(energy_target, axis=0) 
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    energy_draw = (energy_target - min_vals) / ranges 
    angles = np.arange(7.5, 180, 15)
    #-----------------------------------
    im = axes[5].pcolormesh(time_num, angles, energy_draw,
                       cmap='jet', shading='auto',
                       vmin=0, vmax=1)
    axes[5].set_yticks(angles)
    axes[5].tick_params(axis='y', labelsize=6)
    
    pos = axes[5].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Energy Ratio', fontsize=8)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=6)
    '''
    times_mask = (values0['time_s'] >= start) & (values0['time_s'] <= end)
    time_num = Time(values0['time_s'][times_mask], format='jd').datetime
    energy_target = (values0['strahl3'][times_mask, :, 1]).T
    log_flux = np.log10(energy_target)
    vmax, vmin = np.nanmax(log_flux), np.nanmin(log_flux)
    angles = np.arange(7.5, 181, 15)
    #-----------------------------------
    im = axes[5].pcolormesh(time_num, angles, log_flux,
                       cmap='jet', shading='auto',
                       vmin=vmin, vmax=vmax)
    axes[5].set_yticks(angles)
    axes[5].tick_params(axis='y', labelsize=6)
    
    pos = axes[5].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('log$_{10}$(Flux)', fontsize=8)
    #cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=6)
    axes[5].set_ylabel('Angle Channel',fontsize = 8)
    axes[5].set_xlabel('Time')
    
    for ax in axes[:5]:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)
    axes[5].tick_params(labelbottom=True) 
    time_formatter = mdates.DateFormatter('%H:%M')
    
    for ax in axes[:5]:
        ax.set_xlim(time_dt[0], time_dt[-1])
        ax.xaxis.set_major_formatter(time_formatter)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    axes[5].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[5].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    axes[5].set_xlabel('Time (UTC)', fontsize=9)
    #-----------------------------------
    strahl3_mean = np.mean(values0['strahl3'][times_mask,:,1], axis=0)
    ax6 = fig.add_axes([0.62, 0.63, 0.25, 0.25]) 
    ax6.plot(angles, strahl3_mean, 'ko-', linewidth=2.5, 
            markersize=6, markerfacecolor='lightgray', markeredgecolor='black',
            markeredgewidth=1.5)
    ax6.yaxis.tick_right()
    ax6.yaxis.set_label_position("right")
    ax6.set_yscale('log')
    ax6.set_title('PSP SPE Averaged Pitch Angle\n Distribution of Strahl Electrons', fontsize=10)
    ax6.set_xlabel('Pitch Angle (deg)', fontsize=9)
    ax6.set_ylabel('Omnidirectional Flux\n(cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$)', fontsize=9)
    ax6.grid(ax6, alpha=0.3)
    info_text = f"Energy: {values0['energy_name'][1]:.0f} eV\n"
    info_text += f'F0 = {strahl3_mean[0]:.2e}\n'
    info_text += f'F180 = {strahl3_mean[-1]:.2e}\n'
    ratio = strahl3_mean[-1] / strahl3_mean[0]
    if not np.isnan(ratio):
        info_text += f'F180/F0 = {ratio:.2f}'
    
    ax6.text(0.02, 0.98, info_text, transform=ax6.transAxes, 
            verticalalignment='top', fontsize=6, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    #-----------------------------------
    δBr_fft, δBt_fft, δBn_fft = np.fft.rfft(δBr), np.fft.rfft(δBt), np.fft.rfft(δBn)#FFT
    
    Pr = np.abs(δBr_fft)**2 * 2 / len(Br) * values0['resolution']#分量功率谱
    Pt = np.abs(δBt_fft)**2 * 2 / len(Bt) * values0['resolution']
    Pn = np.abs(δBn_fft)**2 * 2 / len(Bn) * values0['resolution']
    P_total = Pr + Pt + Pn#磁场波动功率谱的迹
    freq = np.fft.rfftfreq(len(Br), d = values0['resolution'])
    
    smooth_points = 7 #(1/600=0.00167)*6=0.01说明在0.01Hz之前有5个点，超过三个可以进行7点平滑
    P_total_smooth = np.copy(P_total)
    half_points = smooth_points // 2
    for i in range(half_points, len(P_total) - half_points):#数据平滑处理
        P_total_smooth[i] = np.mean(P_total[i-half_points:i+half_points+1])
    for i in range(half_points):#左边界
        P_total_smooth[i] = np.mean(P_total[:i+half_points+1])
    for i in range(len(P_total) - half_points, len(P_total)):#右边界
        P_total_smooth[i] = np.mean(P_total[i-half_points:])   
        
    freq_range = [0.01, 0.1]#设定拟合范围
    mask = (freq >= freq_range[0]) & (freq <= freq_range[1])#选取拟定范围内的点
    freq_fit = freq[mask]
    P_fit = P_total_smooth[mask]
    valid_mask = (P_fit > 0) & np.isfinite(P_fit) & (freq_fit > 0)#去除负值和无穷大值
    freq_fit = freq_fit[valid_mask]
    P_fit = P_fit[valid_mask]
    
    if len(freq_fit) >= 2:#斜率、截距、相关系数、显著性水平、标准误差
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.log10(freq_fit), np.log10(P_fit))
    else:slope, intercept, r_value, std_err = np.nan, np.nan, np.nan, np.nan
    
    ax7 = fig.add_axes([0.62, 0.2, 0.25, 0.25]) 
    freq_limitation = 1.0 / (len(Br) * values0['resolution'])
    ax7.loglog(freq[freq>=freq_limitation], P_total_smooth[freq>=freq_limitation],
                 'k-', linewidth=1.5)  
    ax7.yaxis.tick_right()
    ax7.yaxis.set_label_position("right")
    ax7.set_title(' Power spectral density scaling analysis', fontsize=10)
    ax7.set_xlabel('Frequency [Hz]', fontsize=9)
    ax7.set_ylabel('Power Spectral Density \n[nT²/Hz]', fontsize=9)
    ax7.grid(True, alpha=0.3, which='both')
    if not np.isnan(slope):
        fit_freq = np.logspace(np.log10(freq_range[0]), 
                              np.log10(freq_range[1]), 100)
        fit_P = 10**(intercept + slope * np.log10(fit_freq))
        ax7.loglog(fit_freq, fit_P, 'r--', linewidth=2)
        ax7.axvline(freq_range[0], color='gray', linestyle=':', alpha=0.7)
        ax7.axvline(freq_range[1], color='gray', linestyle=':', alpha=0.7)
    if not np.isnan(slope):
        text_str = f' α={slope:.2f} ± {std_err:.2f}'
        ax7.text(0.05, 0.95, text_str, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax7.tick_params(axis='both', which='major', labelsize=12)
    ax7.tick_params(axis='both', which='minor', labelsize=10)
    plt.show()
# ============================================================================
#四图波动相关性分析
# ============================================================================
def calculate_aligned_ylim(left_data, right_data):
    """计算对齐的纵轴范围，保证零刻度对齐"""
    left_range = np.max(left_data) - np.min(left_data)
    right_range = np.max(right_data) - np.min(right_data)
    if left_range == 0 or right_range == 0:
        return [-1, 1], [-1, 1]
    left_max_abs = max(abs(np.max(left_data)), abs(np.min(left_data)))
    right_max_abs = max(abs(np.max(right_data)), abs(np.min(right_data)))
    base_range = max(left_max_abs, right_max_abs) * 1.2
    left_ylim = [-base_range, base_range]
    right_ylim = [-base_range, base_range]
    min_range = max(left_range, right_range) * 0.5
    if base_range < min_range:
        left_ylim = [-min_range, min_range]
        right_ylim = [-min_range, min_range]
    return left_ylim, right_ylim
def draw_4wave(values0, start, end):
    """绘制插值并去除等离子体毛刺的数据图（磁场三分量与磁场和密度相关系数波动图）"""
    time_mask = (values0['time_data'] >= start) & (values0['time_data'] <= end)
    time = values0['time_data'][time_mask]
    time_dt = Time(time, format='jd').datetime
    nep, tp = values0['nep_data'][time_mask], values0['tp_data'][time_mask]
    time_dt = Time(time, format='jd').datetime
    
    Br, Bt, Bn = values0['b_data'][:, 0][time_mask],\
        values0['b_data'][:, 1][time_mask], values0['b_data'][:, 2][time_mask]
    vr, vt, vn = values0['v_data'][:, 0][time_mask],\
        values0['v_data'][:, 1][time_mask], values0['v_data'][:, 2][time_mask]

    nep0 = np.mean(nep)
    B_total, v_total = np.linalg.norm(values0['b_data'][time_mask], axis=1),\
                       np.linalg.norm(values0['v_data'][time_mask], axis=1)
    B_total0, v_total0 = np.mean(B_total), np.mean(v_total)
    
    δB_total, δnep = B_total - B_total0, nep - nep0
    Br0, Bt0, Bn0 = np.mean(Br), np.mean(Bt), np.mean(Bn)
    δB, δBr, δBt, δBn = values0['b_data'][time_mask]\
        -np.mean(values0['b_data'][time_mask],axis=0), Br - Br0, Bt - Bt0, Bn - Bn0
    vr0, vt0, vn0 = np.mean(vr), np.mean(vt), np.mean(vn)
    δV, δvr, δvt, δvn = values0['v_data'][time_mask]\
        -np.mean(values0['v_data'][time_mask],axis=0), vr - vr0, vt - vt0, vn - vn0
    
    Ca = 1e-9 * 1e-3 / np.sqrt(4 * np.pi * 1e-7 * 1.6726e-27 *\
                               np.mean(nep) * 1e+6)
    δB2v, δBr2v, δBt2v, δBn2v = Ca * δB, Ca * δBr, Ca * δBt, Ca * δBn
    δVr2b, δVt2b, δVn2b = δvr / Ca, δvt / Ca, δvn / Ca
    
    cc_r = np.corrcoef(δBr, δVr2b)[0, 1]
    cc_t = np.corrcoef(δBt, δVt2b)[0, 1]
    cc_n = np.corrcoef(δBn, δVn2b)[0, 1]
    cc_bp = np.corrcoef(δB_total/B_total0, δnep/nep0)[0, 1]
    
    σc = 2*np.mean(np.sum(δV*δB2v,axis=1))/np.mean(np.sum(δV**2,axis=1)+np.sum(δB2v**2,axis=1))
    γ_A = np.mean(np.sum(δV**2,axis=1)) / np.mean(np.sum(δB2v**2,axis=1))
    σNep = np.std(nep) / np.mean(nep)
    
    fig, axes = plt.subplots(4, 1, figsize=(8, 6))
    plt.subplots_adjust(top=0.98, bottom=0.13, hspace=0.15, left=0.1, right=0.9)
    time_formatter = mdates.DateFormatter('%H:%M:%S')
    
    #-----------------------------------
    components = [
        ('Br,Vr', δBr, δVr2b, Br0, vr0, cc_r, 'k', 'b', axes[0]),
        ('Bt,Vt', δBt, δVt2b, Bt0, vt0, cc_t, 'k', 'b', axes[1]),
        ('Bn,Vn', δBn, δVn2b, Bn0, vn0, cc_n, 'k', 'b', axes[2]),
        ('|B|/<|B|>,Nep/<Nep>', δB_total/B_total0, δnep/nep0, B_total0, nep0, cc_bp, 'gray', 'g', axes[3])
    ]
    
    for i, (lbl, left_data, right_data, left_avg, right_avg, 
            cc, left_color, right_color, ax) in enumerate(components):
        ax_left = ax

        if ',' in lbl:
            left_label = f'δ{lbl.split(",")[0]}'
            right_label = f'δ{lbl.split(",")[1]}'
        else:
            left_label = f'δ{lbl}/{lbl}'
            right_label = 'δNep/Nep'

        line_left = ax_left.plot(time_dt, left_data, lw=1.2, color=left_color, label=left_label)
        ax_left.set_ylabel(left_label, color=left_color, fontsize=9)
        ax_left.tick_params(axis='y', labelcolor=left_color)
        
        left_ylim, right_ylim = calculate_aligned_ylim(left_data, right_data)
        ax_left.set_ylim(left_ylim)

        ax_right = ax_left.twinx()
        line_right = ax_right.plot(time_dt, right_data, lw=1.2, color=right_color, label=right_label)
        ax_right.set_ylabel(right_label, color=right_color, fontsize=9)
        ax_right.tick_params(axis='y', labelcolor=right_color)
        ax_right.set_ylim(right_ylim)
        
        ax_left.grid(True, alpha=0.3)

        if ',' in lbl and '|B|' not in lbl:
            info_text = f'CC={cc:.3f}\n<B>={left_avg:.2f}nT\n<V>={right_avg:.1f}km/s'
        elif lbl == '|B|/<|B|>,Nep/<Nep>':
            info_text = f'CC={cc:.3f}\n<|B|>={left_avg:.2f}nT\n<Np>={right_avg:.2f}cm$^{{-3}}$'
        else:
            info_text = f'CC={cc:.3f}'
        
        ax_left.text(0.02, 0.92, info_text, transform=ax_left.transAxes, 
                     fontsize=8, ha='left', va='top', 
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        lines = line_left + line_right
        labels = [l.get_label() for l in lines]
        ax_left.legend(lines, labels, loc='upper right', fontsize=7)

    for ax in axes[:3]:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)
        ax.set_xlim(time_dt[0], time_dt[-1])
        ax.xaxis.set_major_formatter(time_formatter)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
    
    axes[3].tick_params(labelbottom=True)
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    axes[3].xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
    axes[3].set_xlabel('Time (UTC)', fontsize=9)
    axes[3].set_xlim(time_dt[0], time_dt[-1])
    fig.suptitle(f"σ_c = {σc:.2f} | "
                 f"γ_A = {γ_A:.2f} | "
                 f"σ(Nep)/<Nep> = {σNep:.3f}",
                 y=0.04, fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
# ============================================================================
#十二图（加四波动相关性分析）
# ============================================================================    
def draw_12xiangguan(values0, start, end):
    """绘制插值并去除等离子体毛刺的数据图（磁场、速度、密度、温度、磁场三分量与磁场和密度相关系数
    目标超热电子时序图、超热电子颜色图，超热电子角度分布图，磁场幂律拟合）"""
    time_mask = (values0['time_data'] >= start) & (values0['time_data'] <= end)
    time = values0['time_data'][time_mask]
    time_dt = Time(time, format='jd').datetime
    nep, tp = values0['nep_data'][time_mask], values0['tp_data'][time_mask]
    time_dt = Time(time, format='jd').datetime
    
    Br, Bt, Bn = values0['b_data'][:, 0][time_mask],\
        values0['b_data'][:, 1][time_mask], values0['b_data'][:, 2][time_mask]
    vr, vt, vn = values0['v_data'][:, 0][time_mask],\
        values0['v_data'][:, 1][time_mask], values0['v_data'][:, 2][time_mask]

    nep0 = np.mean(nep)
    b_total, v_total = np.linalg.norm(values0['b_data'][time_mask], axis=1),\
                       np.linalg.norm(values0['v_data'][time_mask], axis=1)
    B_total0, v_total0 = np.mean(b_total), np.mean(v_total)
    
    δB_total, δnep = b_total - B_total0, nep - nep0
    Br0, Bt0, Bn0 = np.mean(Br), np.mean(Bt), np.mean(Bn)
    δB, δBr, δBt, δBn = values0['b_data'][time_mask]\
        -np.mean(values0['b_data'][time_mask],axis=0), Br - Br0, Bt - Bt0, Bn - Bn0
    vr0, vt0, vn0 = np.mean(vr), np.mean(vt), np.mean(vn)
    δV, δvr, δvt, δvn = values0['v_data'][time_mask]\
        -np.mean(values0['v_data'][time_mask],axis=0), vr - vr0, vt - vt0, vn - vn0
    
    Ca = 1e-9 * 1e-3 / np.sqrt(4 * np.pi * 1e-7 * 1.6726e-27 *\
                               np.mean(nep) * 1e+6)
    δB2v, δBr2v, δBt2v, δBn2v = Ca * δB, Ca * δBr, Ca * δBt, Ca * δBn
    δVr2b, δVt2b, δVn2b = δvr / Ca, δvt / Ca, δvn / Ca
    
    cc_r = np.corrcoef(δBr, δVr2b)[0, 1]
    cc_t = np.corrcoef(δBt, δVt2b)[0, 1]
    cc_n = np.corrcoef(δBn, δVn2b)[0, 1]
    cc_bp = np.corrcoef(δB_total/B_total0, δnep/nep0)[0, 1]
    
    σc = 2*np.mean(np.sum(δV*δB2v,axis=1))/np.mean(np.sum(δV**2,axis=1)+np.sum(δB2v**2,axis=1))
    γ_A = np.mean(np.sum(δV**2,axis=1)) / np.mean(np.sum(δB2v**2,axis=1))
    σNep = np.std(nep) / np.mean(nep)

    fig, axes = plt.subplots(10, 1, figsize = (8, 6))
    plt.subplots_adjust(top=0.95, bottom=0.12, hspace=0.15, left=0.1,right=0.5)
    time_formatter = mdates.DateFormatter('%H:%M:%S')
    #-----------------------------------
    colors, labels = ['k','r','g','b'], ['|B|', 'Br', 'Bt', 'Bn'] 
    axes[0].set_title(f'PSP Unified Analysis: {time_dt[0]} to {time_dt[-1]}', 
                fontsize=12, fontweight='bold',loc = 'center',  pad=10) 
    axes[0].plot(time_dt, b_total, '-', linewidth = 1.5, alpha = 1.0,
               color = colors[0], label = labels[0])
    for i in range(3):
        axes[0].plot(time_dt, values0['b_data'][:, i][time_mask], '-', linewidth = 1.5, 
                     alpha = 1.0, color = colors[i+1], label = labels[i+1])
    axes[0].set_ylabel('Magnetic Field\n[nT]', fontsize=8)
    axes[0].legend(loc='upper right', fontsize=6)
    #-----------------------------------
    labels = ['|V|', 'Vr', 'Vt', 'Vn'] 
    axes[1].plot(time_dt, v_total, '-', linewidth = 1.5, alpha = 1.0,
               color = colors[0], label = labels[0])
    for i in range(3):
        axes[1].plot(time_dt, values0['v_data'][:, i][time_mask], '-', linewidth = 1.5, 
                     alpha = 1.0, color = colors[i+1], label = labels[i+1])
    axes[1].set_ylabel('Velocity\n[km/s]', fontsize=8)
    axes[1].legend(loc='upper right', fontsize=6)
    
    axes[2].plot(time_dt, values0['nep_data'][time_mask], '-', linewidth = 1.5, alpha = 1.0,
               color = 'darkorange', label = 'Np')
    axes[2].set_ylabel('Density\n[cm$^{-3}$]', fontsize=8)
    axes[2].legend(loc='upper right', fontsize=6)
    #-----------------------------------
    axes[3].plot(time_dt, values0['tp_data'][time_mask], '-', linewidth = 1.5, alpha = 1.0,
               color = 'purple', label = 'Tem')
    axes[3].set_ylabel('Temperature\n[eV]', fontsize=8)
    axes[3].legend(loc='upper right', fontsize=6)
    #-----------------------------------
    components = [
        ('Br,Vr', δBr, δVr2b, Br0, vr0, cc_r, 'k', 'b', axes[4]),
        ('Bt,Vt', δBt, δVt2b, Bt0, vt0, cc_t, 'k', 'b', axes[5]),
        ('Bn,Vn', δBn, δVn2b, Bn0, vn0, cc_n, 'k', 'b', axes[6]),
        ('|B|/<|B|>,Nep/<Nep>', δB_total/B_total0, δnep/nep0, B_total0, nep0, cc_bp, 'gray', 'g', axes[7])
    ]
    
    for i, (lbl, left_data, right_data, left_avg, right_avg, 
            cc, left_color, right_color, ax) in enumerate(components):
        ax_left = ax

        if ',' in lbl:
            left_label = f'δ{lbl.split(",")[0]}'
            right_label = f'δ{lbl.split(",")[1]}'
        else:
            left_label = f'δ{lbl}/{lbl}'
            right_label = 'δNep/Nep'

        line_left = ax_left.plot(time_dt, left_data, lw=1.2, color=left_color, label=left_label)
        ax_left.set_ylabel(left_label, color=left_color, fontsize=9)
        ax_left.tick_params(axis='y', labelcolor=left_color)

        left_ylim, right_ylim = calculate_aligned_ylim(left_data, right_data)
        ax_left.set_ylim(left_ylim)

        ax_right = ax_left.twinx()
        line_right = ax_right.plot(time_dt, right_data, lw=1.2, color=right_color, label=right_label)
        ax_right.set_ylabel(right_label, color=right_color, fontsize=9)
        ax_right.tick_params(axis='y', labelcolor=right_color)
        ax_right.set_ylim(right_ylim)
        
        ax_left.grid(True, alpha=0.3)

        if ',' in lbl and '|B|' not in lbl:
            info_text = f'CC={cc:.3f}\n<B>={left_avg:.2f}nT\n<V>={right_avg:.1f}km/s'
        elif lbl == '|B|/<|B|>,Nep/<Nep>':
            info_text = f'CC={cc:.3f}\n<|B|>={left_avg:.2f}nT\n<Np>={right_avg:.2f}cm$^{{-3}}$'
        else:
            info_text = f'CC={cc:.3f}'
        
        ax_left.text(0.02, 0.92, info_text, transform=ax_left.transAxes, 
                     fontsize=8, ha='left', va='top', 
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        lines = line_left + line_right
        labels = [l.get_label() for l in lines]
        ax_left.legend(lines, labels, loc='upper right', fontsize=7)
    #-----------------------------------
    label3 = ['l', 'o', 'r']
    time_mask = (values0['time_s'] >= start) & (values0['time_s'] <= end)
    time_num = Time(values0['time_s'][time_mask], format='jd').datetime
    strahl3_total = np.sum(values0['strahl3'][time_mask], axis = 1)
    for i in range(3):
        axes[8].plot(time_num, strahl3_total[:, i], 
                     '-', linewidth = 1.5, alpha = 1.0,
                     color = colors[i+1], label = label3[i])
    axes[8].set_ylabel('Strahl Flux\n(cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$)', fontsize=8)
    axes[8].set_yscale('log')
    axes[8].legend(loc='upper right', fontsize=6)
    axes[8].tick_params(axis='y', which='both', labelsize=6)
    #-----------------------------------
    '''
    energy_target = (values0['strahl3'][time_mask, :, 1]).T
    min_vals = np.nanmin(energy_target, axis=0) 
    max_vals = np.nanmax(energy_target, axis=0) 
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    energy_draw = (energy_target - min_vals) / ranges 
    angles = np.arange(7.5, 180, 15)
    
    im = axes[9].pcolormesh(time_num, angles, energy_draw,
                       cmap='jet', shading='auto',
                       vmin=0, vmax=1)
    axes[9].set_yticks(angles)
    axes[9].tick_params(axis='y', labelsize=6)
    
    pos = axes[9].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Energy Ratio', fontsize=8)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=6)
    '''
    times_mask = (values0['time_s'] >= start) & (values0['time_s'] <= end)
    time_num = Time(values0['time_s'][times_mask], format='jd').datetime
    energy_target = (values0['strahl3'][times_mask, :, 1]).T
    log_flux = np.log10(energy_target)
    vmax, vmin = np.nanmax(log_flux), np.nanmin(log_flux)
    angles = np.arange(7.5, 181, 15)
    #-----------------------------------
    im = axes[9].pcolormesh(time_num, angles, log_flux,
                       cmap='jet', shading='auto',
                       vmin=vmin, vmax=vmax)
    axes[9].set_yticks(angles)
    axes[9].tick_params(axis='y', labelsize=6)
    
    pos = axes[9].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('log$_{10}$(Flux)', fontsize=8)
    #cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=6)
    axes[9].set_ylabel('Angle Channel',fontsize = 8)
    axes[9].set_xlabel('Time')
    
    for ax in axes[:9]:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)
    axes[9].tick_params(labelbottom=True) 
    time_formatter = mdates.DateFormatter('%H:%M')
    
    for ax in axes[:9]:
        ax.set_xlim(time_dt[0], time_dt[-1])
        ax.xaxis.set_major_formatter(time_formatter)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
    axes[9].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    axes[9].xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
    axes[9].set_xlabel('Time (UTC)', fontsize=9)
    #-----------------------------------
    strahl3_mean = np.mean(values0['strahl3'][time_mask,:,1], axis=0)
    ax6 = fig.add_axes([0.62, 0.63, 0.25, 0.25]) 
    ax6.plot(angles, strahl3_mean, 'ko-', linewidth=2.5, 
            markersize=6, markerfacecolor='lightgray', markeredgecolor='black',
            markeredgewidth=1.5)
    ax6.yaxis.tick_right()
    ax6.yaxis.set_label_position("right")
    ax6.set_yscale('log')
    ax6.set_title('PSP SPE Averaged Pitch Angle\n Distribution of Strahl Electrons', fontsize=10)
    ax6.set_xlabel('Pitch Angle (deg)', fontsize=9)
    ax6.set_ylabel('Omnidirectional Flux\n(cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$)', fontsize=9)
    ax6.grid(ax6, alpha=0.3)
    info_text = f"Energy: {values0['energy_name'][1]:.0f} eV\n"
    info_text += f'F0 = {strahl3_mean[0]:.2e}\n'
    info_text += f'F180 = {strahl3_mean[-1]:.2e}\n'
    ratio = strahl3_mean[-1] / strahl3_mean[0]
    if not np.isnan(ratio):
        info_text += f'F180/F0 = {ratio:.2f}'
    
    ax6.text(0.02, 0.98, info_text, transform=ax6.transAxes, 
            verticalalignment='top', fontsize=6, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    #-----------------------------------
    δBr_fft, δBt_fft, δBn_fft = np.fft.rfft(δBr), np.fft.rfft(δBt), np.fft.rfft(δBn)#FFT
    
    Pr = np.abs(δBr_fft)**2 * 2 / len(Br) * values0['resolution']#分量功率谱
    Pt = np.abs(δBt_fft)**2 * 2 / len(Bt) * values0['resolution']
    Pn = np.abs(δBn_fft)**2 * 2 / len(Bn) * values0['resolution']
    P_total = Pr + Pt + Pn#磁场波动功率谱的迹
    freq = np.fft.rfftfreq(len(Br), d = values0['resolution'])
    
    smooth_points = 7 #(1/600=0.00167)*6=0.01说明在0.01Hz之前有5个点，超过三个可以进行7点平滑
    P_total_smooth = np.copy(P_total)
    half_points = smooth_points // 2
    for i in range(half_points, len(P_total) - half_points):#数据平滑处理
        P_total_smooth[i] = np.mean(P_total[i-half_points:i+half_points+1])
    for i in range(half_points):#左边界
        P_total_smooth[i] = np.mean(P_total[:i+half_points+1])
    for i in range(len(P_total) - half_points, len(P_total)):#右边界
        P_total_smooth[i] = np.mean(P_total[i-half_points:])   
        
    freq_range = [0.01, 0.1]#设定拟合范围
    mask = (freq >= freq_range[0]) & (freq <= freq_range[1])#选取拟定范围内的点
    freq_fit = freq[mask]
    P_fit = P_total_smooth[mask]
    valid_mask = (P_fit > 0) & np.isfinite(P_fit) & (freq_fit > 0)#去除负值和无穷大值
    freq_fit = freq_fit[valid_mask]
    P_fit = P_fit[valid_mask]
    
    if len(freq_fit) >= 2:#斜率、截距、相关系数、显著性水平、标准误差
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.log10(freq_fit), np.log10(P_fit))
    else:slope, intercept, r_value, std_err = np.nan, np.nan, np.nan, np.nan
    
    ax7 = fig.add_axes([0.62, 0.2, 0.25, 0.25]) 
    freq_limitation = 1.0 / (len(Br) * values0['resolution'])
    ax7.loglog(freq[freq>=freq_limitation], P_total_smooth[freq>=freq_limitation],
                 'k-', linewidth=1.5)  
    ax7.yaxis.tick_right()
    ax7.yaxis.set_label_position("right")
    ax7.set_title(' Power spectral density scaling analysis', fontsize=10)
    ax7.set_xlabel('Frequency [Hz]', fontsize=9)
    ax7.set_ylabel('Power Spectral Density \n[nT²/Hz]', fontsize=9)
    ax7.grid(True, alpha=0.3, which='both')
    if not np.isnan(slope):
        fit_freq = np.logspace(np.log10(freq_range[0]), 
                              np.log10(freq_range[1]), 100)
        fit_P = 10**(intercept + slope * np.log10(fit_freq))
        ax7.loglog(fit_freq, fit_P, 'r--', linewidth=2)
        ax7.axvline(freq_range[0], color='gray', linestyle=':', alpha=0.7)
        ax7.axvline(freq_range[1], color='gray', linestyle=':', alpha=0.7)
    if not np.isnan(slope):
        text_str = f' α={slope:.2f} ± {std_err:.2f}'
        ax7.text(0.05, 0.95, text_str, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax7.tick_params(axis='both', which='major', labelsize=12)
    ax7.tick_params(axis='both', which='minor', labelsize=10)
    fig.suptitle(f"σ_c = {σc:.2f} | "
                 f"γ_A = {γ_A:.2f} | "
                 f"σ(Nep)/<Nep> = {σNep:.3f}",
                 x=0.3, y=0.04, fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))    
    plt.show()
# ============================================================================
#磁场数据研究湍流特性（幂律谱、小波变换、角度分布）
# ============================================================================ 
def mean_bysegment(value, window):
    n_windows = int(np.ceil(len(value) / window))
    filled_means = np.zeros_like(value)
    for i in range(n_windows):
        start_idx = i * window
        end_idx = min((i + 1) * window, len(value))  # 防止越界

        window_data = value[start_idx:end_idx]
        window_mean = np.mean(window_data)
        
        filled_means[start_idx:end_idx] = window_mean
    return filled_means
def draw_9byangle(values0, start, end):
    time_mask = (values0['time_data'] >= start) & (values0['time_data'] <= end)
    time = values0['time_data'][time_mask]
    time_dt = Time(time, format='jd').datetime
    
    Br, Bt, Bn = values0['b_data'][:, 0][time_mask],\
        values0['b_data'][:, 1][time_mask], values0['b_data'][:, 2][time_mask]
    Br0, Bt0, Bn0 = np.mean(Br), np.mean(Bt), np.mean(Bn)
    δBr, δBt, δBn = Br - Br0, Bt - Bt0, Bn - Bn0
    δBr_fft, δBt_fft, δBn_fft = np.fft.rfft(δBr), np.fft.rfft(δBt), np.fft.rfft(δBn)#FFT
    
    Pr = np.abs(δBr_fft)**2 * 2 / len(Br) * values0['resolution']#分量功率谱
    Pt = np.abs(δBt_fft)**2 * 2 / len(Bt) * values0['resolution']
    Pn = np.abs(δBn_fft)**2 * 2 / len(Bn) * values0['resolution']
    P_total = Pr + Pt + Pn#磁场波动功率谱的迹
    freq = np.fft.rfftfreq(len(Br), d = values0['resolution'])
    
    smooth_points = 7 #(1/600=0.00167)*6=0.01说明在0.01Hz之前有5个点，超过三个可以进行7点平滑
    P_total_smooth = np.copy(P_total)
    half_points = smooth_points // 2
    for i in range(half_points, len(P_total) - half_points):#数据平滑处理
        P_total_smooth[i] = np.mean(P_total[i-half_points:i+half_points+1])
    for i in range(half_points):#左边界
        P_total_smooth[i] = np.mean(P_total[:i+half_points+1])
    for i in range(len(P_total) - half_points, len(P_total)):#右边界
        P_total_smooth[i] = np.mean(P_total[i-half_points:])   
        
    freq_range = [0.01, 0.1]#设定拟合范围
    mask = (freq >= freq_range[0]) & (freq <= freq_range[1])#选取拟定范围内的点
    freq_fit = freq[mask]
    P_fit = P_total_smooth[mask]
    valid_mask = (P_fit > 0) & np.isfinite(P_fit) & (freq_fit > 0)#去除负值和无穷大值
    freq_fit = freq_fit[valid_mask]
    P_fit = P_fit[valid_mask]
    
    if len(freq_fit) >= 2:#斜率、截距、相关系数、显著性水平、标准误差
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.log10(freq_fit), np.log10(P_fit))
    else:slope, intercept, r_value, std_err = np.nan, np.nan, np.nan, np.nan
    #-----------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))  # 宽15，高10
    plt.subplots_adjust(top=0.92, bottom=0.08, wspace=0.3, hspace=0.4, left=0.08, right=0.92)
    freq_limitation = 1.0 / (len(Br) * values0['resolution'])
    axes[0][0].loglog(freq[freq>=freq_limitation], P_total_smooth[freq>=freq_limitation],
                 'k-', linewidth=1.5)  
    axes[0][0].set_title(' Power spectral density scaling analysis', fontsize=10)
    axes[0][0].set_xlabel('Frequency [Hz] - FFT', fontsize=9)
    axes[0][0].set_ylabel('Power Spectral Density \n[nT²/Hz]', fontsize=9)
    axes[0][0].grid(True, alpha=0.3, which='both')
    if not np.isnan(slope):
        fit_freq = np.logspace(np.log10(freq_range[0]), 
                              np.log10(freq_range[1]), 100)
        fit_P = 10**(intercept + slope * np.log10(fit_freq))
        axes[0][0].loglog(fit_freq, fit_P, 'r--', linewidth=2)
        axes[0][0].axvline(freq_range[0], color='gray', linestyle=':', alpha=0.7)
        axes[0][0].axvline(freq_range[1], color='gray', linestyle=':', alpha=0.7)
    if not np.isnan(slope):
        text_str = f' α={slope:.2f} ± {std_err:.2f}'
        axes[0][0].text(0.05, 0.95, text_str, transform=axes[0][0].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0][0].tick_params(axis='both', which='major', labelsize=12)
    axes[0][0].tick_params(axis='both', which='minor', labelsize=10)
    #-----------------------------------
    wavelet = 'cmor1-0.9549'
    f_min, f_max = 8 / ((end-start)*86400), 1 / values0['resolution'] / 2
    f_c = pywt.scale2frequency(wavelet, 1)
    scales_min = f_c / (f_max * values0['resolution'])
    if scales_min < 2 : scales_min = 2
    scales_max = f_c / (f_min * values0['resolution'])
    #scales = np.linspace(scales_min, scales_max, 32)
    scales = np.logspace(np.log10(scales_min), np.log10(scales_max), 32)
    coeff_Br, freq2 = pywt.cwt(δBr, scales, wavelet, sampling_period=values0['resolution'])
    coeff_Bt, _ = pywt.cwt(δBt, scales, wavelet, sampling_period=values0['resolution'])
    coeff_Bn, _ = pywt.cwt(δBn, scales, wavelet, sampling_period=values0['resolution'])
    time = np.arange(len(δBr)) * values0['resolution'] 
    power_total = (np.abs(coeff_Br)**2 + np.abs(coeff_Bt)**2 + np.abs(coeff_Bn)**2) 
    power_timesum = np.mean(power_total, axis=1)
    
    mask1 = (freq2 >= freq_range[0]) & (freq2 <= freq_range[1])#选取拟定范围内的点
    freq2_fit = freq2[mask1]
    power_timesum_fit = power_timesum[mask1]
    valid_mask2 = (power_timesum_fit > 0) & np.isfinite(power_timesum_fit) & (freq2_fit > 0)#去除负值和无穷大值
    freq2_fit = freq2_fit[valid_mask2]
    power_timesum_fit = power_timesum_fit[valid_mask2]
    
    if len(freq2_fit) >= 2:#斜率、截距、相关系数、显著性水平、标准误差
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
            np.log10(freq2_fit), np.log10(power_timesum_fit))
    else:slope2, intercept2, r_value2, std_err2 = np.nan, np.nan, np.nan, np.nan
    
    axes[0][1].loglog(freq2, power_timesum,'k-', linewidth=1.5)  
    axes[0][1].set_title(' Power spectral density scaling analysis', fontsize=10)
    axes[0][1].set_xlabel('Frequency [Hz] - Wavelet', fontsize=9)
    axes[0][1].set_ylabel('Power Spectral Density \n[nT²/Hz]', fontsize=9)
    axes[0][1].grid(True, alpha=0.3, which='both')
    if not np.isnan(slope2):
        fit_freq = np.logspace(np.log10(freq_range[0]), 
                              np.log10(freq_range[1]), 100)
        fit_P = 10**(intercept2 + slope2 * np.log10(fit_freq))
        axes[0][1].loglog(fit_freq, fit_P, 'r--', linewidth=2)
        axes[0][1].axvline(freq_range[0], color='gray', linestyle=':', alpha=0.7)
        axes[0][1].axvline(freq_range[1], color='gray', linestyle=':', alpha=0.7)
    if not np.isnan(slope2):
        text_str = f' α={slope2:.2f} ± {std_err2:.2f}'
        axes[0][1].text(0.05, 0.95, text_str, transform=axes[0][1].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0][1].tick_params(axis='both', which='major', labelsize=12)
    axes[0][1].tick_params(axis='both', which='minor', labelsize=10)
    #-----------------------------------
    im0 = axes[0][2].pcolormesh(time, freq2, power_total, 
                   cmap='jet', 
                   shading='auto',
                   norm=plt.cm.colors.LogNorm())

    pos = axes[0][2].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])
    cbar = fig.colorbar(im0, cax=cbar_ax)
    cbar.set_label('Power Spectral Density \n[nT²/Hz]', fontsize = 8)
    cbar.ax.tick_params(labelsize=6)
    axes[0][2].set_title('Wavelet Scalogram of Magnetic Field Fluctuations', fontsize=10)
    axes[0][2].set_ylabel('Frequencies [Hz]',fontsize = 9)
    axes[0][2].set_xlabel('Time [s]',fontsize = 9)

    axes[0][2].set_yscale('log')
    axes[0][2].set_ylim(freq2[-1], freq2[0])
    axes[0][2].yaxis.set_major_locator(LogLocator(base=2, numticks=10))
    axes[0][2].yaxis.set_major_formatter(LogFormatterMathtext(base=2))
    #-----------------------------------
    b_total, v_total = np.linalg.norm(values0['b_data'][time_mask], axis=1, keepdims=True),\
                       np.linalg.norm(values0['v_data'][time_mask], axis=1, keepdims=True)
    B_norm, V_norm = values0['b_data'][time_mask] / \
        b_total, values0['v_data'][time_mask] / v_total
    BV_angle_cos = np.sum(B_norm * V_norm, axis = 1)
    θʙᴠ_ori =np.degrees(np.arccos(np.clip(BV_angle_cos, -1, 1)))
    masks = [(θʙᴠ_ori>=i)&(θʙᴠ_ori<i+10) for i in range(0,170,10)]
    masks.append((θʙᴠ_ori>=170)&(θʙᴠ_ori<=180))
    bin_counts = [np.sum(mask) for mask in masks]
    angle1_num, angle2_num = bin_counts[:9], bin_counts[9:][::-1]
    angles = np.arange(0, 91, 10) 
    θʙᴠ = np.degrees(np.arccos(np.clip(np.abs(BV_angle_cos), 0, 1)))
    mask00, mask90= θʙᴠ<=10, θʙᴠ>=80
    angle_num = np.array(angle1_num) + np.array(angle2_num)
    axes[2][2].step(angles, np.concatenate([[0], angle_num]), color='k', linewidth=1.5,  
                    label='par~perp', alpha=1, linestyle='-')
    axes[2][2].step(angles, [0]+angle1_num, color='r', linewidth=1.0,  
                    label='0~90°', alpha=0.8, linestyle='-')
    axes[2][2].step(angles, [0]+angle2_num, color='b', linewidth=1.0,  
                    label='180~90°', alpha=0.8, linestyle='-')
    
    axes[2][2].set_xlim(0, 90)
    axes[2][2].legend(loc='upper right', fontsize=9)
    axes[2][2].set_title('Angular binned distribution', fontsize=10)
    axes[2][2].set_ylabel('Occurrences',fontsize = 9)
    axes[2][2].set_xlabel('Degrees [°]',fontsize = 9)
    
    B_total = np.linalg.norm(values0['b_data'][time_mask], axis=1)
    δB_total = B_total - np.mean(B_total) 
    if len(mask00[mask00 == True]) > 2:
        δB_total00 = δB_total[mask00]
        time00 = time[mask00]
        
        axes[1][0].scatter(time00, δB_total00, s=8, color='steelblue', 
                           alpha=0.6, label='δB_total00')#seagreen       

        axes[1][0].set_title('Distribution of quasi-parallel B', fontsize=10)         
        axes[1][0].set_ylabel('Power(0~10°) [nT$^2$/Hz]', fontsize=9)
        axes[1][0].set_xlabel('Time [s]',fontsize = 9)
        axes[1][0].legend(loc='upper right', fontsize=9)
        
        tb00 = LombScargle(time00, δB_total00, normalization='psd')
        freq00, power00 = tb00.autopower()#频率形状是频率网格的密度，不是数据点数
        dt = np.diff(time00)
       
        axes[1][1].loglog(freq00, power00, 'steelblue', linewidth=2)#darkseagreen

        '''
        time00_diff = np.diff(time00)
        dt00_mean = np.mean(time00_diff)
        autocorr = np.correlate(δB_total00, δB_total00, mode='full')
        autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
        if len(autocorr) > 1 and autocorr[1] > 0:
            rho = autocorr[1]
        else: rho = 0.5
        tau = -dt00_mean / np.log10(rho)
        power00_mean = np.mean(power00)
        ar_power = power00_mean * (1 - rho**2) / (1 + rho**2\
                                - 2*rho*np.cos(np.pi*freq00/np.max(freq00)))
        axes[1][1].loglog(freq00, ar_power, 'r--', linewidth=1.5, alpha=0.8, label='AR(1) noise')
        print("\n时间间隔统计:")
        print(f"  最小间隔: {time00_diff.min():.3f} 秒")
        print(f"  最大间隔: {time00_diff.max():.3f} 秒")
        print(f"  平均间隔: {time00_diff.mean():.3f} 秒")
        print(f"  间隔标准差: {time00_diff.std():.3f} 秒")
        # 蒙特卡洛模拟计算显著性水平
        n_simulations = 1000
        max_powers = []

        for i in range(n_simulations):

            noise = np.random.randn(len(δB_total00))
            ar1_noise = np.zeros_like(noise)
            ar1_noise[0] = noise[0]
    
        for j in range(1, len(noise)):
            ar1_noise[j] = rho * ar1_noise[j-1] + np.sqrt(1-rho**2) * noise[j]
    

        ls_noise = LombScargle(time00, ar1_noise)
        freq_sim, power_sim = ls_noise.autopower(minimum_frequency=min(freq00),
                                             maximum_frequency=max(freq00))
        max_powers.append(np.max(power_sim))


        confidence_level = np.percentile(max_powers, 99.74)

        axes[1][1].axhline(y=confidence_level, color='blue', linestyle='-.', 
                   linewidth=1.5, alpha=0.8, label='99.74% (3σ)')
        '''
        
    if len(mask90[mask90 == True]) > 2:
        δB_total90 = δB_total[mask90]
        time90 = time[mask90]
        
        axes[2][0].scatter(time90, δB_total90, s=8, color='seagreen', 
                           alpha=0.6, label='δB_total90')#seagreen       

        axes[2][0].set_title('Distribution of quasi-perpendicular B', fontsize=10)         
        axes[2][0].set_ylabel('Power(80~90°) [nT$^2$/Hz]', fontsize=9)
        axes[2][0].set_xlabel('Time [s]',fontsize = 9)
        axes[2][0].legend(loc='upper right', fontsize=9)
        
        tb90 = LombScargle(time90, δB_total90)
        freq90, power90 = tb90.autopower()#频率形状是频率网格的密度，不是数据点数
        axes[2][1].loglog(freq90, power90, 'seagreen', linewidth=2)#darkseagreen
        
        '''
        time00_diff = np.diff(time00)
        dt00_mean = np.mean(time00_diff)
        autocorr = np.correlate(δB_total00, δB_total00, mode='full')
        autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
        if len(autocorr) > 1 and autocorr[1] > 0:
            rho = autocorr[1]
        else: rho = 0.5
        tau = -dt00_mean / np.log10(rho)
        power00_mean = np.mean(power00)
        ar_power = power00_mean * (1 - rho**2) / (1 + rho**2\
                                - 2*rho*np.cos(np.pi*freq00/np.max(freq00)))
        axes[1][1].loglog(freq00, ar_power, 'r--', linewidth=1.5, alpha=0.8, label='AR(1) noise')
        print("\n时间间隔统计:")
        print(f"  最小间隔: {time00_diff.min():.3f} 秒")
        print(f"  最大间隔: {time00_diff.max():.3f} 秒")
        print(f"  平均间隔: {time00_diff.mean():.3f} 秒")
        print(f"  间隔标准差: {time00_diff.std():.3f} 秒")
        # 蒙特卡洛模拟计算显著性水平
        n_simulations = 1000
        max_powers = []

        for i in range(n_simulations):

            noise = np.random.randn(len(δB_total00))
            ar1_noise = np.zeros_like(noise)
            ar1_noise[0] = noise[0]
    
        for j in range(1, len(noise)):
            ar1_noise[j] = rho * ar1_noise[j-1] + np.sqrt(1-rho**2) * noise[j]
    

        ls_noise = LombScargle(time00, ar1_noise)
        freq_sim, power_sim = ls_noise.autopower(minimum_frequency=min(freq00),
                                             maximum_frequency=max(freq00))
        max_powers.append(np.max(power_sim))


        confidence_level = np.percentile(max_powers, 99.74)

        axes[1][1].axhline(y=confidence_level, color='blue', linestyle='-.', 
                   linewidth=1.5, alpha=0.8, label='99.74% (3σ)')
        '''
    #-----------------------------------
    points_time = 30
    window = int(points_time*values0['resolution'])
    Br0_seg, Bt0_seg, Bn0_seg = mean_bysegment(Br, window),\
                mean_bysegment(Bt, window), mean_bysegment(Bn, window)
    δBr_seg, δBt_seg, δBn_seg = Br - Br0_seg, Bt - Bt0_seg, Bn - Bn0_seg
    

    coeff_Br_seg, freq2 = pywt.cwt(δBr_seg, scales, wavelet, sampling_period=values0['resolution'])
    coeff_Bt_seg, _ = pywt.cwt(δBt_seg, scales, wavelet, sampling_period=values0['resolution'])
    coeff_Bn_seg, _ = pywt.cwt(δBn_seg, scales, wavelet, sampling_period=values0['resolution'])
    time_seg = np.arange(len(δBr_seg)) * values0['resolution'] 
    power_total_seg = (np.abs(coeff_Br_seg)**2 + np.abs(coeff_Bt_seg)**2 + np.abs(coeff_Bn_seg)**2) 
    im0 = axes[1][2].pcolormesh(time_seg, freq2, power_total_seg, 
                   cmap='jet', 
                   shading='auto',
                   norm=plt.cm.colors.LogNorm())

    pos = axes[1][2].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])
    cbar = fig.colorbar(im0, cax=cbar_ax)
    cbar.set_label('Power Spectral Density \n[nT²/Hz]', fontsize = 8)
    cbar.ax.tick_params(labelsize=6)
    axes[1][2].set_title('Wavelet Scalogram of Magnetic Field Fluctuations', fontsize=10)
    axes[1][2].set_ylabel('Frequencies [Hz]',fontsize = 9)
    axes[1][2].set_xlabel(f'Time [s] - by {points_time}s segments',fontsize = 9)

    axes[1][2].set_yscale('log')
    axes[1][2].set_ylim(freq2[-1], freq2[0])
    axes[1][2].yaxis.set_major_locator(LogLocator(base=2, numticks=10))
    axes[1][2].yaxis.set_major_formatter(LogFormatterMathtext(base=2))
    #-----------------------------------
    plt.show()
# ============================================================================
#磁场数据研究湍流特性（磁场强度、幂律谱时序变化、等离子体β、阿尔芬马赫数、温度各向异性、B-V夹角）
# ============================================================================ 
def draw_7tuandong(values0, start, end, vertical_lines=[2451545.0, 2451545.0]):
    time_mask = (values0['time_data'] >= start) & (values0['time_data'] <= end)
    time = values0['time_data'][time_mask]
    time_dt = Time(time, format='jd').datetime
    vertical_lines = Time(vertical_lines, format='jd').datetime
    b_total = np.linalg.norm(values0['b_data'][time_mask], axis = 1)
    v_total = np.linalg.norm(values0['v_data'][time_mask], axis = 1)
    Br, Bt, Bn = values0['b_data'][:, 0][time_mask], \
                 values0['b_data'][:, 1][time_mask], \
                 values0['b_data'][:, 2][time_mask]
    Vr, Vt, Vn = values0['v_data'][:, 0][time_mask], \
                 values0['v_data'][:, 1][time_mask], \
                 values0['v_data'][:, 2][time_mask]
                 
    fig, axes = plt.subplots(7, 1, figsize = (8, 6))
    plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.15, left=0.1,right=0.9)
    #-----------------------------------
    colors, labels = ['k','r','g','b'], ['|B|', 'Br', 'Bt', 'Bn'] 
    axes[0].set_title(f'PSP Unified Analysis: {time_dt[0]} to {time_dt[-1]}', 
                fontsize=10, fontweight='bold',loc = 'center',  pad=10) 
    axes[0].plot(time_dt, b_total, '-', alpha = 1.0,
               color = colors[0], label = labels[0])
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.7)
    for i in range(3):
        axes[0].plot(time_dt, values0['b_data'][:, i][time_mask], '-',
                     alpha = 1.0, color = colors[i+1], label = labels[i+1])
        
    axes[0].set_ylabel('Magnetic Field\n[nT]', fontsize=8)
    axes[0].legend(loc='upper right', fontsize=8)
    #-----------------------------------
    Br0, Bt0, Bn0 = np.mean(Br), np.mean(Bt), np.mean(Bn)
    δBr, δBt, δBn = Br - Br0, Bt - Bt0, Bn - Bn0
    dB = np.column_stack((δBr, δBt, δBn))
    M = (dB.T @ dB) / len(dB)
    
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    e_max, e_mid, e_min = eigenvectors[:, 0], eigenvectors[:, 1], eigenvectors[:, 2]

    e_N = e_min.copy()
    if e_N[0] > 0: e_N = -e_N
    e_L_temp = e_max.copy()
    e_M = np.cross(e_N, e_L_temp)
    e_M = e_M / np.linalg.norm(e_M)
    e_L = np.cross(e_M, e_N)
    e_L = e_L / np.linalg.norm(e_L)
    B_rtn = np.column_stack((Br, Bt, Bn))
    V_rtn = np.column_stack((Vr, Vt, Vn))
    V_L_temp = V_rtn @ e_L
    V_L_mean = np.mean(V_L_temp)
    if V_L_mean < 0:
        e_L = -e_L
        e_M = -e_M  
    r12 = eigenvalues[0] / eigenvalues[1]
    r23 = eigenvalues[1] / eigenvalues[2]
    V_mean = np.mean(V_rtn, axis=0) 
    B_LMN = B_rtn @ np.column_stack((e_L, e_M, e_N))
    B_L, B_M, B_N = B_LMN[:, 0], B_LMN[:, 1], B_LMN[:, 2]
    V_LMN = (V_rtn - V_mean) @ np.column_stack((e_L, e_M, e_N))
    V_L, V_M, V_N = V_LMN[:, 0], V_LMN[:, 1], V_LMN[:, 2]
    
    B_L_max = np.max(np.abs(B_L))
    B_M_max = np.median(np.abs(B_M))
    guide_field_ratio = B_M_max / B_L_max if B_L_max > 0 else 0

    V_L_mean = np.mean(V_L)
    V_L_std = np.std(V_L)
    '''
    V_N_mean = np.median(np.abs(V_N))
    d = V_N_mean * (end - start) / 3 * 86400
    '''
    print(f"特征值: {eigenvalues}")
    print(f"r12 = {r12:.2f}, r23 = {r23:.2f}")
    print(f"L: {e_L}, M: {e_M}, N: {e_N}")
    print(f"导向场比: {guide_field_ratio:.4f}")
    print(f"V_L = {V_L_mean:.2f} ± {V_L_std:.2f} km/s")
    #print(f"厚度约为 {d:.4f} km")
    axes[1].plot(time_dt, B_L, 'r-', label='B_L')
    axes[1].plot(time_dt, B_M, 'g-', label='B_M')
    axes[1].plot(time_dt, B_N, 'b-', label='B_N')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('B in LMN\n[nT]', fontsize=9)
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True)
    #-----------------------------------
    axes[2].plot(time_dt, V_L, 'r-', label='V_L ')
    axes[2].plot(time_dt, V_M, 'g-', label='V_M')
    axes[2].plot(time_dt, V_N, 'b-', label='V_N')
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('V in LMN\n[km/s]', fontsize=9)
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.3)    
    #-----------------------------------
    axes[3].plot(time_dt, values0['nep_data'][time_mask], '-', alpha = 1.0,
               color = 'darkorange', label = 'Np')
    axes[3].set_ylabel('Density\n[cm$^{-3}$]', fontsize=8)
    axes[3].legend(loc='upper right', fontsize=8)
    #-----------------------------------
    axes[4].plot(time_dt, values0['tp_data'][time_mask], '-', alpha = 1.0,
               color = 'purple', label = 'Tem')
    axes[4].set_ylabel('Temperature\n[eV]', fontsize=9)
    axes[4].legend(loc='upper right', fontsize=8)
    #-----------------------------------
    #Tp = m * wp**2 / 2 / kB, β = 2 * u0 * kB * Np * Tp / (|B|**2), k = 1.380649e-23
    kB, u0, mp = 1.380649e-23, 4*np.pi*1e-7, 1.6726e-27
    #Tem = mp * (1000 * values0['tp_data'][time_mask])**2 / 2 / kB
    Tp_K = 11604.518 * values0['tp_data'][time_mask]#eV->K
    βp = 2 * u0 * values0['nep_data'][time_mask] * 1e6 * kB * Tp_K / ((b_total*1e-9)**2)
    axes[5].plot(time_dt, βp, 'k-', alpha = 1.0, label = 'βp')
    axes[5].axhline(5, color='red', linestyle='--', alpha=0.7)
    axes[5].axhline(1, color='blue', linestyle='--', alpha=0.7)
    axes[5].set_yscale('log')
    axes[5].set_ylabel('β', fontsize=9)
    #-----------------------------------
    times_mask = (values0['time_s'] >= start) & (values0['time_s'] <= end)
    time_num = Time(values0['time_s'][times_mask], format='jd').datetime
    energy_target = (values0['strahl3'][times_mask, :, 1]).T
    energy_target_clean = np.where(energy_target == 0, 1e-10, energy_target)
    log_flux = np.log10(energy_target_clean)
    vmax, vmin = np.nanmax(log_flux), np.nanmin(log_flux)
    angles = np.arange(7.5, 181, 15)
    #-----------------------------------
    im = axes[6].pcolormesh(time_num, angles, log_flux,
                       cmap='jet', shading='auto',
                       vmin=vmin, vmax=vmax)
    axes[6].set_yticks(angles)
    axes[6].tick_params(axis='y', labelsize=6)
    
    pos = axes[6].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('log$_{10}$(Flux)', fontsize=8)
    #cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=6)

    axes[6].set_ylabel('Angle Channel',fontsize = 8)
    axes[6].set_xlabel('Time')
    

    for ax in axes:
        for vl in vertical_lines:
            ax.axvline(vl, color='red', linestyle='--', alpha=1.0, linewidth=2.0)
    
    for ax in axes[:-1]:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)
    axes[-1].tick_params(labelbottom=True) 
    time_formatter = mdates.DateFormatter('%H:%M:%S')
    
    for ax in axes[:-1]:
        ax.set_xlim(time_dt[0], time_dt[-1])
        ax.xaxis.set_major_formatter(time_formatter)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    axes[-1].set_xlabel('Time (UTC)', fontsize=9)
    axes[-1].set_xlim(time_dt[0], time_dt[-1])
    plt.show() 
# ============================================================================
#小波技术研究局部功率谱
# ============================================================================
def draw_xiaobobysegment(values0, start, end):
    time = values0['time_data']
    time_mask = (time >= start) & (time <= end)
    Br, Bt, Bn = values0['b_data'][:, 0][time_mask],\
        values0['b_data'][:, 1][time_mask], values0['b_data'][:, 2][time_mask]
    Vr, Vt, Vn = values0['v_data'][:, 0][time_mask],\
        values0['v_data'][:, 1][time_mask], values0['v_data'][:, 2][time_mask]
    #小波变换前的数据准备
    Ori_N = len(time[time_mask])
    w0, scale_num = 6.0, 32
    f_min, f_max = 8/((end-start)*86400.0), 1/values0['resolution']/4
    ratio = (f_max / f_min)**(1.0/(scale_num - 1))
    normalization = np.pi**(-0.25)  
    freqs = f_max * ratio**(-np.arange(scale_num)) 
    #scales = w0 / (2*np.pi) / freqs
    scales = 1.031 / freqs
    #拓展数据，两边各自拓展6s_max的范围用于边界处理
    s_max, support_sigma = np.max(scales), 6.0 
    half_support = support_sigma * s_max / 86400.0
    start_extended, end_extended = start - half_support, end + half_support
    time_extended_mask = (time >= start_extended ) & (time <= end_extended)    
    N = len(time[time_extended_mask])
    B_components = np.array([values0['b_data'][:, 0][time_extended_mask],\
        values0['b_data'][:, 1][time_extended_mask], values0['b_data'][:, 2][time_extended_mask]])
    wave_coeffs = np.zeros((3, scale_num, N), dtype=complex)
    power_components = np.zeros((3, scale_num, N))
    omega = 2 * np.pi * np.fft.fftfreq(N, values0['resolution'])
    for i in range(3): 
        B_hat = np.fft.fft(B_components[i])
        for l in range(scale_num):  
            s = scales[l] 
            psi_hat = np.sqrt(2*np.pi) * normalization * np.sqrt(s) * \
                      np.exp(-0.5 * (s * omega - w0)**2)
            psi_hat = np.where(omega > 0, psi_hat, 0)
            wave_coeffs[i, l] = np.fft.ifft(B_hat * psi_hat)
            power_components[i, l] = 2 * (np.abs(wave_coeffs[i, l])**2)
    total_power_extended = np.nansum(power_components, axis=0)
    s_mask = (time[time_extended_mask]>=start)&(time[time_extended_mask]<=end)
    total_power = total_power_extended[:, s_mask]
    power_time_mean = np.nanmean(total_power, axis=1)
    V_components = np.array([values0['v_data'][:, 0][time_extended_mask],\
        values0['v_data'][:, 1][time_extended_mask], values0['v_data'][:, 2][time_extended_mask]])
    local_B = np.zeros((3, scale_num, N))
    local_V = np.zeros((3, scale_num, N))    
    for i in range(3):
        for l in range(scale_num):
            s = scales[l]
            sigma = 1.67 * s / values0['resolution']
            half_width = max(int(4 * sigma), 1)
            t_window = np.arange(-half_width, half_width + 1) * values0['resolution']
            gauss_win = np.exp(-(t_window / s)**2) 
            gauss_win /= gauss_win.sum()
            local_B[i, l] = fftconvolve(B_components[i], gauss_win, mode='same')
            local_V[i, l] = fftconvolve(V_components[i], gauss_win, mode='same')
    B_mag, V_mag = np.sqrt(np.sum(local_B[:, :, s_mask]**2, axis=0)),\
        np.sqrt(np.sum(local_V[:, :, s_mask]**2, axis=0))
    B_mag_safe, V_mag_safe = np.where(B_mag > 1e-10, B_mag, 1e-10),\
        np.where(V_mag > 1e-10, V_mag, 1e-10)
    B_unit, V_unit = local_B[:, :, s_mask] / B_mag_safe,\
                     local_V[:, :, s_mask] / V_mag_safe
    cos_theta_BV = np.sum(B_unit * V_unit, axis=0) 
    cos_theta_BV = np.clip(cos_theta_BV, -1.0, 1.0)
    
    cos_theta_B = B_unit[0]  
    phi_B = np.arctan2(B_unit[2], B_unit[1])
    phi_B = np.where(phi_B < 0, phi_B + 2 * np.pi, phi_B)
      
    M_theta, N_phi = 101, 4
    cos_edges, phi_edges = np.linspace(-1, 1, M_theta + 1), \
                           np.linspace(0, 2 * np.pi, N_phi + 1)
    cos_a, cos_b = cos_edges[:-1], cos_edges[1:]
    F_a = cos_a * np.arccos(cos_a) - np.sqrt(1 - cos_a**2)
    F_b = cos_b * np.arccos(cos_b) - np.sqrt(1 - cos_b**2)
    theta_centers = (F_a - F_b) / (cos_a - cos_b) 
    theta_centers_deg = np.degrees(theta_centers)
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
    P_box, counts_box = np.zeros((scale_num, M_theta, N_phi)),\
                        np.zeros((scale_num, M_theta, N_phi))  
    for l in range(scale_num):
        cos_l, phi_l = cos_theta_B[l, :], phi_B[l, :]
        power_l = total_power[l, :]
        i_cos, i_phi = np.digitize(cos_l, cos_edges) - 1, \
                       np.digitize(phi_l, phi_edges) - 1
        i_cos, i_phi = np.clip(i_cos, 0, M_theta - 1), \
                       np.clip(i_phi, 0, N_phi - 1)
        counts_flat = np.bincount(i_cos*N_phi+i_phi, minlength=M_theta*N_phi)
        power_flat = np.bincount(i_cos*N_phi+i_phi, weights=power_l, \
                               minlength=M_theta*N_phi)
            
        counts_box[l] = counts_flat.reshape(M_theta, N_phi)
        P_box[l] = power_flat.reshape(M_theta, N_phi)
        
    mask_box = counts_box > 0
    P_box[mask_box] /= counts_box[mask_box]
    P_box[~mask_box] = np.nan

    phi_consistency = np.zeros((scale_num, M_theta))
    for l in range(scale_num):
        for i in range(M_theta):
            phi_powers = P_box[l, i, :]  # 4个ϕ方向的功率
            valid = ~np.isnan(phi_powers)
            if np.sum(valid) >= 2:
                mean_p = np.nanmean(phi_powers)
                if mean_p > 0:
                    rel_std = np.nanstd(phi_powers) / mean_p
                    phi_consistency[l, i] = rel_std
    
    mean_consistency = np.nanmean(phi_consistency)
    print(f"ϕ方向平均相对标准差: {mean_consistency*100:.1f}%")
    if mean_consistency < 0.2: print("✓ 轴对称性假设成立（差异<20%）")
    else: print("⚠ 存在显著ϕ方向各向异性（差异>20%），仅用θ分箱可能不准确")
    P_theta = np.nanmean(P_box, axis=2)  
    counts_theta = np.nansum(counts_box, axis=2)  
    
    mask_theta = counts_theta > 0
    P_theta[~mask_theta] = np.nan
    freq_low, freq_high = 0.01,0.1
    fit_indices = np.where((freqs >= freq_low) & (freqs <= freq_high))[0]
    if len(fit_indices) >= 3:
        f_fit = freqs[fit_indices]
        idx_parallel = np.where((theta_centers_deg <= 10) | (theta_centers_deg >= 170))[0]
        idx_perp = np.where((theta_centers_deg >= 80) & (theta_centers_deg <= 100))[0]
        power_parallel = np.nanmean(P_theta[:, idx_parallel], axis=1)
        power_perp = np.nanmean(P_theta[:, idx_perp], axis=1)
        power_all = np.nanmean(P_theta, axis=1) 
        def fit_power_law(f, p):
            valid = (f > 0) & (p > 0) & np.isfinite(p)
            if np.sum(valid) < 3:
                return np.nan, np.nan, 0, np.nan
            log_f = np.log10(f[valid])
            log_p = np.log10(p[valid])
            alpha, intercept, r_value, p_value, std_err = stats.linregress(log_f, log_p) 
            A = 10**intercept 
            return alpha, A, r_value**2, std_err
        alpha_par, A_par, r2_par, err_par = fit_power_law(f_fit, power_parallel[fit_indices])
        alpha_perp, A_perp, r2_perp, err_perp = fit_power_law(f_fit, power_perp[fit_indices])
        alpha_all, A_all, r2_all, err_all = fit_power_law(f_fit, power_all[fit_indices])
        alpha_time, A_time, r2_time, err_time = fit_power_law(f_fit, power_time_mean[fit_indices])
        print(f"所有方向平均: α_all = {alpha_all:.3f} ± {err_all:.3f}, R² = {r2_all:.4f}")
        print(f"平行方向谱指数: α_par = {alpha_par:.3f} ± {err_par:.3f}, R² = {r2_par:.4f}")
        print(f"垂直方向谱指数: α_perp = {alpha_perp:.3f} ± {err_perp:.3f}, R² = {r2_perp:.4f}")
        print(f"时间平均谱指数: α_time = {alpha_time:.3f} ± {err_time:.3f}, R² = {r2_time:.4f}")
    #-----------------------------------
    if len(fit_indices) >= 3:
        alpha_theta, err_theta = np.full(M_theta, np.nan), np.full(M_theta, np.nan)
        for i in range(M_theta):
            power_spectrum = P_theta[:, i]   
            alpha, A, r2, err = fit_power_law(freqs[fit_indices], power_spectrum[fit_indices])
            alpha_theta[i] = alpha
            err_theta[i] = err
    if len(fit_indices) > 0:
        mid_freq = (freq_high + freq_low) / 2
        freq_idx = np.argmin(np.abs(freqs - mid_freq))
        
        power_tar = P_theta[freq_idx, :]
        power_std = np.zeros_like(power_tar)  
    
        for i in range(len(theta_centers_deg)):
            phi_powers = P_box[freq_idx, i, :]  
            valid_phi = phi_powers[~np.isnan(phi_powers)]
            if len(valid_phi) > 1:
                power_std[i] = np.std(valid_phi)
            elif len(valid_phi) == 1:
                power_std[i] = valid_phi[0] * 0.1 
            else:power_std[i] = 0
    #-----------------------------------
    fig = plt.figure(figsize=(8, 6), constrained_layout=False)
    #axs = fig.subplots(4, 1, height_ratios=[1.2, 1, 1, 1], sharex=False)
    axes = []
    positions = [
        [0.1, 0.65, 0.38, 0.15],  
        [0.1, 0.37, 0.38, 0.15],
        [0.1, 0.22, 0.38, 0.15],
        [0.1, 0.07, 0.38, 0.15],
        [0.65, 0.45, 0.3, 0.3],
        [0.1, 0.80, 0.38, 0.15 ]
    ]
    for pos in positions:
        axes.append(fig.add_axes(pos))

    for i in range(1, 4):
        axes[i].sharex(axes[1])
    #-----------------------------------
    im = axes[5].pcolormesh(time[time_extended_mask], freqs, np.log10(total_power_extended), 
                           cmap='jet', shading='auto')
    im = axes[0].pcolormesh(time[time_extended_mask][s_mask], freqs, np.log(total_power), 
                           cmap='jet', shading='auto')
    axes[5].axvline(x=start, color='k', linestyle='--', linewidth=2)
    axes[5].axvline(x=end, color='k', linestyle='--', linewidth=2)
    axes[0].set_yscale('log')
    axes[5].set_yscale('log')
    axes[5].tick_params(axis='x', labelbottom=False)
    axes[0].set_ylim(freqs[-1], freqs[0])
    axes[0].set_ylabel('Frequency [Hz]', fontsize=9)
    axes[0].set_xlabel('Time [s]', fontsize=9)
    axes[5].set_title('Wavelet Power Spectrogram', fontsize=10, pad=10)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[0].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=0, ha='right')
    axes[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=10)) 
    
    pos = axes[0].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('log$_{10}$(Power [nT²/Hz])', fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    #-----------------------------------
    all_cos_theta = cos_theta_BV.flatten()  
    all_theta_deg = np.degrees(np.arccos(np.clip(all_cos_theta, -1, 1)))
    theta_bins_deg = np.arange(0, 181, 10)  
    counts, _ = np.histogram(all_theta_deg, bins=theta_bins_deg)
    axes[1].step(theta_bins_deg, [0] + list(counts), 
                color='seagreen', linewidth=2)
    axes[1].fill_between(theta_bins_deg[:-1], 0, counts, 
                        alpha=0.3, step='post', color='lightgreen')
    axes[1].set_ylabel('Number', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    #-----------------------------------
    axes[2].errorbar(theta_centers_deg, power_tar,
                    yerr=power_std, fmt='s', color='k', ecolor='gray', capsize=3,
                    label = f"{freqs[freq_idx]*1000:.0f} mHz ± σ")
    axes[2].set_yscale('log')
    axes[2].set_ylabel('Power [nT²/Hz]', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    #-----------------------------------
    axes[3].errorbar(theta_centers_deg, alpha_theta,
                    yerr=err_theta, fmt='s', color='k', ecolor='gray', capsize=3,
                    label = 'α ± σ')
    axes[3].set_ylabel('α', fontsize=9)
    axes[3].set_xlabel('θ [degrees]', fontsize=9)
    axes[3].grid(True, alpha=0.3)
    for i in range(1, 4):
        for angle in [10, 90, 170]:
            axes[i].axvline(angle, color='gray', linestyle=':', alpha=0.8, linewidth=1)
        if i == 1:  
            axes[i].axvspan(0, 10, alpha=0.15, color='steelblue', label='Parallel')
            axes[i].axvspan(170, 180, alpha=0.15, color='steelblue')
            axes[i].axvspan(80, 100, alpha=0.15, color='red', label='Perpendicular')
            axes[i].legend(fontsize=8)
    axes[3].axhline(-5/3, color='r', linestyle='--', alpha=0.6, label='-5/3')
    axes[3].axhline(-2.0, color='b', linestyle='--', alpha=0.6, label='-2.0')
    axes[2].legend(fontsize=8)
    axes[3].legend(fontsize=8)
    for i in range(1, 3):
        axes[i].tick_params(labelbottom=False)
    plt.show()
    #-----------------------------------
    axes[4].loglog(freqs, power_all, 'kD', markersize=5,  
               label=f'All directions:α={alpha_all:.2f} ± {err_all:.3f}')
    axes[4].loglog(freqs, power_parallel, 'b^', markersize=6, 
               label=f'Parallel θ<10°:α={alpha_par:.2f} ± {err_par:.3f}')
    axes[4].loglog(freqs, power_perp, 'rv', markersize=6, 
               label=f'Perp 80°<θ<100°:α={alpha_perp:.2f} ± {err_perp:.3f}')
    axes[4].loglog(freqs, power_time_mean, 'g-', linewidth=2, 
               label=f'Time-averaged:α={alpha_time:.2f} ± {err_time:.3f}', alpha=0.8)
    f_line = np.linspace(0.01, 0.1, 50)
    if not np.isnan(alpha_all):
        axes[4].loglog(f_line, A_all * f_line**(alpha_all), 'k--', alpha=1.0, linewidth=2)
    if not np.isnan(alpha_par):
        axes[4].loglog(f_line, A_par * f_line**(alpha_par), 'b--', alpha=0.8, linewidth=2)
    if not np.isnan(alpha_perp):
        axes[4].loglog(f_line, A_perp * f_line**(alpha_perp), 'r--', alpha=0.8, linewidth=2)
    if not np.isnan(alpha_time):
        axes[4].loglog(f_line, A_time * f_line**(alpha_time), 'm--', alpha=1.0, linewidth=1.5)
    axes[4].axvline(x=0.01, color='gray', linestyle='--', alpha=0.3)
    axes[4].axvline(x=0.1, color='gray', linestyle='--', alpha=0.3)
    axes[4].set_xlabel('Frequency [Hz] - By segments', fontsize=9)
    axes[4].set_ylabel('Power Spectral Density \n[nT²/Hz]', fontsize=9)
    axes[4].set_title('Angle-Resolved Power Spectra', fontsize=10)
    axes[4].legend(loc='lower left', fontsize=8)
    axes[4].grid(True, which='both', alpha=0.3)
    axes[4].set_xlim(0.001, 1)
# ============================================================================
#小波变换加平行场PSD(判断压缩性)和角度(诊断波的传播特性-判断是平行还是垂直传播)
# ============================================================================
def draw_3xiaoboandangle(values0, start, end, vertical_lines=[2451545.0, 2451545.0]):
    time = values0['time_data']
    time_mask = (time >= start) & (time <= end)
    Br, Bt, Bn = values0['b_data'][:, 0][time_mask],\
        values0['b_data'][:, 1][time_mask], values0['b_data'][:, 2][time_mask]
    Vr, Vt, Vn = values0['v_data'][:, 0][time_mask],\
        values0['v_data'][:, 1][time_mask], values0['v_data'][:, 2][time_mask]
    #小波变换前的数据准备
    Ori_N = len(time[time_mask])
    w0, scale_num = 6.0, 32
    f_min, f_max = 64/((end-start)*86400.0), 1/values0['resolution']/4
    ratio = (f_max / f_min)**(1.0/(scale_num - 1))
    normalization = np.pi**(-0.25)  
    freqs = f_max * ratio**(-np.arange(scale_num)) 
    #scales = w0 / (2*np.pi) / freqs
    scales = 1.031 / freqs
    #拓展数据，两边各自拓展6s_max的范围用于边界处理
    s_max, support_sigma = np.max(scales), 6.0 
    half_support = support_sigma * s_max / 86400.0
    start_extended, end_extended = start - half_support, end + half_support
    time_extended_mask = (time >= start_extended ) & (time <= end_extended)    
    N = len(time[time_extended_mask])
    B_components = np.array([values0['b_data'][:, 0][time_extended_mask],\
        values0['b_data'][:, 1][time_extended_mask], values0['b_data'][:, 2][time_extended_mask]])
    wave_coeffs = np.zeros((3, scale_num, N), dtype=complex)
    power_components = np.zeros((3, scale_num, N))
    omega = 2 * np.pi * np.fft.fftfreq(N, values0['resolution'])
    for i in range(3): 
        B_hat = np.fft.fft(B_components[i])
        for l in range(scale_num):  
            s = scales[l] 
            psi_hat = np.sqrt(2*np.pi) * normalization * np.sqrt(s) * \
                      np.exp(-0.5 * (s * omega - w0)**2)
            psi_hat = np.where(omega > 0, psi_hat, 0)
            wave_coeffs[i, l] = np.fft.ifft(B_hat * psi_hat)
            power_components[i, l] = 2 * (np.abs(wave_coeffs[i, l])**2)
    total_power_extended = np.nansum(power_components, axis=0)
    s_mask = (time[time_extended_mask]>=start)&(time[time_extended_mask]<=end)
    total_power = total_power_extended[:, s_mask]
    
    V_components = np.array([values0['v_data'][:, 0][time_extended_mask],\
        values0['v_data'][:, 1][time_extended_mask], values0['v_data'][:, 2][time_extended_mask]])
    local_B = np.zeros((3, scale_num, N))
    local_V = np.zeros((3, scale_num, N))    
    for i in range(3):
        for l in range(scale_num):
            s = scales[l]
            sigma = 1.67 * s / values0['resolution']
            half_width = max(int(4 * sigma), 1)
            t_window = np.arange(-half_width, half_width + 1) * values0['resolution']
            gauss_win = np.exp(-(t_window / s)**2) 
            gauss_win /= gauss_win.sum()
            local_B[i, l] = fftconvolve(B_components[i], gauss_win, mode='same')
            local_V[i, l] = fftconvolve(V_components[i], gauss_win, mode='same')
    B_mag, V_mag = np.sqrt(np.sum(local_B**2, axis=0)),\
        np.sqrt(np.sum(local_V**2, axis=0))
    B_mag_safe, V_mag_safe = np.where(B_mag > 1e-10, B_mag, 1e-10),\
        np.where(V_mag > 1e-10, V_mag, 1e-10)
    B_unit, V_unit = local_B / B_mag_safe,\
                     local_V / V_mag_safe
    cos_theta_BV = np.sum(B_unit * V_unit, axis=0) 
    cos_theta_BV = np.clip(cos_theta_BV, -1.0, 1.0)
    
    wave_coeffs_par = np.zeros((scale_num, N), dtype=complex)
    power_components_par = np.zeros((scale_num, N))
    for l in range(scale_num):  
        δB_par = B_components - local_B[:, l, :]
        B_par = np.sum(δB_par * B_unit[:, l, :], axis=0)
        B_par_hat = np.fft.fft(B_par)
        s = scales[l] 
        psi_hat = np.sqrt(2*np.pi) * normalization * np.sqrt(s) * \
                  np.exp(-0.5 * (s * omega - w0)**2)
        psi_hat = np.where(omega > 0, psi_hat, 0)
        wave_coeffs_par[l] = np.fft.ifft(B_par_hat * psi_hat)
        power_components_par[l] = 2 * (np.abs(wave_coeffs_par[l])**2)
    total_power_ratio = power_components_par[:, s_mask] / total_power 
    
    theta_kB = np.zeros((scale_num, N))
    sigma_max = np.zeros((scale_num, N))
    #sigma_ratio = np.zeros((scale_num, N)) 
    for l in range(scale_num):
        if l % 4 == 0 or l == scale_num - 1:
            print(f"  处理尺度 {l+1}/{scale_num} (f={freqs[l]:.4f} Hz)...")
        for n in range(N):
            dB_vec = wave_coeffs[:, l, n]
            power_total = np.sum(np.abs(dB_vec)**2)
            if power_total < 1e-20:
                theta_kB[l, n] = np.nan
                continue
            S = np.outer(dB_vec, np.conj(dB_vec))
            try:
                U, s_vals, Vh = linalg.svd(np.real(S))
                sigma_max[l, n] = s_vals[0]
                #if s_vals[0] > 0:
                #    sigma_ratio[l, n] = s_vals[1] / s_vals[0]  
                #    if sigma_ratio[l, n] < 8:
                #        theta_kB[l, n] = np.nan
                #        continue
                k_vec = U[:, -1]
                B0_vec = B_unit[:, l, n]
                cos_angle = np.abs(np.dot(k_vec, B0_vec))
                angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
                theta_kB[l, n] = angle_deg
            except np.linalg.LinAlgError:
                theta_kB[l, n] = np.nan
    theta_kB_crop = theta_kB[:, s_mask]
    #print(f"  数据中NaN比例: {np.sum(np.isnan(theta_kB))/theta_kB.size*100:.2f}%")
    #-----------------------------------
    fig = plt.figure(figsize=(12, 6), constrained_layout=False)
    #axs = fig.subplots(4, 1, height_ratios=[1.2, 1, 1, 1], sharex=False)
    axes = []
    positions = [
        [0.1, 0.68, 0.8, 0.30],  
        [0.1, 0.37, 0.8, 0.30],  
        [0.1, 0.06, 0.8, 0.30]  
    ]
    for pos in positions:
        axes.append(fig.add_axes(pos))

    for ax in axes[1:]:
        ax.set_xlabel('')
        ax.sharex(axes[0])
        ax.tick_params(labelbottom=False)
    axes[-1].tick_params(labelbottom=True) 
    time_formatter = mdates.DateFormatter('%H:%M')
    #-----------------------------------
    time_select = Time(time[time_extended_mask][s_mask], format='jd').datetime
    im0 = axes[0].pcolormesh(time_select, freqs, np.log10(total_power), 
                           cmap='jet', shading='auto')
    im1 = axes[1].pcolormesh(time_select, freqs, total_power_ratio, 
                           cmap='jet', shading='auto', vmin=0, vmax=1.0)
    im2 = axes[2].pcolormesh(time_select, freqs, theta_kB_crop, 
                           cmap='jet', shading='auto', vmin=0, vmax=90)
    for i in range(3):
        axes[i].set_yscale('log')
        axes[i].set_ylim(freqs[-1], freqs[0])
        axes[i].set_ylabel('Frequency [Hz]', fontsize=9)
        axes[i].set_xlabel('Time [s]', fontsize=9)
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[i].xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=0, ha='right')
        axes[i].xaxis.set_major_locator(mdates.MinuteLocator(interval=30)) 
    
    pos = axes[0].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0+pos.height/10, 0.01, pos.height*0.8])
    cbar = fig.colorbar(im0, cax=cbar_ax)
    cbar.set_label('log$_{10}$(Power [nT²/Hz])', fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    pos = axes[1].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0+pos.height/10, 0.01, pos.height*0.8])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('par/all(PSD)', fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    pos = axes[2].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0+pos.height/10, 0.01, pos.height*0.8])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label(r'$\theta_{kB_0}$ [°]', fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    vertical_lines = Time(vertical_lines, format='jd').datetime
    start_Time = Time(start, format='jd').datetime
    end_Time = Time(end, format='jd').datetime
    for ax in axes:
        ax.axvline(start_Time, color='black', linestyle='--', alpha=1.0, linewidth=2.0)
        ax.axvline(end_Time, color='black', linestyle='--', alpha=1.0, linewidth=2.0)
        for vl in vertical_lines:
            ax.axvline(vl, color='red', linestyle='--', alpha=1.0, linewidth=2.0)
# ============================================================================
#最小方差混合MVA建立LMN坐标系判断HCS
# ============================================================================
def draw_MVALMN(values0, start, end):
    time = values0['time_data']
    time_mask = (time >= start) & (time <= end)
    time0 = time[time_mask]
    time0_dt = Time(time0, format='jd').datetime
    
    Br, Bt, Bn = values0['b_data'][:, 0][time_mask], \
                 values0['b_data'][:, 1][time_mask], \
                 values0['b_data'][:, 2][time_mask]
    Vr, Vt, Vn = values0['v_data'][:, 0][time_mask], \
                 values0['v_data'][:, 1][time_mask], \
                 values0['v_data'][:, 2][time_mask]
    
    Br0, Bt0, Bn0 = np.mean(Br), np.mean(Bt), np.mean(Bn)
    δBr, δBt, δBn = Br - Br0, Bt - Bt0, Bn - Bn0
    dB = np.column_stack((δBr, δBt, δBn))
    M = (dB.T @ dB) / len(dB)
    
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    e_max, e_mid, e_min = eigenvectors[:, 0], eigenvectors[:, 1], eigenvectors[:, 2]

    e_N = e_min.copy()
    if e_N[0] > 0: e_N = -e_N
    e_L_temp = e_max.copy()
    e_M = np.cross(e_N, e_L_temp)
    e_M = e_M / np.linalg.norm(e_M)
    e_L = np.cross(e_M, e_N)
    e_L = e_L / np.linalg.norm(e_L)
    B_rtn = np.column_stack((Br, Bt, Bn))
    V_rtn = np.column_stack((Vr, Vt, Vn))
    V_L_temp = V_rtn @ e_L
    V_L_mean = np.mean(V_L_temp)
    if V_L_mean < 0:
        e_L = -e_L
        e_M = -e_M  
    r12 = eigenvalues[0] / eigenvalues[1]
    r23 = eigenvalues[1] / eigenvalues[2]
    V_mean = np.mean(V_rtn, axis=0) 
    B_LMN = B_rtn @ np.column_stack((e_L, e_M, e_N))
    B_L, B_M, B_N = B_LMN[:, 0], B_LMN[:, 1], B_LMN[:, 2]
    V_LMN = (V_rtn - V_mean) @ np.column_stack((e_L, e_M, e_N))
    V_L, V_M, V_N = V_LMN[:, 0], V_LMN[:, 1], V_LMN[:, 2]
    
    B_L_max = np.max(np.abs(B_L))
    B_M_max = np.median(np.abs(B_M))
    guide_field_ratio = B_M_max / B_L_max if B_L_max > 0 else 0

    V_L_mean = np.mean(V_L)
    V_L_std = np.std(V_L)
    
    print(f"特征值: {eigenvalues}")
    print(f"r12 = {r12:.2f}, r23 = {r23:.2f}")
    print(f"L: {e_L}, M: {e_M}, N: {e_N}")
    print(f"导向场比: {guide_field_ratio:.4f}")
    print(f"V_L = {V_L_mean:.2f} ± {V_L_std:.2f} km/s")
    #-----------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time0_dt, Br, 'r-', label='Br')
    axes[0].plot(time0_dt, Bt, 'g-', label='Bt')
    axes[0].plot(time0_dt, Bn, 'b-', label='Bn')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('B in RTN (nT)', fontsize=9)
    axes[0].legend()
    axes[0].grid(True)
    #-----------------------------------
    axes[1].plot(time0_dt, B_L, 'r-', label='B_L')
    axes[1].plot(time0_dt, B_M, 'g-', label='B_M')
    axes[1].plot(time0_dt, B_N, 'b-', label='B_N')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('B in LMN (nT)', fontsize=9)
    axes[1].legend()
    axes[1].grid(True)
    #-----------------------------------
    axes[2].plot(time0_dt, Vr, 'r-', label='Vr ')
    axes[2].plot(time0_dt, Vt, 'g-', label='Vt')
    axes[2].plot(time0_dt, Vn, 'b-', label='Vn')
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('V in RTN [km/s]', fontsize=9)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    #-----------------------------------
    axes[3].plot(time0_dt, V_L, 'r-', label='V_L ')
    axes[3].plot(time0_dt, V_M, 'g-', label='V_M')
    axes[3].plot(time0_dt, V_N, 'b-', label='V_N')
    axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[3].set_ylabel('V in LMN [km/s]', fontsize=9)
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    #-----------------------------------
    axes[-1].set_xlim([time0_dt[0], time0_dt[-1]])
    axes[-1].set_xlabel('Time', fontsize=9)
    plt.show()
    #-----------------------------------   
    if r23 >= 2:
        print("✅ MVA质量良好")
    else:
        print(f"⚠️ MVA质量一般 (r23={r23:.1f})")
    
    if guide_field_ratio < 0.1:
        print("✅ 弱导向场")
    elif guide_field_ratio < 0.3:
        print("⚠️ 中等导向场")
    else:
        print("⚠️ 强导向场")
# ============================================================================
#独立计算激波坐标系诊断慢激波边界，验证Petschek重联
# ============================================================================
def draw_jibozhenduan(values0, front, rear, window_sec=60, gap_sec=60):
    t_up_start = front - (window_sec + gap_sec) / 86400
    t_up_end   = front - gap_sec / 86400
    
    t_down_start = rear + gap_sec / 86400
    t_down_end   = rear + (gap_sec + window_sec) / 86400
    
    time = values0['time_data']
    mask_up = (time >= t_up_start) & (time <= t_up_end)
    mask_down = (time >= t_down_start) & (time <= t_down_end)
    
    B_up = np.mean(values0['b_data'][mask_up], axis=0)  
    B_down = np.mean(values0['b_data'][mask_down], axis=0)
    
    V_up = np.mean(values0['v_data'][mask_up], axis=0)
    V_down = np.mean(values0['v_data'][mask_down], axis=0)
    
    n_up = np.mean(values0['nep_data'][mask_up])
    n_down = np.mean(values0['nep_data'][mask_down])
    
    T_up = np.mean(values0['tp_data'][mask_up])
    T_down = np.mean(values0['tp_data'][mask_down])
    
    # ========== 激波法向计算 ==========
    B_cross = np.cross(B_up, B_down)
    n_shock = B_cross / np.linalg.norm(B_cross)
    
    # 确保 n_shock 指向上游（上游速度法向分量应为负）
    V_n_up_check = np.dot(V_up, n_shock)
    if V_n_up_check > 0:  # 如果为正，说明指向下游，需要反转
        n_shock = -n_shock
    
    B_n_up = np.dot(B_up, n_shock)
    B_n_down = np.dot(B_down, n_shock)
    V_n_up = np.dot(V_up, n_shock)
    V_n_down = np.dot(V_down, n_shock)

    B_t_up = np.linalg.norm(B_up - B_n_up * n_shock)
    B_t_down = np.linalg.norm(B_down - B_n_down * n_shock)
    
    # ========== 常数 ==========
    mp = 1.67e-27  # kg
    mu0 = 4 * np.pi * 1e-7  # H/m
    kb = 1.38e-23  # J/K
    
    # ========== 单位转换 ==========
    B_up_T = B_up * 1e-9  # nT -> T
    B_down_T = B_down * 1e-9
    B_t_up_T = B_t_up * 1e-9
    B_t_down_T = B_t_down * 1e-9
    B_mag_up_T = np.linalg.norm(B_up_T)
    B_mag_down_T = np.linalg.norm(B_down_T)
    
    n_up_m3 = n_up * 1e6  # cm^-3 -> m^-3
    n_down_m3 = n_down * 1e6
    T_up_K = T_up * 11600  # eV -> K
    T_down_K = T_down * 11600
    V_n_up_ms = V_n_up * 1e3  # km/s -> m/s
    V_n_down_ms = V_n_down * 1e3
    
    rho_up = n_up_m3 * mp
    rho_down = n_down_m3 * mp
    
    # ========== Rankine-Hugoniot 检验 ==========
    # 质量通量守恒
    mass_flux_up = rho_up * abs(V_n_up_ms)
    mass_flux_down = rho_down * abs(V_n_down_ms)
    mass_flux_ratio = mass_flux_down / mass_flux_up
    
    # 法向磁场守恒
    Bn_ratio = abs(B_n_down / B_n_up) if abs(B_n_up) > 1e-10 else np.nan
    
    # 热压
    P_up = n_up_m3 * kb * T_up_K
    P_down = n_down_m3 * kb * T_down_K
    
    # 总压守恒（法向动量通量）
    total_pressure_up = rho_up * V_n_up_ms**2 + P_up + B_t_up_T**2 / (2 * mu0)
    total_pressure_down = rho_down * V_n_down_ms**2 + P_down + B_t_down_T**2 / (2 * mu0)
    pressure_ratio = total_pressure_down / total_pressure_up
    
    # ========== 关键修正：Alfvén 速度（基于切向磁场，用于慢激波） ==========
    V_A_up = B_t_up_T / np.sqrt(mu0 * rho_up) if rho_up > 0 and B_t_up_T > 0 else np.nan
    V_A_down = B_t_down_T / np.sqrt(mu0 * rho_down) if rho_down > 0 and B_t_down_T > 0 else np.nan
    
    # 声速
    gamma = 5/3
    V_s_up = np.sqrt(gamma * P_up / rho_up) if rho_up > 0 else np.nan
    V_s_down = np.sqrt(gamma * P_down / rho_down) if rho_down > 0 else np.nan
    
    # ========== 慢磁声速度计算（修正版） ==========
    # 计算磁场与激波法向的夹角
    if B_mag_up_T > 0:
        cos_theta = abs(np.dot(B_up_T, n_shock)) / B_mag_up_T
        cos_theta = min(cos_theta, 1.0)  # 确保不超过1
    else:
        cos_theta = 0
    
    # 总 Alfvén 速度（基于总磁场）
    V_A_total_up = B_mag_up_T / np.sqrt(mu0 * rho_up) if rho_up > 0 else np.nan
    V_A_total_down = B_mag_down_T / np.sqrt(mu0 * rho_down) if rho_down > 0 else np.nan
    
    def calc_slow_mach_speed(V_s, V_A_total, cos_theta):
        """计算慢磁声速度"""
        if np.isnan(V_s) or np.isnan(V_A_total) or V_s <= 0 or V_A_total <= 0:
            return np.nan
        
        V_s2 = V_s**2
        V_A2 = V_A_total**2
        cos2 = cos_theta**2
        
        a = V_s2 + V_A2
        b = 4 * V_s2 * V_A2 * cos2
        
        discriminant = a**2 - b
        if discriminant < 0:
            discriminant = 0  # 数值误差导致负数时设为0
        
        return np.sqrt(0.5 * (a - np.sqrt(discriminant)))
    
    V_ms_up = calc_slow_mach_speed(V_s_up, V_A_total_up, cos_theta)
    V_ms_down = calc_slow_mach_speed(V_s_down, V_A_total_down, cos_theta)
    
    # ========== Mach 数计算 ==========
    M_A_up = abs(V_n_up_ms) / V_A_up if (V_A_up is not None and V_A_up > 0 and not np.isnan(V_A_up)) else np.nan
    M_A_down = abs(V_n_down_ms) / V_A_down if (V_A_down is not None and V_A_down > 0 and not np.isnan(V_A_down)) else np.nan
    M_ms_up = abs(V_n_up_ms) / V_ms_up if (V_ms_up is not None and V_ms_up > 0 and not np.isnan(V_ms_up)) else np.nan
    M_ms_down = abs(V_n_down_ms) / V_ms_down if (V_ms_down is not None and V_ms_down > 0 and not np.isnan(V_ms_down)) else np.nan
    
    # ========== 慢激波判据（考虑数值误差） ==========
    # 检查是否所有值都有效
    mach_valid = not any(np.isnan([M_ms_up, M_ms_down, M_A_up, M_A_down]))
    rh_valid = not np.isnan(mass_flux_ratio) and not np.isnan(pressure_ratio)
    
    is_slow_shock = False
    if mach_valid and rh_valid:
        is_slow_shock = (
            (M_ms_up > 1) and 
            (M_ms_down < 1) and 
            (M_A_up < 1) and 
            (M_A_down < 1) and
            (0.7 < mass_flux_ratio < 1.3) and  # 放宽到30%误差
            (0.7 < pressure_ratio < 1.3) and
            (np.isnan(Bn_ratio) or 0.7 < Bn_ratio < 1.3)
        )
    
    # ========== 打印结果 ==========
    print("="*50)
    print("激波分析结果")
    print("="*50)
    print(f"\n时间窗口:")
    print(f"  上游: {t_up_start:.6f} - {t_up_end:.6f} JD")
    print(f"  下游: {t_down_start:.6f} - {t_down_end:.6f} JD")
    
    print(f"\n上下游参数:")
    print(f"  B_up: [{B_up[0]:.1f}, {B_up[1]:.1f}, {B_up[2]:.1f}] nT")
    print(f"  B_down: [{B_down[0]:.1f}, {B_down[1]:.1f}, {B_down[2]:.1f}] nT")
    print(f"  V_n_up: {V_n_up:.1f} km/s")
    print(f"  V_n_down: {V_n_down:.1f} km/s")
    print(f"  n_up: {n_up:.1f} cm^-3")
    print(f"  n_down: {n_down:.1f} cm^-3")
    print(f"  T_up: {T_up:.1f} eV")
    print(f"  T_down: {T_down:.1f} eV")
    
    print(f"\n激波法向: [{n_shock[0]:.3f}, {n_shock[1]:.3f}, {n_shock[2]:.3f}]")
    
    print(f"\n物理参数（SI单位转换后）:")
    if not np.isnan(V_A_up):
        print(f"  V_A_up: {V_A_up/1e3:.1f} km/s (基于切向磁场)")
    else:
        print(f"  V_A_up: nan")
    if not np.isnan(V_ms_up):
        print(f"  V_ms_up: {V_ms_up/1e3:.1f} km/s")
    else:
        print(f"  V_ms_up: nan")
    
    print(f"\nRankine-Hugoniot检验:")
    print(f"  质量通量比: {mass_flux_ratio:.3f}")
    print(f"  总压比: {pressure_ratio:.3f}")
    print(f"  法向磁场比: {Bn_ratio:.3f}")
    
    print(f"\nMach数:")
    print(f"  M_A_up: {M_A_up:.3f}")
    print(f"  M_A_down: {M_A_down:.3f}")
    print(f"  M_ms_up: {M_ms_up:.3f}")
    print(f"  M_ms_down: {M_ms_down:.3f}")
    
    print(f"\n慢激波判据: {'✅ 是' if is_slow_shock else '❌ 否'}")
    
    # ========== 返回结果字典 ==========
    result = {
        'windows': {
            'upstream': (t_up_start, t_up_end),
            'downstream': (t_down_start, t_down_end)
        },
        'upstream': {
            'B': B_up, 'V': V_up, 'n': n_up, 'T': T_up,
            'V_n': V_n_up, 'B_n': B_n_up, 'B_t': B_t_up
        },
        'downstream': {
            'B': B_down, 'V': V_down, 'n': n_down, 'T': T_down,
            'V_n': V_n_down, 'B_n': B_n_down, 'B_t': B_t_down
        },
        'shock_normal': n_shock,
        'rh_test': {
            'mass_flux_ratio': mass_flux_ratio,
            'pressure_ratio': pressure_ratio,
            'Bn_ratio': Bn_ratio
        },
        'mach': {
            'M_A_up': M_A_up, 'M_A_down': M_A_down,
            'M_ms_up': M_ms_up, 'M_ms_down': M_ms_down,
            'V_A_up': V_A_up, 'V_A_down': V_A_down,
            'V_ms_up': V_ms_up, 'V_ms_down': V_ms_down
        },
        'is_slow_shock': is_slow_shock
    }
    
    return result
# ============================================================================
#绘制HCS前后以及中心HCS的阿尔芬性变化
# ============================================================================
def sliding_alfvenicity_triple_segment(values0, start_time, end_time, window_sec=600, step_sec=100):

    #计算范围外扩1小时，但分段仍基于目标端
    time = values0['time_data']
    
    # 计算范围外扩1小时
    analysis_start = start_time - 1/48  # 前扩半小时
    analysis_end = end_time + 1/48      # 后扩半小时
    
    # 提取分析范围内的数据
    time_mask = (time >= analysis_start) & (time <= analysis_end)
    time0 = time[time_mask]
    
    # 找到目标端在分析时间轴上的索引
    start_idx_in_analysis = np.searchsorted(time0, start_time)
    end_idx_in_analysis = np.searchsorted(time0, end_time)
    
    # 目标端内的点数
    target_points = end_idx_in_analysis - start_idx_in_analysis
    segment_points = target_points // 3
    
    # 计算关键时间点（在分析时间轴上）
    t1 = time0[0]                    # 分析开始
    t2 = time0[start_idx_in_analysis + segment_points]  # HCS开始
    t3 = time0[start_idx_in_analysis + 2*segment_points]  # HCS结束
    t4 = time0[-1]                   # 分析结束
    
    print(f"\n三段结构分析:")
    print(f"  目标端: {start_time} - {end_time}")
    print(f"  分析范围: {analysis_start} - {analysis_end} (外扩1小时)")
    print(f"  总分析点数: {len(time0)}")
    print(f"  目标端点数: {target_points}")
    print(f"  每段点数: {segment_points}")
    print(f"  分段:")
    print(f"    前段(分析): {t1} - {t2}")
    print(f"    HCS(分析): {t2} - {t3}")
    print(f"    后段(分析): {t3} - {t4}")
    
    Br = values0['b_data'][:, 0][time_mask]
    Bt = values0['b_data'][:, 1][time_mask]
    Bn = values0['b_data'][:, 2][time_mask]
    B = np.column_stack([Br, Bt, Bn])
    
    Vr = values0['v_data'][:, 0][time_mask]
    Vt = values0['v_data'][:, 1][time_mask]
    Vn = values0['v_data'][:, 2][time_mask]
    V = np.column_stack([Vr, Vt, Vn])
    
    nep = values0['nep_data'][time_mask]
    resolution = values0['resolution']
    
    window_points = int(window_sec / resolution)
    step_points = int(step_sec / resolution)
    
    print(f"\n滑动窗口参数:")
    print(f"  窗口长度: {window_sec}秒 ({window_points}点)")
    print(f"  滑动步长: {step_sec}秒 ({step_points}点)")
    
    n_windows = (len(time0) - window_points) // step_points + 1
    
    results = {
        'time': [],           
        'rel_time': [],       
        'segment': [],        
        'cc_r': [], 'cc_t': [], 'cc_n': [], 'cc_bn' : [], 
        'sigma_c': [], 'gamma_A': [], 'sigma_nep': [],
        'B_mean': [], 'nep_mean': []
    }
    
    hcs_center = (t2 + t3) / 2  # HCS中心在分析时间轴上的位置
    
    for i in range(0, len(time0) - window_points + 1, step_points):
        time_win = time0[i:i+window_points]
        B_win = B[i:i+window_points]
        V_win = V[i:i+window_points]
        nep_win = nep[i:i+window_points]
        
        t_center = time_win[len(time_win)//2]
        
        # 分段判断（基于HCS在分析时间轴上的位置）
        if t_center < t2:
            segment = 'front'
        elif t_center < t3:
            segment = 'center'
        else:
            segment = 'rear'
        
        rel_time_min = (t_center - hcs_center) * 1440
        
        B0 = np.mean(B_win, axis=0)
        V0 = np.mean(V_win, axis=0)
        nep0 = np.mean(nep_win)
        
        dB = B_win - B0
        dV = V_win - V0
        
        B_mag = np.linalg.norm(B_win, axis=1)
        B_mag0 = np.mean(B_mag)
        dbcb = (B_mag-B_mag0) / B_mag0
        dnepcnep = nep_win / nep0
        
        Ca = 1e-9 * 1e-3 / np.sqrt(4 * np.pi * 1e-7 * 1.6726e-27 * nep0 * 1e6)
        dB2v = Ca * dB
        
        cc_r = np.corrcoef(dB2v[:, 0], dV[:, 0])[0, 1] if len(dB) > 1 else np.nan
        cc_t = np.corrcoef(dB2v[:, 1], dV[:, 1])[0, 1] if len(dB) > 1 else np.nan
        cc_n = np.corrcoef(dB2v[:, 2], dV[:, 2])[0, 1] if len(dB) > 1 else np.nan
        cc_bn = np.corrcoef(dbcb, dnepcnep)[0, 1] if len(dB) > 1 else np.nan
        
        cross = 2 * np.mean(np.sum(dV * dB2v, axis=1))
        energy = np.mean(np.sum(dV**2, axis=1) + np.sum(dB2v**2, axis=1))
        sigma_c = cross / energy if energy != 0 else np.nan
        
        Ek = np.mean(np.sum(dV**2, axis=1))
        Eb = np.mean(np.sum(dB2v**2, axis=1))
        gamma_A = Ek / Eb if Eb != 0 else np.nan
        
        sigma_nep = np.std(nep_win) / nep0 if nep0 != 0 else np.nan
        
        results['time'].append(t_center)
        results['rel_time'].append(rel_time_min)
        results['segment'].append(segment)
        results['cc_r'].append(cc_r)
        results['cc_t'].append(cc_t)
        results['cc_n'].append(cc_n)
        results['cc_bn'].append(cc_bn)
        results['sigma_c'].append(sigma_c)
        results['gamma_A'].append(gamma_A)
        results['sigma_nep'].append(sigma_nep)
        results['B_mean'].append(B_mag0)
        results['nep_mean'].append(nep0)
    
    for key in results:
        if key not in ['segment']:
            results[key] = np.array(results[key])
    
    print(f"\n计算完成: {len(results['time'])} 个窗口")
    print(f"  前段窗口数: {np.sum(np.array(results['segment']) == 'front')}")
    print(f"  HCS窗口数: {np.sum(np.array(results['segment']) == 'center')}")
    print(f"  后段窗口数: {np.sum(np.array(results['segment']) == 'rear')}")
    
    return results, {'t1': t1, 't2': t2, 't3': t3, 't4': t4}
def plot_triple_segment_results(results, seg_times):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    time_dt = Time(results['time'], format='jd').datetime
    segments = np.array(results['segment'])
    colors = {'front': 'blue', 'center': 'red', 'rear': 'green'}
    
    t2_dt = Time(seg_times['t2'], format='jd').datetime
    t3_dt = Time(seg_times['t3'], format='jd').datetime
    
    # 子图1：σ_c
    ax = axes[0]
    for seg in ['front', 'center', 'rear']:
        mask = segments == seg
        ax.scatter(time_dt[mask], results['sigma_c'][mask], 
                  c=colors[seg], s=10, alpha=0.7, label=f'{seg}')
    ax.plot(time_dt, results['sigma_c'], 'k-', alpha=0.3, linewidth=0.5)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.8, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('σ_c')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=t2_dt, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(x=t3_dt, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(x=t2_dt-(t3_dt-t2_dt), color='k', linestyle='--', linewidth=1.5)
    ax.axvline(x=t3_dt+(t3_dt-t2_dt), color='k', linestyle='--', linewidth=1.5)
    
    # 子图2：γ_A
    ax = axes[1]
    for seg in ['front', 'center', 'rear']:
        mask = segments == seg
        ax.scatter(time_dt[mask], results['gamma_A'][mask], 
                  c=colors[seg], s=10, alpha=0.7, label=f'{seg}')
    ax.plot(time_dt, results['gamma_A'], 'k-', alpha=0.3, linewidth=0.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('γ_A')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=t2_dt, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(x=t3_dt, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(x=t2_dt-(t3_dt-t2_dt), color='k', linestyle='--', linewidth=1.5)
    ax.axvline(x=t3_dt+(t3_dt-t2_dt), color='k', linestyle='--', linewidth=1.5)
    
    # 子图3：相关系数
    ax = axes[2]
    ax.plot(time_dt, results['cc_r'], 'r-', alpha=0.7, label='CC_r', linewidth=1)
    ax.plot(time_dt, results['cc_t'], 'g-', alpha=0.7, label='CC_t', linewidth=1)
    ax.plot(time_dt, results['cc_n'], 'b-', alpha=0.7, label='CC_n', linewidth=1)
    if 'cc_bn' in results:
        ax.plot(time_dt, results['cc_bn'], 'k-', alpha=0.7, label='CC_BN', linewidth=1)
    ax.set_ylabel('Correlation Coefficients')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1, 1])
    ax.axvline(x=t2_dt, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(x=t3_dt, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(x=t2_dt-(t3_dt-t2_dt), color='k', linestyle='--', linewidth=1.5)
    ax.axvline(x=t3_dt+(t3_dt-t2_dt), color='k', linestyle='--', linewidth=1.5)
    
    # 子图4：密度波动
    ax = axes[3]
    ax.plot(time_dt, results['sigma_nep'], 'o-', alpha=0.7, label='δN/<N>', linewidth=0.5, markersize=2)
    ax.set_ylabel('Relative Density Fluctuation')
    ax.set_xlabel('Time (UTC)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.12, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=t2_dt, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(x=t3_dt, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(x=t2_dt-(t3_dt-t2_dt), color='k', linestyle='--', linewidth=1.5)
    ax.axvline(x=t3_dt+(t3_dt-t2_dt), color='k', linestyle='--', linewidth=1.5)
    
    # 设置时间格式
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[3].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=30)
    
    # 添加HCS标注
    axes[0].text(t2_dt, 0.95, 'HCS Start', transform=axes[0].get_xaxis_transform(),
                ha='right', fontsize=8, rotation=90)
    axes[0].text(t3_dt, 0.95, 'HCS End', transform=axes[0].get_xaxis_transform(),
                ha='left', fontsize=8, rotation=90)
    
    plt.tight_layout()
    return fig
def get_segment_statistics(results):
    segments = np.array(results['segment'])
    stats = {}
    for seg in ['front', 'center', 'rear']:
        mask = segments == seg
        if np.any(mask):
            stats[seg] = {
                'sigma_c_mean': np.nanmean(results['sigma_c'][mask]),
                'sigma_c_std': np.nanstd(results['sigma_c'][mask]),
                'sigma_c_median': np.nanmedian(results['sigma_c'][mask]),
                'gamma_A_mean': np.nanmean(results['gamma_A'][mask]),
                'gamma_A_std': np.nanstd(results['gamma_A'][mask]),
                'n_windows': np.sum(mask)
            }
    if 'front' in stats and 'center' in stats:
        stats['change'] = {
            'front_to_center': stats['front']['sigma_c_mean'] - stats['center']['sigma_c_mean'],
            'front_to_center_pct': (stats['front']['sigma_c_mean'] - stats['center']['sigma_c_mean']) / stats['front']['sigma_c_mean'] * 100
        }
    if 'center' in stats and 'rear' in stats:
        stats['change']['center_to_rear'] = stats['rear']['sigma_c_mean'] - stats['center']['sigma_c_mean']
        stats['change']['recovery_pct'] = (stats['rear']['sigma_c_mean'] - stats['center']['sigma_c_mean']) / (stats['front']['sigma_c_mean'] - stats['center']['sigma_c_mean']) * 100
    return stats
def analyze_my_hcs_event(values0, start_time, end_time, window_sec=600, step_sec=100):
    print("="*60)
    print(f"分析HCS事件: {start_time} - {end_time}")
    print("="*60)
    
    results, seg_times = sliding_alfvenicity_triple_segment(
        values0, start_time, end_time, 
        window_sec=window_sec, step_sec=step_sec)
    
    stats = get_segment_statistics(results)
    
    print("\n" + "="*60)
    print("统计结果:")
    print("-"*60)
    for seg in ['front', 'center', 'rear']:
        if seg in stats:
            print(f"{seg:>10}: σ_c = {stats[seg]['sigma_c_mean']:.3f} ± {stats[seg]['sigma_c_std']:.3f} "
                  f"(n={stats[seg]['n_windows']})")
    
    if 'change' in stats:
        print("-"*60)
        print(f"下降幅度: {stats['change']['front_to_center']:.3f} ({stats['change']['front_to_center_pct']:.1f}%)")
        if 'recovery_pct' in stats['change']:
            print(f"恢复程度: {stats['change']['recovery_pct']:.1f}%")

    fig = plot_triple_segment_results(results, seg_times)
    return results, stats, fig
# ============================================================================
#按不同滑动窗口寻找良好的MVA分析区间
# ============================================================================
def jd_to_datetime(jd):
    jd2000 = 2451545.0
    days_since_2000 = jd - jd2000
    return datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=float(days_since_2000))
def draw_MVA_step(values0, start, end, window_sizes=[60, 90, 120, 180, 240],
                  r12_threshold=10, r23_threshold=2,
                  target_window=120, overlap=0.75,
                  keep_partial=True, extend_factor=2):
    
    max_window = max(window_sizes)
    extend_seconds = max_window * extend_factor
    extend_days = extend_seconds / 86400.0
    
    data_start = start - extend_days
    data_end = end + extend_days
    
    time = values0['time_data']
    data_mask = (time >= data_start) & (time <= data_end)
    time_ext = time[data_mask]
    Br_ext = values0['b_data'][:, 0][data_mask]
    Bt_ext = values0['b_data'][:, 1][data_mask]
    Bn_ext = values0['b_data'][:, 2][data_mask]
    
    target_days = end - start
    all_results = {}
    highlight_intervals = []  

    for window_sec in window_sizes:
        window_days = window_sec / 86400.0
        step_days = window_days * (1 - overlap)
        
        if step_days > 0:
            n_intervals = int(np.ceil(target_days / step_days))
            centers = np.linspace(start, end, n_intervals + 1)
        else:
            centers = np.array([(start + end)/2])
            
        r12_list, r23_list, time_center = [], [], []
        for center in centers:
            ws = center - window_days/2
            we = center + window_days/2

            mask = (time_ext >= ws) & (time_ext < we)
            if np.sum(mask) < 3:
                r12_list.append(np.nan)
                r23_list.append(np.nan)
                time_center.append(center)
                continue
            
            Brs, Bts, Bns = Br_ext[mask], Bt_ext[mask], Bn_ext[mask]
            Br0, Bt0, Bn0 = np.mean(Brs), np.mean(Bts), np.mean(Bns)
            dB = np.column_stack((Brs-Br0, Bts-Bt0, Bns-Bn0))
            M = (dB.T @ dB) / len(dB)
            eigvals, _ = np.linalg.eigh(M)
            eigvals = np.sort(eigvals)[::-1]
            eps = 1e-12
            r12 = eigvals[0] / max(eigvals[1], eps)
            r23 = eigvals[1] / max(eigvals[2], eps)
            r12_list.append(r12)
            r23_list.append(r23)
            time_center.append(center)
        
        time_center_array = np.array(time_center)
        r12_array = np.array(r12_list)
        r23_array = np.array(r23_list)

        all_results[window_sec] = {
            'time': [jd_to_datetime(t) for t in time_center_array],
            'r12': r12_array,
            'r23': r23_array,
            'time_jd': time_center_array,
            'window_sec': window_sec,
            'overlap': overlap,
            'windows_count': len(centers)
        }

        if window_sec == target_window:
            for i, (t_jd, r12, r23) in enumerate(zip(time_center_array, r12_array, r23_array)):
                if r12 > r12_threshold and r23 > r23_threshold:
                    ws_hl = t_jd - window_days / 2
                    we_hl = t_jd + window_days / 2
                    
                    if keep_partial:
                        ws_hl = max(ws_hl, start)
                        we_hl = min(we_hl, end)
                        if (we_hl - ws_hl) * 86400 >= window_sec * 0.3:
                            highlight_intervals.append((ws_hl, we_hl))
                    else:
                        if ws_hl >= start and we_hl <= end:
                            highlight_intervals.append((ws_hl, we_hl))

    fig, axes = plt.subplots(len(window_sizes), 1, figsize=(12, 8), sharex=True)
    if len(window_sizes) == 1:
        axes = [axes]
    
    for idx, (ax, ws) in enumerate(zip(axes, window_sizes)):
        if ws not in all_results or len(all_results[ws]['r12']) == 0:
            continue
        d = all_results[ws]
        times = d['time']
        r12 = d['r12']
        r23 = d['r23']

        condition = (r12 > r12_threshold) & (r23 > r23_threshold)
        
        ax.step(times, r12, where='mid', color='steelblue', 
               linestyle='-', linewidth=1.5, alpha=1.0, label='r12')
        ax.step(times, r23, where='mid', color='seagreen', 
               linestyle='--', linewidth=1.5, alpha=1.0, label='r23')

        r12_good = np.where(condition, r12, np.nan)
        r23_good = np.where(condition, r23, np.nan)
        ax.step(times, r12_good, where='mid', color='black', 
               linestyle='-', linewidth=3, label=f'r12 > {r12_threshold}')
        ax.step(times, r23_good, where='mid', color='red', 
               linestyle='--', linewidth=3, label=f'r23 > {r23_threshold}')
        
        if np.any(condition):
            times_arr = np.array(times)
            ax.scatter(times_arr[condition], r12[condition], color='black', s=40, zorder=5)
            ax.scatter(times_arr[condition], r23[condition], color='red', s=40, zorder=5, marker='s')

        n_total = len(r12)
        n_good = np.sum(condition)

        stats = (f'{ws}s: 总窗口={n_total}, 高质量={n_good} ({100*n_good/n_total:.1f}%)\n'
                f'r12={np.mean(r12):.1f}±{np.std(r12):.1f}, r23={np.mean(r23):.1f}±{np.std(r23):.1f}\n'
                f'重叠率={overlap*100:.0f}%, 步长={ws*(1-overlap):.0f}s')
        
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.set_ylabel(f'{ws}s\nRatio', fontsize=10)
        ax.set_yscale('log')
        ax.set_ylim(0.5, 2000)
        ax.grid(True, alpha=0.3, which='both')
        ax.axhline(y=r12_threshold, color='blue', linestyle=':', alpha=0.5)
        ax.axhline(y=r23_threshold, color='green', linestyle=':', alpha=0.5)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=2)
        if idx < len(window_sizes) - 1:
            ax.tick_params(labelbottom=False)
            
    axes[-1].set_xlabel('Time', fontsize=12)
    axes[0].set_title(f'RED = High Quality (r12>{r12_threshold} & r23>{r23_threshold}) | '
                     f'窗口中心从start到end均匀分布 | Overlap={overlap*100:.0f}%', 
                     fontsize=14, color='darkred', fontweight='bold')
    plt.tight_layout()
    return fig, all_results, highlight_intervals
def MVAfind_draw(values0, start, end, overlap=0.75, keep_partial=True):
    fig_mva, all_results, highlight_intervals = draw_MVA_step(
        values0, start, end,
        window_sizes=[60, 90, 120, 180, 240],
        r12_threshold=10,
        r23_threshold=2,
        target_window=120,
        overlap=overlap,
        keep_partial=keep_partial) 
    print(f"\n{'='*50}")
    print(f"滑动窗口MVA分析完成")
    print(f"重叠率: {overlap*100:.0f}%")
    print(f"保留部分重叠区间: {'是' if keep_partial else '否'}")
    print(f"找到 {len(highlight_intervals)} 个高质量MVA区间 (r12>10 & r23>2)")
    print(f"{'='*50}")
    for start, end in highlight_intervals:
        draw_MVALMN(values0, start, end)
# ============================================================================
#小波分析阿尔芬性质
# ============================================================================
def draw_Alfenxiaobo(values0, start, end, vertical_lines=[2451545.0, 2451545.0]):
    time = values0['time_data']
    start_n = start #- 1/48
    end_n = end #+ 1/48
    time_mask = (time >= start_n) & (time <= end_n)
    u0, mp = 4*np.pi*1e-7, 1.6726e-27
    #小波变换前的数据准备
    w0, scale_num = 6.0, 32
    f_min, f_max = 64/((end_n-start_n)*86400.0), 1/values0['resolution']/4
    ratio = (f_max / f_min)**(1.0/(scale_num - 1))
    normalization = np.pi**(-0.25)  
    freqs = f_max * ratio**(-np.arange(scale_num)) 
    #scales = w0 / (2*np.pi) / freqs
    scales = 1.031 / freqs
    #拓展数据，两边各自拓展6s_max的范围用于边界处理
    s_max, support_sigma = np.max(scales), 6.0 
    half_support = support_sigma * s_max / 86400.0
    start_extended, end_extended = start_n - half_support, end_n + half_support
    time_extended_mask = (time >= start_extended ) & (time <= end_extended)    
    N = len(time[time_extended_mask])
    B_components = np.array([values0['b_data'][:, 0][time_extended_mask],\
        values0['b_data'][:, 1][time_extended_mask], values0['b_data'][:, 2][time_extended_mask]])
    B_T_ext = B_components * 1e-9
    n_m3_ext = values0['nep_data'][time_extended_mask] * 1e6
    rho_ext = n_m3_ext * mp
    b_ext = np.zeros_like(B_T_ext)
    for i in range(N):
        B_mag = np.sqrt(B_T_ext[0,i]**2 + B_T_ext[1,i]**2 + B_T_ext[2,i]**2)
        if B_mag > 0:
            VA_mag = B_mag / np.sqrt(u0 * rho_ext[i])
            b_ext[:,i] = B_T_ext[:,i] / B_mag * VA_mag / 1000
    V_components = np.array([values0['v_data'][:, 0][time_extended_mask],\
        values0['v_data'][:, 1][time_extended_mask], values0['v_data'][:, 2][time_extended_mask]])
    wave_coeffs_b = np.zeros((3, scale_num, N), dtype=complex)
    power_components_b = np.zeros((3, scale_num, N))
    wave_coeffs_v = np.zeros((3, scale_num, N), dtype=complex)
    power_components_v = np.zeros((3, scale_num, N))
    omega = 2 * np.pi * np.fft.fftfreq(N, values0['resolution'])
    for i in range(3): 
        B_hat = np.fft.fft(b_ext[i])
        V_hat = np.fft.fft(V_components[i])
        for l in range(scale_num):  
            s = scales[l] 
            psi_hat = np.sqrt(2*np.pi) * normalization * np.sqrt(s) * \
                      np.exp(-0.5 * (s * omega - w0)**2)
            psi_hat = np.where(omega > 0, psi_hat, 0)
            
            wave_coeffs_b[i, l] = np.fft.ifft(B_hat * psi_hat)
            power_components_b[i, l] = 2 * (np.abs(wave_coeffs_b[i, l])**2)
            
            wave_coeffs_v[i, l] = np.fft.ifft(V_hat * psi_hat)
            power_components_v[i, l] = 2 * (np.abs(wave_coeffs_v[i, l])**2)
    total_power_b_extended = np.nansum(power_components_b, axis=0)        
    total_power_v_extended = np.nansum(power_components_v, axis=0)
    s_mask = (time[time_extended_mask]>=start_n)&(time[time_extended_mask]<=end_n)
    total_power_v = total_power_v_extended[:, s_mask]
    total_power_b = total_power_b_extended[:, s_mask]
    gamma_A_ts = total_power_v / total_power_b 
    
    v_dot_b = 2 * np.real(np.sum(wave_coeffs_v * np.conj(wave_coeffs_b), axis=0))     
    denominator = total_power_v_extended + total_power_b_extended  
    σ_c_extended = np.where(denominator > 1e-15,
                        2 * v_dot_b / denominator, np.nan)
    σ_c = σ_c_extended[:, s_mask]
    
    σ_r_extended = (total_power_v_extended - total_power_b_extended) / (total_power_v_extended + total_power_b_extended + 1e-15)
    σ_r = σ_r_extended[:, s_mask]
    
    numerator_m = 2 * np.imag(np.conj(wave_coeffs_b[1]) * wave_coeffs_b[2])
    σ_m_extended = np.where(total_power_b_extended > 1e-15,
                            2 * numerator_m / total_power_b_extended,
                            np.nan)

    σ_m = σ_m_extended[:, s_mask]  
    #-----------------------------------
    time_select = Time(time[time_extended_mask][s_mask], format='jd').datetime
    fig = plt.figure(figsize=(12, 6), constrained_layout=False)
    axes = []
    positions = [
        [0.1, 0.68, 0.8, 0.30],  
        [0.1, 0.37, 0.8, 0.30],  
        [0.1, 0.06, 0.8, 0.30]  
    ]
    for pos in positions:
        axes.append(fig.add_axes(pos))

    for ax in axes[1:]:
        ax.set_xlabel('')
        ax.sharex(axes[0])
        ax.tick_params(labelbottom=False)
    axes[-1].tick_params(labelbottom=True) 
    #-----------------------------------
    time_select = Time(time[time_extended_mask][s_mask], format='jd').datetime
    im0 = axes[0].pcolormesh(time_select, freqs, σ_m, 
                           cmap='jet', shading='auto', vmin=-1.0, vmax=1.0)
    im1 = axes[1].pcolormesh(time_select, freqs, σ_c, 
                           cmap='jet', shading='auto', vmin=-1.0, vmax=1.0)
    im2 = axes[2].pcolormesh(time_select, freqs, σ_r, 
                           cmap='jet', shading='auto', vmin=-1.0, vmax=1.0)
    for i in range(3):
        axes[i].set_yscale('log')
        axes[i].set_ylim(freqs[-1], freqs[0])
        axes[i].set_ylabel('Frequency [Hz]', fontsize=9)
        axes[i].set_xlabel('Time [s]', fontsize=9)
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[i].xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=0, ha='right')
        axes[i].xaxis.set_major_locator(mdates.MinuteLocator(interval=30)) 
    
    pos = axes[0].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0+pos.height/10, 0.01, pos.height*0.8])
    cbar = fig.colorbar(im0, cax=cbar_ax)
    cbar.set_label('σ_m', fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    pos = axes[1].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0+pos.height/10, 0.01, pos.height*0.8])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('σ_c', fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    pos = axes[2].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0+pos.height/10, 0.01, pos.height*0.8])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label(r'σ_r', fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    start_Time, end_Time = Time(start, format='jd').datetime,\
            Time(end, format='jd').datetime
    vertical_lines = Time(vertical_lines, format='jd').datetime
    for ax in axes:
        ax.axvline(start_Time, color='black', linestyle='--', alpha=1.0, linewidth=2.0)
        ax.axvline(end_Time, color='black', linestyle='--', alpha=1.0, linewidth=2.0)
        for vl in vertical_lines:
            ax.axvline(vl, color='red', linestyle='--', alpha=1.0, linewidth=2.0)
    return σ_m, σ_c, σ_r
# ============================================================================
#CWT连续小波分析阿尔芬性质
# ============================================================================
def draw_CwtAlfensigma(values0, start, end, vertical_lines=[2451545.0, 2451545.0]):
    pass
# ============================================================================
#σ_c与σ_r二维图像
# ============================================================================
def draw_σ_cr_2Dfig(σ_m, σ_c, σ_r, time_data, vertical_lines=[2451545.0, 2451545.0]):
    """
    绘制σ_c-σ_r相空间散点图
    
    Parameters:
    -----------
    σ_m, σ_c, σ_r : 2D arrays [time, scale]
        三个小波变换参数
    time_data : 1D array
        时间数据（儒略日），长度应等于σ_m.shape[0]
    vertical_lines : list
        [HCS_start_time, HCS_end_time] 儒略日时间
    """
    from matplotlib.patches import Rectangle
    from astropy.time import Time
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 设置颜色和标记
    colors = {'before': 'blue', 'hcs': 'red', 'after': 'green'}
    markers = {'before': '^', 'hcs': 'o', 'after': 's'}
    alpha = 0.1
    point_size = 2
    
    # 获取HCS开始和结束时间
    t_start_hcs, t_end_hcs = vertical_lines
    
    # 转换时间为datetime用于比较
    time_dt = Time(time_data, format='jd').datetime
    t_start_hcs_dt = Time(t_start_hcs, format='jd').datetime
    t_end_hcs_dt = Time(t_end_hcs, format='jd').datetime
    
    # 创建时间掩码（每个时间点一个布尔值）
    time_before_mask = time_dt < t_start_hcs_dt
    time_hcs_mask = (time_dt >= t_start_hcs_dt) & (time_dt <= t_end_hcs_dt)
    time_after_mask = time_dt > t_end_hcs_dt
    
    # 打印时间范围确认
    print(f"时间范围: {time_data[0]:.5f} - {time_data[-1]:.5f}")
    print(f"HCS: {t_start_hcs:.5f} - {t_end_hcs:.5f}")
    print(f"前背景点数: {np.sum(time_before_mask)} 时间点")
    print(f"HCS点数: {np.sum(time_hcs_mask)} 时间点")
    print(f"后背景点数: {np.sum(time_after_mask)} 时间点")
    
    # 初始化存储每个区域的σ_c和σ_r
    data = {
        'before': {'sigma_c': [], 'sigma_r': []},
        'hcs': {'sigma_c': [], 'sigma_r': []},
        'after': {'sigma_c': [], 'sigma_r': []}
    }
    
    # 遍历所有时间和尺度
    n_times = σ_m.shape[0]
    n_scales = σ_c.shape[1]
    
    for i in range(n_times):  
        for j in range(n_scales): 
            # 检查是否有效（非NaN）
            if np.isnan(σ_c[i, j]) or np.isnan(σ_r[i, j]):
                continue
            
            # 根据时间分配到不同区域
            if time_before_mask[i]:
                data['before']['sigma_c'].append(σ_c[i, j])
                data['before']['sigma_r'].append(σ_r[i, j])
            elif time_hcs_mask[i]:
                data['hcs']['sigma_c'].append(σ_c[i, j])
                data['hcs']['sigma_r'].append(σ_r[i, j])
            elif time_after_mask[i]:
                data['after']['sigma_c'].append(σ_c[i, j])
                data['after']['sigma_r'].append(σ_r[i, j])
    
    # 转换为numpy数组
    for region in data:
        data[region]['sigma_c'] = np.array(data[region]['sigma_c'])
        data[region]['sigma_r'] = np.array(data[region]['sigma_r'])
        print(f"{region}: {len(data[region]['sigma_c'])} 个点 (时间点×尺度)")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制散点
    if len(data['before']['sigma_c']) > 0:
        ax.scatter(data['before']['sigma_c'], data['before']['sigma_r'],
                  c=colors['before'], marker=markers['before'], 
                  s=point_size, alpha=alpha, 
                  label=f'Upstream ({len(data["before"]["sigma_c"])} pts)',
                  rasterized=True)
    
    if len(data['hcs']['sigma_c']) > 0:
        ax.scatter(data['hcs']['sigma_c'], data['hcs']['sigma_r'],
                  c=colors['hcs'], marker=markers['hcs'], 
                  s=point_size, alpha=alpha, 
                  label=f'HCS Center ({len(data["hcs"]["sigma_c"])} pts)',
                  rasterized=True)
    
    if len(data['after']['sigma_c']) > 0:
        ax.scatter(data['after']['sigma_c'], data['after']['sigma_r'],
                  c=colors['after'], marker=markers['after'], 
                  s=point_size, alpha=alpha, 
                  label=f'Downstream ({len(data["after"]["sigma_c"])} pts)',
                  rasterized=True)
    
    # 添加矩形框
    rect = Rectangle((-0.4, -1.0), 0.8, 0.5,
                     linewidth=2, edgecolor='black',
                     facecolor='none', linestyle='--',
                     label='Flux Rope Region')
    ax.add_patch(rect)
    
    # 添加参考线
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 设置坐标轴
    ax.set_xlabel('σ_c', fontsize=12)
    ax.set_ylabel('σ_r', fontsize=12)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper right', fontsize=9)
    
    # 添加统计信息
    total_points = len(data['before']['sigma_c']) + len(data['hcs']['sigma_c']) + len(data['after']['sigma_c'])
    textstr = f'Total points: {total_points}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    return fig, ax, data
# ============================================================================
#σ_c与σ_r二维图像
# ============================================================================
def pushixu(values0, start, end, vertical_lines=[2451545.0, 2451545.0]):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from astropy.time import Time
    
    time = values0['time_data']
    t_start_hcs, t_end_hcs = vertical_lines
    segments = [
        {'name': 'Full Interval', 'start': start, 'end': end, 'color': 'black'},
        {'name': 'Upstream', 'start': start, 'end': t_start_hcs, 'color': 'blue'},
        {'name': 'HCS', 'start': t_start_hcs, 'end': t_end_hcs, 'color': 'red'},
        {'name': 'Downstream', 'start': t_end_hcs, 'end': end, 'color': 'green'}
    ]
    fig = plt.figure(figsize=(8, 6))
    positions = [
        [0.08, 0.55, 0.35, 0.35],  # Full Interval - 左上
        [0.55, 0.55, 0.35, 0.35],  # Upstream - 右上
        [0.08, 0.08, 0.35, 0.35],  # HCS - 左下
        [0.55, 0.08, 0.35, 0.35]   # Downstream - 右下
    ]
    axes = []
    for pos in positions:
        ax = fig.add_axes(pos)
        axes.append(ax)
    results = {}
    
    for idx, (segment, ax) in enumerate(zip(segments, axes)):
        seg_start, seg_end = segment['start'], segment['end']
        
        print(f"\n处理 {segment['name']} 区间: {seg_start:.5f} - {seg_end:.5f}")
        time_mask = (time >= seg_start) & (time <= seg_end)
        time_seg = time[time_mask]
        
        if len(time_seg) == 0:
            print(f"警告: {segment['name']} 区间无数据")
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
            results[segment['name']] = {'slope': np.nan, 'std_err': np.nan}
            continue
            
        print(f"数据点数: {len(time_seg)}")
        
        Br = values0['b_data'][:, 0][time_mask]
        Bt = values0['b_data'][:, 1][time_mask]
        Bn = values0['b_data'][:, 2][time_mask]
        Br0, Bt0, Bn0 = np.mean(Br), np.mean(Bt), np.mean(Bn)
        δBr, δBt, δBn = Br - Br0, Bt - Bt0, Bn - Bn0
        δBr_fft = np.fft.rfft(δBr)
        δBt_fft = np.fft.rfft(δBt)
        δBn_fft = np.fft.rfft(δBn)

        Pr = np.abs(δBr_fft)**2 * 2 / len(Br) * values0['resolution']
        Pt = np.abs(δBt_fft)**2 * 2 / len(Bt) * values0['resolution']
        Pn = np.abs(δBn_fft)**2 * 2 / len(Bn) * values0['resolution']
        P_total = Pr + Pt + Pn
        freq = np.fft.rfftfreq(len(Br), d=values0['resolution'])

        smooth_points = 7
        P_total_smooth = np.copy(P_total)
        half_points = smooth_points // 2
        
        for i in range(half_points, len(P_total) - half_points):
            P_total_smooth[i] = np.mean(P_total[i-half_points:i+half_points+1])
        for i in range(half_points):
            P_total_smooth[i] = np.mean(P_total[:i+half_points+1])
        for i in range(len(P_total) - half_points, len(P_total)):
            P_total_smooth[i] = np.mean(P_total[i-half_points:])

        freq_range = [0.01, 0.1]
        mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
        freq_fit = freq[mask]
        P_fit = P_total_smooth[mask]
        
        valid_mask = (P_fit > 0) & np.isfinite(P_fit) & (freq_fit > 0)
        freq_fit = freq_fit[valid_mask]
        P_fit = P_fit[valid_mask]

        if len(freq_fit) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.log10(freq_fit), np.log10(P_fit))
            print(f"拟合斜率: {slope:.3f} ± {std_err:.3f}")
        else:
            slope, intercept, r_value, std_err = np.nan, np.nan, np.nan, np.nan
            print("数据点不足，无法拟合")

        results[segment['name']] = {'slope': slope, 'std_err': std_err}

        freq_limitation = 1.0 / (len(Br) * values0['resolution'])
        ax.loglog(freq[freq >= freq_limitation], 
                 P_total_smooth[freq >= freq_limitation],
                 'k-', linewidth=1.5)

        if not np.isnan(slope):
            fit_freq = np.logspace(np.log10(freq_range[0]), 
                                  np.log10(freq_range[1]), 100)
            fit_P = 10**(intercept + slope * np.log10(fit_freq))
            ax.loglog(fit_freq, fit_P, 'r--', linewidth=2)
            ax.axvline(freq_range[0], color='gray', linestyle=':', alpha=0.7)
            ax.axvline(freq_range[1], color='gray', linestyle=':', alpha=0.7)

        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        ax.set_title(f'{segment["name"]}', fontsize=12, color=segment['color'])

        ax.set_xlabel('Frequency [Hz]', fontsize=10)
        ax.set_ylabel('Power Spectral Density [nT²/Hz]', fontsize=10)

        ax.grid(True, alpha=0.3, which='both')

        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)

        if not np.isnan(slope):
            text_str = f' α={slope:.2f} ± {std_err:.2f}'
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    start_time = Time(start, format='jd').datetime.strftime("%Y-%m-%d %H:%M")
    end_time = Time(end, format='jd').datetime.strftime("%Y-%m-%d %H:%M")
    fig.suptitle(f'Power Spectral Density Scaling Analysis\n{start_time} - {end_time}', 
                 fontsize=14, y=0.98)
    
    plt.show()

    print("\n" + "="*50)
    print("拟合结果总结:")
    print("="*50)
    for name, res in results.items():
        if not np.isnan(res['slope']):
            print(f"{name:12s}: α = {res['slope']:.3f} ± {res['std_err']:.3f}")
    
    return fig, results
def draw_lianhe(data, start_time, end_time, vertical_lines = [2451545.0, 2451545.0]):
    draw_3xiaoboandangle(data, start_time, end_time, vertical_lines = vertical_lines)
    pushixu(data, start_time, end_time, vertical_lines = vertical_lines)
    draw_Alfenxiaobo(data, start_time, end_time, vertical_lines = [front, rear])#直接小波计算有问题
# ============================================================================
#σ_c与σ_r二维图像
# ============================================================================
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
def find_hcs_boundaries_with_background(values0, start, end, 
                                        smooth_window=30,
                                        b_drop_threshold=0.2,
                                        angle_threshold=90,
                                        background_extension=0.2): 

    time = values0['time_data']
    time_mask = (time >= start) & (time <= end)
    time0 = time[time_mask]
    time0_dt = Time(time0, format='jd').datetime
    
    Br, Bt, Bn = values0['b_data'][:, 0][time_mask], \
                 values0['b_data'][:, 1][time_mask], \
                 values0['b_data'][:, 2][time_mask]
    
    B_mag = np.sqrt(Br**2 + Bt**2 + Bn**2)
    B_mag_smooth = gaussian_filter1d(B_mag, sigma=smooth_window/5)
    
    shear_angles = np.zeros(len(B_mag))
    for i in range(1, len(B_mag)-1):
        B1 = np.array([Br[i-1], Bt[i-1], Bn[i-1]])
        B2 = np.array([Br[i+1], Bt[i+1], Bn[i+1]])
        norm1 = np.linalg.norm(B1)
        norm2 = np.linalg.norm(B2)
        if norm1 > 0 and norm2 > 0:
            cos_angle = np.dot(B1, B2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            shear_angles[i] = np.arccos(cos_angle) * 180 / np.pi

    Br_smooth = gaussian_filter1d(Br, sigma=smooth_window/5)
    Br_polarity = np.sign(Br_smooth)

    polarity_changes = np.where(np.diff(Br_polarity) != 0)[0]

    if len(polarity_changes) > 0:
        center_idx = polarity_changes[0]
        
        n_points = len(B_mag)
        left_bg_idx = slice(0, n_points//4)
        right_bg_idx = slice(3*n_points//4, n_points)
        B_left_bg = np.median(B_mag[left_bg_idx])
        B_right_bg = np.median(B_mag[right_bg_idx])
        B_bg = (B_left_bg + B_right_bg) / 2
        
        left_boundary = 0
        threshold_up = B_bg * (1 - b_drop_threshold * 0.5)  
        for i in range(center_idx, 0, -1):
            if B_mag_smooth[i] > threshold_up and shear_angles[i] < angle_threshold * 0.7:
                left_boundary = i
                break
        if left_boundary == 0:
            left_boundary = max(0, center_idx - n_points//10)
        
        right_boundary = len(B_mag) - 1
        for i in range(center_idx, len(B_mag)):
            if B_mag_smooth[i] > threshold_up and shear_angles[i] < angle_threshold * 0.7:
                right_boundary = i
                break
        if right_boundary == len(B_mag) - 1:
            right_boundary = min(len(B_mag)-1, center_idx + n_points//10)

        hcs_start_idx = left_boundary
        hcs_end_idx = right_boundary
        
    else:
        print("未检测到明显极性反转，使用磁场下降和剪切角综合识别")

        B_norm = B_mag_smooth / np.max(B_mag_smooth)
        angle_norm = shear_angles / 180.0
        composite_score = (1 - B_norm) * angle_norm

        peak_idx = np.argmax(composite_score)

        threshold = np.max(composite_score) * 0.3
        hcs_start_idx = peak_idx
        hcs_end_idx = peak_idx

        for i in range(peak_idx, 0, -1):
            if composite_score[i] < threshold:
                hcs_start_idx = i
                break

        for i in range(peak_idx, len(composite_score)):
            if composite_score[i] < threshold:
                hcs_end_idx = i
                break

    hcs_start = time0[hcs_start_idx]
    hcs_end = time0[hcs_end_idx]

    cs_duration = hcs_end_idx - hcs_start_idx
    bg_extension = int(cs_duration * background_extension)
    
    bg_left_start = max(0, hcs_start_idx - bg_extension)
    bg_left_end = hcs_start_idx
    bg_right_start = hcs_end_idx
    bg_right_end = min(len(time0)-1, hcs_end_idx + bg_extension)
    
    print(f"\n{'='*50}")
    print(f"HCS识别结果:")
    print(f"{'='*50}")
    print(f"HCS起始时间: {Time(hcs_start, format='jd').iso}")
    print(f"HCS终止时间: {Time(hcs_end, format='jd').iso}")
    print(f"HCS持续时间: {(hcs_end - hcs_start)*24*60:.1f} 分钟")
    print(f"HCS厚度(点数): {hcs_end_idx - hcs_start_idx} 点")
    print(f"\n背景区间:")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(time0_dt, Br, 'r-', label='Br', alpha=0.7, linewidth=1)
    axes[0].plot(time0_dt, Bt, 'g-', label='Bt', alpha=0.7, linewidth=1)
    axes[0].plot(time0_dt, Bn, 'b-', label='Bn', alpha=0.7, linewidth=1)
    axes[0].axvline(x=time0_dt[hcs_start_idx], color='red', linewidth=2, linestyle='-', label='HCS边界')
    axes[0].axvline(x=time0_dt[hcs_end_idx], color='red', linewidth=2, linestyle='-')

    axes[0].axvspan(time0_dt[bg_left_start], time0_dt[bg_left_end], 
                    alpha=0.2, color='gray', label='背景区间')
    axes[0].axvspan(time0_dt[bg_right_start], time0_dt[bg_right_end], 
                    alpha=0.2, color='gray')
    axes[0].set_ylabel('B (nT)')
    axes[0].legend(loc='upper right', ncol=4)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('磁场三分量 (RTN坐标系)')

    ax2 = axes[1]
    ax2.plot(time0_dt, B_mag, 'k-', label='|B|', alpha=0.5, linewidth=1)
    ax2.plot(time0_dt, B_mag_smooth, 'b-', linewidth=2, label='|B|平滑')
    ax2.axvline(x=time0_dt[hcs_start_idx], color='red', linewidth=2, linestyle='-')
    ax2.axvline(x=time0_dt[hcs_end_idx], color='red', linewidth=2, linestyle='-')
    ax2.axvspan(time0_dt[bg_left_start], time0_dt[bg_left_end], alpha=0.2, color='gray')
    ax2.axvspan(time0_dt[bg_right_start], time0_dt[bg_right_end], alpha=0.2, color='gray')
    ax2.set_ylabel('|B| (nT)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(time0_dt, Br_smooth, 'r--', label='Br平滑', alpha=0.7)
    ax2_twin.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2_twin.set_ylabel('Br (nT)')
    ax2_twin.legend(loc='upper right')

    axes[2].plot(time0_dt, shear_angles, 'g-', label='剪切角', alpha=0.7)
    axes[2].axhline(y=angle_threshold, color='orange', linestyle='--', 
                    label=f'阈值 ({angle_threshold}°)')
    axes[2].axvline(x=time0_dt[hcs_start_idx], color='red', linewidth=2, linestyle='-')
    axes[2].axvline(x=time0_dt[hcs_end_idx], color='red', linewidth=2, linestyle='-')
    axes[2].axvspan(time0_dt[bg_left_start], time0_dt[bg_left_end], alpha=0.2, color='gray')
    axes[2].axvspan(time0_dt[bg_right_start], time0_dt[bg_right_end], alpha=0.2, color='gray')
    axes[2].set_ylabel('剪切角 (度)')
    axes[2].set_ylim(0, 180)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time0_dt, B_mag_smooth, 'b-', linewidth=2, label='|B|平滑')
    axes[3].fill_between(time0_dt[hcs_start_idx:hcs_end_idx], 
                         0, B_mag_smooth[hcs_start_idx:hcs_end_idx],
                         color='red', alpha=0.3, label='HCS区域')
    axes[3].axvline(x=time0_dt[hcs_start_idx], color='red', linewidth=2, linestyle='-')
    axes[3].axvline(x=time0_dt[hcs_end_idx], color='red', linewidth=2, linestyle='-')
    axes[3].axvspan(time0_dt[bg_left_start], time0_dt[bg_left_end], alpha=0.2, color='gray')
    axes[3].axvspan(time0_dt[bg_right_start], time0_dt[bg_right_end], alpha=0.2, color='gray')
    axes[3].set_ylabel('|B| (nT)')
    axes[3].set_xlabel('时间')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle(f'HCS自动识别结果 ({Time(start, format="jd").iso[:10]} 到 {Time(end, format="jd").iso[:10]})')
    plt.tight_layout()
    
    return hcs_start, hcs_end, fig
def analyze_hcs_with_mva(values0, hcs_start, hcs_end, bg_extension=0.2):
    print(f"\n{'='*50}")
    print(f"HCS详细MVA分析")
    print(f"{'='*50}")

    draw_MVALMN(values0, hcs_start, hcs_end)

    time = values0['time_data']
    time_mask = (time >= hcs_start) & (time <= hcs_end)
    time0 = time[time_mask]
    
    cs_duration = len(time0)
    bg_ext = int(cs_duration * bg_extension)

    hcs_start_idx = np.where(time == hcs_start)[0][0]
    hcs_end_idx = np.where(time == hcs_end)[0][0]
    
    bg_left_start = max(0, hcs_start_idx - bg_ext)
    bg_left_end = hcs_start_idx
    bg_left_time = time[bg_left_start]
    
    bg_right_start = hcs_end_idx
    bg_right_end = min(len(time)-1, hcs_end_idx + bg_ext)
    bg_right_time = time[bg_right_start]
    
    print(f"\n背景区间对比分析:")
    if bg_left_end > bg_left_start:
        print(f"\n左背景 MVA:")
        draw_MVALMN(values0, time[bg_left_start], time[bg_left_end])

    if bg_right_end > bg_right_start:
        print(f"\n右背景 MVA:")
        draw_MVALMN(values0, time[bg_right_start], time[bg_right_end])
# ============================================================================
if __name__ == "__main__":
    """
    #对磁场、等离子体、超热电子数据拼接
    start_date = "2024-09-26"
    end_date = "2024-10-05"
    src_dir: str = r"E:/science/20240926/"
    mag_data = mag_data_done(start_date, end_date, src_dir=src_dir)
    print("磁场数据总点数:", len(mag_data["time_B"]))
    plasma_data = plasma_data_done(start_date, end_date, src_dir=src_dir)
    print("等离子体数据总点数:", len(plasma_data["time_p"]))
    
    strahl_data = strahl_data_done(start_date, end_date, src_dir=src_dir)
    print("超热电子数据总点数:", len(strahl_data["time_s"]))
    
    data = mag_data | plasma_data | strahl_data
    pkl_save(data,dst_dir=src_dir,save_pkl = True)
    """
    with open("E:/science/20230923/data_yuchuli.pkl",'rb')as f:
        data = pickle.load(f)
    start_time = julian.to_jd(datetime(2023, 9, 27, 19, 8, 30), fmt = 'jd')
    end_time = julian.to_jd(datetime(2023, 9, 27, 20, 50, 30), fmt = 'jd')
    
    front = julian.to_jd(datetime(2024, 9, 30, 11, 52, 30), fmt = 'jd')
    rear  = julian.to_jd(datetime(2024, 9, 30, 12, 36, 30), fmt = 'jd')
    """
    #直观检查原始数据是否有太大数据缺失
    draw3vectors(data['time_B'], data['B3_rtn'], start_time, end_time,
                 labels=['|B|','Br', 'Bt', 'Bn'],ylabel='Magnetic Field [nT]',
                 xlabel='Time',title='PSP Magnetic Field')
    
    draw3vectors(data['time_p'], data['v3_rtn'], start_time, end_time,
                 labels=['|V|','Vr', 'Vt', 'Vn'],ylabel='Velocity[km/s]',
                 xlabel='Time',title='PSP Velocity')
    """
    
    is_correct, plasma_mask,_ = data_judge(data, start_time, end_time)
    if is_correct:
        insert_data(data, start_time, end_time, resolution = 0.2185)#插值为同轴同分辨率数据
        #draw_8yuchuli(data, start_time, end_time)
        #draw_4wave(data, start_time, end_time)
        #draw_12xiangguan(data, start_time, end_time)
        #calculate_num(data)#计算关键数据复现三个小波变换图
        #draw_9byangle(data, front, rear)
        #draw_7tuandong(data, start_time, end_time, vertical_lines = [front, rear])#用于观察HCS事件
        #draw_xiaobobysegment(data, front, rear)
        #draw_3xiaoboandangle(data, start_time, end_time, vertical_lines = [front, rear])#看压缩性用不上
        #draw_MVALMN(data, start_time, end_time)
        #analyze_my_hcs_event(data, start_time, end_time, window_sec=600, step_sec=60)
        #draw_jibozhenduan(data, front, rear, window_sec = 60, gap_sec = 60)
        #MVAfind_draw(data, front, rear, overlap=0.75, keep_partial=True)
        #σ_m, σ_c, σ_r=draw_Alfenxiaobo(data, start_time, end_time, vertical_lines = [front, rear])#直接小波计算有问题
        #draw_CwtAlfensigma(data, start_time, end_time, vertical_lines = [front, rear])
        #draw_σ_cr_2Dfig(σ_m, σ_c, σ_r, data['time_data'], vertical_lines = [front, rear])
        #draw_lianhe(data, start_time, end_time, vertical_lines = [front, rear])
        #hcs_start, hcs_end, fig = find_hcs_boundaries_with_background(data, start=start_time,end=end_time,smooth_window=30,b_drop_threshold=0.2,angle_threshold=90,background_extension=0.2)
    else :
        print('数据检验不合格，需要重新查找数据！') 
    