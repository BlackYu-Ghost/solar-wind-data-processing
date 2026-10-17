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
                            src_dir: str = r"E:/science/HCS/",
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
                          src_dir: str = r"E:/science/HCS/",
                          save_pkl: bool = True):
    fmt_day = "%Y-%m-%d"
    start = datetime.strptime(t_start, fmt_day)
    end   = datetime.strptime(t_end, fmt_day)
    days = (end - start).days + 1
    time_slots = [start + timedelta(days=d) for d in range(days)]
    total_days = len(time_slots)
    day_with_data = 0
    
    time_p, nproton, nprotonfit, tp, tp_fit= [], [], [], [], []
    v_x, v_y, v_z = [], [], []
    vup_xx, vup_yy, vup_zz = [], [], []
    vdn_xx, vdn_yy, vdn_zz = [], [], []
    vsc_xx, vsc_yy, vsc_zz = [], [], []
    vfit_xx, vfit_yy, vfit_zz = [], [], []
    rs_x, rs_y, rs_z = [], [], []
    carr_lat, carr_lon = [], []
    
    for slot in tqdm(time_slots, desc="L3i 处理进度", unit="天"):
        yyyymmdd = slot.strftime("%Y%m%d")
        pattern = f"psp_swp_spc_l3i_{yyyymmdd}_v*.cdf"
        #pattern = f"psp_swp_spi_sf00_l3_mom_{yyyymmdd}_v*.cdf"
        files = sorted(fnmatch.filter(os.listdir(src_dir), pattern))
        if not files:
            continue
        day_with_data += 1
        for fname in files:
            cdf_path = os.path.join(src_dir, fname)
            with CDF(cdf_path) as Data:
                epoch = Data['Epoch'][:]
                #general_flag = Data['general_flag'][:]
                #good = (general_flag == 0) | (general_flag != 0)
                good = np.full(len(epoch), True, dtype = bool)
                #if good.sum() == 0:
                #    continue
                jd = np.array([julian.to_jd(ts, fmt='jd') for ts in epoch[good]],
                              dtype='float64')
                
                vp_rtn = Data['vp_moment_RTN'][:][good] #数据更稳定
                #vp_rtn = Data['VEL_RTN_SUN'][:][good] #数据更稳定
                vfit   = Data['vp_fit_RTN'][:][good] #数据更精确但缺失值比较多
                #vfit   = Data['VEL_RTN_SUN'][:][good] #数据更精确但缺失值比较多
                np1    = Data['np_moment'][:][good]
                #np1    = Data['DENS'][:][good]
                tp1    = Data['wp_moment'][:][good]
                #tp1    = Data['TEMP'][:][good]
                
                np1fit = Data['np_fit'][:][good]
                tp1fit = Data['wp_fit'][:][good]
                
                vup    = Data['vp_moment_RTN_deltahigh'][:][good]
                vdn    = Data['vp_moment_RTN_deltalow'][:][good]
                vsc    = Data['vp_moment_SC'][:][good]
                rs     = Data['sc_pos_HCI'][:][good]
                lat    = Data['carr_latitude'][:][good]
                lon    = Data['carr_longitude'][:][good]
                '''
                np1fit = Data['TEMP'][:][good]
                tp1fit = Data['TEMP'][:][good]
                
                vup    = Data['VEL_RTN_SUN'][:][good] #数据更稳定
                vdn    = Data['VEL_RTN_SUN'][:][good] #数据更稳定
                vsc    = Data['VEL_RTN_SUN'][:][good] #数据更稳定
                rs     = Data['VEL_RTN_SUN'][:][good] #数据更稳定
                lat    = Data['TEMP'][:][good]
                lon    = Data['TEMP'][:][good]
                '''
                mask = (np1 != BadVal) & (tp1 != BadVal) & \
                       (vp_rtn[:, 0] != BadVal) & (vp_rtn[:, 1] != BadVal) & (vp_rtn[:, 2] != BadVal)
                if mask.sum() == 0:
                    continue
                time_p.append(jd[mask])
                nproton.append(np1[mask])
                nprotonfit.append(np1fit[mask])
                tp.append(tp1[mask])
                tp_fit.append(tp1fit[mask])
                v_x.append(vp_rtn[mask, 0])
                v_y.append(vp_rtn[mask, 1])
                v_z.append(vp_rtn[mask, 2])
                
                vup_xx.append(vup[mask, 0])
                vup_yy.append(vup[mask, 1])
                vup_zz.append(vup[mask, 2])
                vdn_xx.append(vdn[mask, 0])
                vdn_yy.append(vdn[mask, 1])
                vdn_zz.append(vdn[mask, 2])
                vsc_xx.append(vsc[mask, 0])
                vsc_yy.append(vsc[mask, 1])
                vsc_zz.append(vsc[mask, 2])
                vfit_xx.append(vfit[mask, 0])
                vfit_yy.append(vfit[mask, 1])
                vfit_zz.append(vfit[mask, 2])
                rs_x.append(rs[mask, 0])
                rs_y.append(rs[mask, 1])
                rs_z.append(rs[mask, 2])
                carr_lat.append(lat[mask])
                carr_lon.append(lon[mask])
                # 在处理每个文件时添加
                time_diff = np.diff(jd[mask])
                if len(time_diff) > 0:
                    print(f"文件 {fname}:")
                    print(f"  数据点数: {len(jd[mask])}")
                    print(f"  时间范围: {jd[mask][0]:.6f} 到 {jd[mask][-1]:.6f}")
                    print(f"  时间间隔统计:")
                    print(f"    最小: {time_diff.min()*86400:.3f} 秒")  # 转换为秒
                    print(f"    最大: {time_diff.max()*86400:.3f} 秒")
                    print(f"    平均: {time_diff.mean()*86400:.3f} 秒")
                
                
                
                
                
    if not time_p:
        print("⚠  该时段无有效数据，返回空字典")
        return {
            "time_p": np.empty(0),
            "v3_rtn": np.empty((0, 3)),
            "v3_sc": np.empty((0, 3)),
            "v3_fit": np.empty((0, 3)),
            "v3up_sc": np.empty((0, 3)),
            "v3dn_sc": np.empty((0, 3)),
            "nproton": np.empty(0),
            "nprotonfit": np.empty(0),
            "tp": np.empty(0),
            "rs3_hgi": np.empty((0, 3)),
            "carr_latitude": np.empty(0),
            "carr_longitude": np.empty(0),
            "tp_fit":np.empty(0)
        }
    time_p = np.concatenate(time_p)
    v3_rtn = np.column_stack((np.concatenate(v_x),
                              np.concatenate(v_y),
                              np.concatenate(v_z)))
    v3fit = np.column_stack((np.concatenate(vfit_xx),
                              np.concatenate(vfit_yy),
                              np.concatenate(vfit_zz)))
      
    v3up = np.column_stack((np.concatenate(vup_xx),
                            np.concatenate(vup_yy),
                            np.concatenate(vup_zz)))
    v3dn = np.column_stack((np.concatenate(vdn_xx),
                            np.concatenate(vdn_yy),
                            np.concatenate(vdn_zz)))
    v3sc = np.column_stack((np.concatenate(vsc_xx),
                            np.concatenate(vsc_yy),
                            np.concatenate(vsc_zz)))
    rs3 = np.column_stack((np.concatenate(rs_x),
                           np.concatenate(rs_y),
                           np.concatenate(rs_z)))
    data = {
        "time_p": time_p,
        "v3_rtn": v3_rtn,
        "v3_fit": v3fit,
        "nproton": np.concatenate(nproton),
        "nprotonfit": np.concatenate(nprotonfit),
        "tp": np.concatenate(tp),
        "tp_fit":np.concatenate(tp_fit),
        
        "v3_sc": v3sc,
        "v3up_sc": v3up,
        "v3dn_sc": v3dn,
        "rs3_hgi": rs3,
        "carr_latitude": np.concatenate(carr_lat),
        "carr_longitude": np.concatenate(carr_lon),
    }

    print(f"共 {total_days} 个日历天，其中 {day_with_data} 天有数据 "
             f"（占比 {day_with_data/total_days*100:.1f}%）")
    
    return data
# ============================================================================
#超热电子数据函数
# ============================================================================
def strahl_data_done(t_start: str, t_end: str, target_eV = 314,
                            src_dir: str = r"E:/science/HCS/",
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
def pkl_save(data,dst_dir: str = r"E:/science/HCS/",save_pkl = True):
    if save_pkl:
        pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)
        out_file = os.path.join(dst_dir,
                                "data_yuchuli_fuxian.pkl")
        with open(out_file, "wb") as f:
            pickle.dump(data, f)
        print(f"\n✅ 已保存  →  {out_file}")     
# ============================================================================
#数据处理函数-数据窗口有效性判断
# ============================================================================
def data_judge(data,window_start, window_end, revolution_B = 30, revolution_p = 30, fanwei=1):    
    yvliu = (window_end - window_start) / fanwei
    window_start-=yvliu
    window_end  +=yvliu
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
#数据处理函数-等离子体数据（质子密度、速度、温度）去毛刺
# ============================================================================
def spike_detect(data, max_angle=89, mask = None):#毛刺检测：利用质子密度异常突变
    if len(data['nproton']) < 3:
        print('数据太短，检测毛刺无意义！\n')
        return []
    spike_indices = []
    timestamps = np.array(data['time_p'], dtype=float)
    min_time = np.min(np.diff(timestamps))
    print(f"最小时间间隔是{min_time*86400:0.3f}s")
    if mask is None:
        indices_range = range(1, len(data['nproton']) - 1)
    else:
        valid_indices = np.where(mask)[0]
        if len(valid_indices) < 3:
            return []
        indices_range = valid_indices[1:-1] 
    for i in indices_range:
        dt1 = max(timestamps[i] - timestamps[i-1], min_time) * 86400
        dt2 = max(timestamps[i+1] - timestamps[i], min_time) * 86400
        dy1 = data['nproton'][i] - data['nproton'][i-1]
        dy2 = data['nproton'][i+1] - data['nproton'][i]
        
        angle1 = np.abs(np.arctan2(dy1, dt1) * 180 / np.pi)
        angle2 = np.abs(np.arctan2(dy2, dt2) * 180 / np.pi)
        if angle1 > max_angle and angle2 > max_angle:
            spike_indices.append(i)
    print(f"窗口内共检测到 {len(spike_indices)} 个毛刺位置")
    return spike_indices    
def spike_delete(values, spike_indices, 
                 window_before = 2, window_after = 2):
    if len(spike_indices) == 0:
        return values.copy(), print("没有毛刺，保持原数据！")
    new_values = values.copy()
    for spike_idx in spike_indices:
        left_idx = max(0, spike_idx - window_before)
        right_idx = min(len(values) - 1, spike_idx + window_after)
        
        while left_idx in spike_indices and left_idx > 0:
           left_idx -= 1
        while right_idx in spike_indices and right_idx < len(values) - 1:
           right_idx += 1
           
        left_val = values[left_idx]
        right_val = values[right_idx]
        num_points = right_idx - left_idx - 1
        if num_points > 0:
            for i in range(left_idx + 1, right_idx):
                t = (i - left_idx) / (right_idx - left_idx)
                new_values[i] = left_val + t * (right_val - left_val)
    return new_values, print("毛刺已经插值处理！")
def spike_done(data, mask = None):
    spike_loc = spike_detect(data, mask = mask)
    data['nproton_5mao'], _ = spike_delete(data['nproton'], spike_loc)
    data['v3_rtn_5mao'], _ = spike_delete(data['v3_rtn'], spike_loc)
    data['tp_5mao'], _ = spike_delete(data['tp'], spike_loc)
    return print("质子密度、速度、温度毛刺去除成功！")
# ============================================================================
#数据插值函数
# ============================================================================
def insert_data(data, start_ts, end_ts, resolution = 3, fanwei=100000000):
    yvliu = (end_ts - start_ts) / fanwei
    start_ts-=yvliu
    end_ts  +=yvliu
    num_points = int((end_ts - start_ts) * 86400 / resolution) + 1
    resample_time_data = start_ts + np.arange(num_points) * resolution / 86400 
    print("数据插值:")
    print(f"  原始磁场数据: {len(data['time_B'])} 点")
    print(f"  原始等离子体数据: {len(data['time_p'])} 点")
    print(f"  插值后数据: {num_points} 点 (分辨率: {resolution:.3f} 秒)")
    
    b_resampled = np.zeros((len(resample_time_data), 3))
    for i in range(3):
        b_resampled[:, i] = np.interp(resample_time_data, data['time_B'], data['B3_rtn'][:, i])
    
    v_resampled = np.zeros((len(resample_time_data), 3))    
    for i in range(3):
        v_resampled[:, i] = np.interp(resample_time_data, data['time_p'], data['v3_rtn'][:, i])
    nep_resampled = np.interp(resample_time_data, data['time_p'], data['nproton'])
    tp_resampled = np.interp(resample_time_data, data['time_p'], data['tp'])
    
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
def gap_mask(time, max_gap = 10 * 0.874 / 86400):
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
#毛刺去除前后图像对比
# ============================================================================
def draw_spike(values0, start, end):
    time_mask = (values0['time_p'] >= start) & (values0['time_p'] <= end)
    time = values0['time_p'][time_mask]
    time_dt = Time(time, format='jd').datetime
    values = values0['v3_rtn'][time_mask]
    value_total = np.linalg.norm(values, axis = 1)
    v_total_5mao = np.linalg.norm(values0['v3_rtn_5mao'], axis = 1)
    
    fig, axes = plt.subplots(3, 1, figsize = (12, 6), sharex = True)
    time_formatter = mdates.DateFormatter('%H:%M')
    colors, labels = ['k','r','g','b'], ['|V|', 'Vr', 'Vt', 'Vn'] 
    axes[0].set_title(f'PSP Unified Analysis: {time_dt[0]} to {time_dt[-1]}', 
                fontsize=12, fontweight='bold', pad=10) 
    axes[0].plot(time_dt, value_total, '-', linewidth = 0.8, alpha = 0.5,
               color = colors[0], label = labels[0])
    axes[0].plot(time_dt, v_total_5mao[time_mask], '-', linewidth = 0.8, alpha = 1.0,
               color = colors[0], label = labels[0])
    for i in range(3):
        axes[0].plot(time_dt, values[:, i], '-', linewidth = 0.8, 
                     alpha = 0.5, color = colors[i+1], label = labels[i+1])
    for i in range(3):
        axes[0].plot(time_dt, values0['v3_rtn_5mao'][:, i][time_mask], '-', linewidth = 0.8, 
                     alpha = 1.0, color = colors[i+1], label = labels[i+1])
    axes[0].set_ylabel('Velocity[km/s]', fontsize=10)
    axes[0].legend()
    
    axes[1].plot(time_dt, values0['nproton'][time_mask], '-', linewidth = 0.8, alpha = 0.5,
               color = 'darkorange', label = labels[0])
    axes[1].plot(time_dt, values0['nproton_5mao'][time_mask], '-', linewidth = 0.8, alpha = 1.0,
               color = 'darkorange', label = labels[0])
    axes[1].set_ylabel('Density[cm$^{-3}$]', fontsize=10)
    axes[1].legend()
    
    axes[2].plot(time_dt, values0['tp'][time_mask], '-', linewidth = 0.8, alpha = 0.5,
               color = 'purple', label = labels[0])
    axes[2].plot(time_dt, values0['tp_5mao'][time_mask], '-', linewidth = 0.8, alpha = 1.0,
               color = 'purple', label = labels[0])
    axes[2].set_ylabel('Temperature[km/s]', fontsize=10)
    axes[2].legend()
    axes[2].set_xlabel('Time')
    for ax in axes:
        ax.xaxis.set_major_formatter(time_formatter)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.08, hspace=0.15)
    plt.show()
# ============================================================================
#八图（基本量）
# ============================================================================
def draw_8yuchuli(values0, start, end):
    """绘制插值并去除等离子体毛刺的数据图（磁场、速度、密度、温度、
    目标超热电子时序图、超热电子颜色图，超热电子角度分布图，磁场幂律拟合）"""
    time_mask = (values0['time_data'] >= start) & (values0['time_data'] <= end)
    time = values0['time_data'][time_mask]
    time_dt = Time(time, format='jd').datetime
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
    axes[3].set_ylabel('Temperature\n[km/s]', fontsize=8)
    axes[3].legend(loc='upper right', fontsize=6)
    #-----------------------------------
    label3 = ['l', 'o', 'r']
    times_mask = (values0['time_s'] >= start) & (values0['time_s'] <= end)
    time_num = Time(values0['time_s'][times_mask], format='jd').datetime
    strahl3_total = np.sum(values0['strahl3'][times_mask], axis = 1)
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
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    axes[5].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[5].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
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

    Ca = 1e-9 * 1e-3 / np.sqrt(4 * np.pi * 1e-7 * 1.6726e-27 * nep * 1e+6)  # (N,)
    δB2v, δBr2v, δBt2v, δBn2v = Ca[:, np.newaxis]*δB, Ca * δBr, Ca * δBt, Ca * δBn
    δVr2b, δVt2b, δVn2b = δvr / Ca, δvt / Ca, δvn / Ca
    
    cc_r = np.corrcoef(δBr, δVr2b)[0, 1]
    cc_t = np.corrcoef(δBt, δVt2b)[0, 1]
    cc_n = np.corrcoef(δBn, δVn2b)[0, 1]
    cc_bp = np.corrcoef(δB_total/B_total0, δnep/nep0)[0, 1]
    
    σc = 2*np.mean(np.sum(δV*δB2v,axis=1))/(np.mean(np.sum(δV**2,axis=1))+np.mean(np.sum(δB2v**2,axis=1)))
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
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
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
def pushixu(values0, start, end, window=600, step=30):
    time = values0['time_data']
    time_mask = (values0['time_data'] >= start) & (values0['time_data'] <= end)
    time0 = values0['time_data'][time_mask]
    Br, Bt, Bn = values0['b_data'][:, 0], \
        values0['b_data'][:, 1], values0['b_data'][:, 2]
    windows_num = int((time0[-1]-time0[0]) * 86400.0 // step)
    slopes, start_times = [], []
    for i in range(windows_num+1):
        win_start = start + (i * step - (window-step)/2)/ 86400.0 
        win_end = win_start + window / 86400.0
        start_times.append(win_start+(window-step)/2/86400.0)
        mask = (time >= win_start) & (time <= win_end)
        Br_window, Bt_window, Bn_window = Br[mask], Bt[mask], Bn[mask]    
        Br0, Bt0, Bn0 = np.mean(Br_window), np.mean(Bt_window), np.mean(Bn_window)
        δBr, δBt, δBn  = Br_window - Br0, Bt_window - Bt0, Bn_window - Bn0

        δBr_fft, δBt_fft, δBn_fft = np.fft.rfft(δBr), np.fft.rfft(δBt), np.fft.rfft(δBn)

        Pr = np.abs(δBr_fft)**2 * 2 / len(Br_window) * values0['resolution']
        Pt = np.abs(δBt_fft)**2 * 2 / len(Bt_window) * values0['resolution']
        Pn = np.abs(δBn_fft)**2 * 2 / len(Bn_window) * values0['resolution']
        P_total = Pr + Pt + Pn
        freq = np.fft.rfftfreq(len(Br_window), d=values0['resolution'])
        
        smooth_points = 7
        P_total_smooth = np.copy(P_total)
        half_points = smooth_points // 2
        
        for j in range(half_points, len(P_total) - half_points):
            P_total_smooth[j] = np.mean(P_total[j-half_points:j+half_points+1])
        
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
            slopes.append(slope)
        else:slopes.append(np.nan)
    start_times.append(end) 
    start_times=Time(start_times, format='jd').datetime 
    return slopes, start_times
def draw_7tuandong(values0, start, end):
    time_mask = (values0['time_data'] >= start) & (values0['time_data'] <= end)
    time = values0['time_data'][time_mask]
    time_dt = Time(time, format='jd').datetime
    b_total = np.linalg.norm(values0['b_data'][time_mask], axis = 1)
    v_total = np.linalg.norm(values0['v_data'][time_mask], axis = 1)
    
    fig, axes = plt.subplots(7, 1, figsize = (8, 6))
    plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.15, left=0.1,right=0.9)
    #-----------------------------------
    colors, labels = ['k','r','g','b'], ['|B|', 'Br', 'Bt', 'Bn'] 
    axes[0].set_title(f'PSP Unified Analysis: {time_dt[0]} to {time_dt[-1]}', 
                fontsize=10, fontweight='bold',loc = 'center',  pad=10) 
    axes[0].plot(time_dt, b_total, '-', linewidth = 1.0, alpha = 1.0,
               color = colors[0], label = labels[0])
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.7)
    '''
    for i in range(3):
        axes[0].plot(time_dt, values0['b_data'][:, i][time_mask], '-', linewidth = 1.5, 
                     alpha = 1.0, color = colors[i+1], label = labels[i+1])
        '''
    axes[0].set_ylabel('Magnetic Field\n[nT]', fontsize=8)
    axes[0].legend(loc='upper right', fontsize=8)
    #-----------------------------------
    axes[1].plot(time_dt, values0['b_data'][:, 0][time_mask], '-', linewidth = 1.0, alpha = 1.0,
               color = colors[1], label = labels[1])
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.7)
    axes[1].set_ylabel('Magnetic Field\n[nT]', fontsize=8)
    axes[1].legend(loc='upper right', fontsize=8)
    #-----------------------------------
    slopes, starts = pushixu(data, start, end)
    slopes1, starts1 = pushixu(data, start, end, step=600)
    axes[4].step(starts,[0]+list(slopes), color='k', linewidth=1.0,  
                 label='par~perp', alpha=1, linestyle='-')
    axes[4].step(starts1,[0]+list(slopes1), color='green', linewidth=1.0,  
                 label='par~perp', linestyle='-', alpha=1.0)
    axes[4].axhline(-5/3, color='red', linestyle='--', alpha=0.7)
    axes[4].axhline(-3/2, color='blue', linestyle='--', alpha=0.7)
    axes[4].set_ylim(np.min(slopes)-0.1, np.max(slopes)+0.1)
    #axes[4].invert_yaxis()
    axes[4].set_ylabel('α', fontsize=9)
    axes[4].legend(loc='upper right', fontsize=8)
    #-----------------------------------
    axes[2].plot(time_dt, values0['nep_data'][time_mask], '-', linewidth = 1.0, alpha = 1.0,
               color = 'darkorange', label = 'Np')
    axes[2].set_ylabel('Density\n[cm$^{-3}$]', fontsize=8)
    axes[2].legend(loc='upper right', fontsize=8)
    #-----------------------------------
    axes[3].plot(time_dt, values0['tp_data'][time_mask], '-', linewidth = 1.0, alpha = 1.0,
               color = 'purple', label = 'Tem')
    axes[3].set_ylabel('Temperature\n[km/s]', fontsize=9)
    axes[3].legend(loc='upper right', fontsize=8)
    #-----------------------------------
    #Tp = m * wp**2 / 2 / kB, β = 2 * u0 * kB * Np * Tp / (|B|**2), k = 1.380649e-23
    kB, u0, mp = 1.380649e-23, 4*np.pi*1e-7, 1.6726e-27
    Tem = mp * (1000 * values0['tp_data'][time_mask])**2 / 2 / kB
    βp = 2 * u0 * values0['nep_data'][time_mask] * 1e6 * kB * Tem / ((b_total*1e-9)**2)
    axes[5].plot(time_dt, βp, 'k-', linewidth = 1.0, alpha = 1.0, label = 'βp')
    axes[5].axhline(5, color='red', linestyle='--', alpha=0.7)
    axes[5].axhline(1, color='blue', linestyle='--', alpha=0.7)
    #axes[5].set_ylim(0, 10)
    axes[5].set_yscale('log')
    axes[5].set_ylabel('β', fontsize=9)
    #-----------------------------------
    times_mask = (values0['time_s'] >= start) & (values0['time_s'] <= end)
    time_num = Time(values0['time_s'][times_mask], format='jd').datetime
    energy_target = (values0['strahl3'][times_mask, :, 1]).T
    log_flux = np.log10(energy_target)
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
    
    vertical_lines = [datetime(2024, 9, 26, 9, 36, 0),\
                      datetime(2024, 9, 26, 10, 9, 0)] 
    for ax in axes:
        for vl in vertical_lines:
            ax.axvline(vl, color='red', linestyle='--', alpha=1.0, linewidth=2.0)
    for ax in axes[:-1]:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)
    axes[-1].tick_params(labelbottom=True) 
    time_formatter = mdates.DateFormatter('%H:%M')
    
    for ax in axes[:-1]:
        ax.set_xlim(time_dt[0], time_dt[-1])
        ax.xaxis.set_major_formatter(time_formatter)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
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
    time_all = Time(time[time_extended_mask], format='jd').datetime
    time_select = Time(time[time_extended_mask][s_mask], format='jd').datetime
    im = axes[5].pcolormesh(time_all, freqs, np.log10(total_power_extended), 
                           cmap='jet', shading='auto')
    im = axes[0].pcolormesh(time_select, freqs, np.log10(total_power), 
                           cmap='jet', shading='auto')
    axes[5].axvline(x=Time(start, format='jd').datetime,\
                    color='k', linestyle='--', linewidth=2)
    axes[5].axvline(x=Time(end, format='jd').datetime,\
                    color='k', linestyle='--', linewidth=2)
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
    axes[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=45)) 
    
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
def draw_3xiaoboandangle(values0, start, end):
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
# ============================================================================
#最小方差混合MVA建立LMN坐标系判断HCS
# ============================================================================
"""
def draw_MVALMN(values0, start, end, cut_windows=None):
    time = values0['time_data']
    mask = (time >= start) & (time <= end)
    time0 = time[mask].copy()
    Br, Bt, Bn = values0['b_data'][:, 0][mask].copy(), \
                 values0['b_data'][:, 1][mask].copy(), \
                 values0['b_data'][:, 2][mask].copy()
    Vr, Vt, Vn = values0['v_data'][:, 0][mask].copy(), \
                 values0['v_data'][:, 1][mask].copy(), \
                 values0['v_data'][:, 2][mask].copy()
    if cut_windows is not None:
        for cut_start, cut_end in cut_windows:
            if cut_start >= time0[-1] or cut_end <= time0[0]:
                print(f"警告：切除窗口 [{cut_start}, {cut_end}] 不在当前时间范围内，跳过")
                continue
            mask_before = time0 < cut_start
            mask_after  = time0 > cut_end
            if not np.any(mask_before) and not np.any(mask_after):
                print(f"错误：切除窗口 [{cut_start}, {cut_end}] 移除了全部数据，跳过")
                continue
            time_before = time0[mask_before]
            Br_before, Bt_before, Bn_before = Br[mask_before], Bt[mask_before], Bn[mask_before]
            Vr_before, Vt_before, Vn_before = Vr[mask_before], Vt[mask_before], Vn[mask_before]

            time_after  = time0[mask_after]
            Br_after, Bt_after, Bn_after = Br[mask_after], Bt[mask_after], Bn[mask_after]
            Vr_after, Vt_after, Vn_after = Vr[mask_after], Vt[mask_after], Vn[mask_after]
            if len(time_before) > 0 and len(time_after) > 0:
                dt = np.median(np.diff(time0))
                new_time_after = time_before[-1] + dt * (1 + np.arange(len(time_after)))
                time_new = np.concatenate([time_before, new_time_after])
                Br = np.concatenate([Br_before, Br_after])
                Bt = np.concatenate([Bt_before, Bt_after])
                Bn = np.concatenate([Bn_before, Bn_after])
                Vr = np.concatenate([Vr_before, Vr_after])
                Vt = np.concatenate([Vt_before, Vt_after])
                Vn = np.concatenate([Vn_before, Vn_after])
                time0 = time_new
            elif len(time_before) > 0: 
                time0 = time_before
                Br, Bt, Bn = Br_before, Bt_before, Bn_before
                Vr, Vt, Vn = Vr_before, Vt_before, Vn_before
            else:                    
                time0 = time_after
                Br, Bt, Bn = Br_after, Bt_after, Bn_after
                Vr, Vt, Vn = Vr_after, Vt_after, Vn_after

            print(f"已切除 [{cut_start}, {cut_end}]，剩余点数: {len(time0)}")
    if len(time0) == 0:
        print("错误：无有效数据，退出")
        return
    time0_dt = Time(time0, format='jd').datetime
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
    if e_N[0] > 0:
        e_N = -e_N
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
    B_M_med = np.median(np.abs(B_M))
    guide_field_ratio = B_M_med / B_L_max if B_L_max > 0 else 0
    V_L_mean_val = np.mean(V_L)
    V_L_std_val = np.std(V_L)

    print(f"特征值: {eigenvalues}")
    print(f"r12 = {r12:.2f}, r23 = {r23:.2f}")
    print(f"L: {e_L}, M: {e_M}, N: {e_N}")
    print(f"导向场比: {guide_field_ratio:.4f}")
    print(f"V_L = {V_L_mean_val:.2f} ± {V_L_std_val:.2f} km/s")
    if r23 >= 10:
        print("✅ MVA质量良好")
    else:
        print(f"⚠️ MVA质量一般 (r23={r23:.1f})")

    if guide_field_ratio < 0.1:
        print("✅ 弱导向场")
    elif guide_field_ratio < 0.3:
        print("⚠️ 中等导向场")
    else:
        print("⚠️ 强导向场")
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    # 子图1：RTN 磁场
    axes[0].plot(time0_dt, Br, 'r-', label='Br')
    axes[0].plot(time0_dt, Bt, 'g-', label='Bt')
    axes[0].plot(time0_dt, Bn, 'b-', label='Bn')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('B in RTN (nT)', fontsize=9)
    axes[0].legend()
    axes[0].grid(True)

    # 子图2：LMN 磁场
    axes[1].plot(time0_dt, B_L, 'r-', label='B_L')
    axes[1].plot(time0_dt, B_M, 'g-', label='B_M')
    axes[1].plot(time0_dt, B_N, 'b-', label='B_N')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('B in LMN (nT)', fontsize=9)
    axes[1].legend()
    axes[1].grid(True)

    # 子图3：RTN 速度
    axes[2].plot(time0_dt, Vr, 'r-', label='Vr')
    axes[2].plot(time0_dt, Vt, 'g-', label='Vt')
    axes[2].plot(time0_dt, Vn, 'b-', label='Vn')
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('V in RTN (km/s)', fontsize=9)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # 子图4：LMN 速度扰动
    axes[3].plot(time0_dt, V_L, 'r-', label='V_L')
    axes[3].plot(time0_dt, V_M, 'g-', label='V_M')
    axes[3].plot(time0_dt, V_N, 'b-', label='V_N')
    axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[3].set_ylabel('V in LMN (km/s)', fontsize=9)
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)

    axes[-1].set_xlim([time0_dt[0], time0_dt[-1]])
    axes[-1].set_xlabel('Time', fontsize=9)
    plt.tight_layout()
    plt.show()
"""
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
    if r23 >= 10:
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
#按不同尺度寻找良好的MVA分析区间
# ============================================================================
def jd_to_datetime(jd):
    jd2000 = 2451545.0
    days_since_2000 = jd - jd2000
    return datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=float(days_since_2000))
def draw_MVA_step(values0, start, end, window_sizes=[60, 90, 120, 180, 240],
                                  r12_threshold=10, r23_threshold=10,
                                  target_window=120):
    time = values0['time_data']
    time_mask = (time >= start) & (time <= end)
    time0 = time[time_mask]
    Br = values0['b_data'][:, 0][time_mask]
    Bt = values0['b_data'][:, 1][time_mask]
    Bn_field = values0['b_data'][:, 2][time_mask]
    total_seconds = (end - start) * 86400.0
    all_results = {}
    highlight_intervals = []  

    for window_sec in window_sizes:
        window_days = window_sec / 86400.0
        windows_num = int(np.floor(total_seconds / window_sec))
        
        r12_list, r23_list, time_center = [], [], []
        for j in range(windows_num):
            ws = start + j * window_days
            we = ws + window_days
            mask = (time0 >= ws) & (time0 < we)
            if np.sum(mask) < 3:
                continue
            
            Brs, Bts, Bns = Br[mask], Bt[mask], Bn_field[mask]
            Br0, Bt0, Bn0 = np.mean(Brs), np.mean(Bts), np.mean(Bns)
            dB = np.column_stack((Brs-Br0, Bts-Bt0, Bns-Bn0))
            M = (dB.T @ dB) / len(dB)
            eigvals, _ = np.linalg.eigh(M)
            eigvals = np.sort(eigvals)[::-1]
            eps = 1e-12
            r12 = min(eigvals[0] / max(eigvals[1], eps), 1000)
            r23 = min(eigvals[1] / max(eigvals[2], eps), 1000)
            r12_list.append(r12)
            r23_list.append(r23)
            time_center.append((ws + we) / 2)
        all_results[window_sec] = {
            'time': [jd_to_datetime(t) for t in time_center],
            'r12': np.array(r12_list),
            'r23': np.array(r23_list),
            'time_jd': np.array(time_center)
        }
        if window_sec == target_window:
            for i, (t_jd, r12, r23) in enumerate(zip(time_center, r12_list, r23_list)):
                if r12 > r12_threshold and r23 > r23_threshold:
                    ws = t_jd - window_days / 2
                    we = t_jd + window_days / 2
                    highlight_intervals.append((ws, we))

    fig, axes = plt.subplots(len(window_sizes), 1, figsize=(16, 14), sharex=True)
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
            ax.scatter(times_arr[condition], r12[condition], color='red', s=40, zorder=5)
            ax.scatter(times_arr[condition], r23[condition], color='red', s=40, zorder=5, marker='s')

        n_total = len(r12)
        n_good = np.sum(condition)
        stats = (f'{ws}s: 总={n_total}, 高质量={n_good} ({100*n_good/n_total:.1f}%)\n'
                f'r12={np.mean(r12):.1f}±{np.std(r12):.1f}, r23={np.mean(r23):.1f}±{np.std(r23):.1f}')
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        ax.set_ylabel(f'{ws}s\nRatio', fontsize=10)
        ax.set_yscale('log')
        ax.set_ylim(0.5, 200)
        ax.grid(True, alpha=0.3, which='both')
        ax.axhline(y=r12_threshold, color='blue', linestyle=':', alpha=0.5)
        ax.axhline(y=r23_threshold, color='green', linestyle=':', alpha=0.5)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=2)
        if idx < len(window_sizes) - 1:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel('Time', fontsize=12)
    axes[0].set_title(f'RED = High Quality (r12>{r12_threshold} & r23>{r23_threshold}) | '
                     f'{target_window}s highlights for LMN', 
                     fontsize=14, color='darkred', fontweight='bold')
    plt.tight_layout()
    return fig, all_results, highlight_intervals
def MVAfind_draw(values0, start, end):
    fig_mva, all_results, highlight_intervals = draw_MVA_step(
        data, start_time, end_time,
        window_sizes=[60, 90, 120, 180, 240],
        r12_threshold=10,
        r23_threshold=10,
        target_window=120  
        )
    for start, end in highlight_intervals:
        draw_MVALMN(values0, start, end)
# ============================================================================
#独立计算激波坐标系诊断慢激波边界，验证Petschek重联
# ============================================================================
def draw_wavelet_3sigma(data, start=None, end=None, 
                        icme_start=None, icme_end=None,
                        sfr_start=None, sfr_end=None,
                        title='Wavelet Analysis'):



    Br, Bt, Bn = data['b_data'][:, 0], data['b_data'][:, 1], data['b_data'][:, 2]
    vr, vt, vn = data['v_data'][:, 0], data['v_data'][:, 1], data['v_data'][:, 2]
    time, nep = data['time_data'], data['nep_data']
    
    # 如果指定了时间范围，则截取数据
    if start is not None and end is not None:
        mask = (time >= start) & (time <= end)
        if not np.any(mask):
            print('⚠ 时间区间内无有效数据，使用全部数据')
        else:
            Br, Bt, Bn = Br[mask], Bt[mask], Bn[mask]
            vr, vt, vn = vr[mask], vt[mask], vn[mask]
            time = time[mask]
            nep = nep[mask]
            print(f"  截取时间范围: {start} - {end}")
    
    # 计算扰动（减去均值）
    Br0, Bt0, Bn0 = np.mean(Br), np.mean(Bt), np.mean(Bn)
    δBr, δBt, δBn = Br - Br0, Bt - Bt0, Bn - Bn0
    
    vr0, vt0, vn0 = np.mean(vr), np.mean(vt), np.mean(vn)
    δvr, δvt, δvn = vr - vr0, vt - vt0, vn - vn0
    
    # 计算阿尔文速度转换系数
    Ca = 1e-9 * 1e-3 / np.sqrt(4 * np.pi * 1e-7 * 1.6726e-27 * \
                               np.mean(nep) * 1e+6)
    δBr2v, δBt2v, δBn2v = Ca * δBr, Ca * δBt, Ca * δBn
    
    # 计算埃尔萨瑟变量
    zrplus, ztplus, znplus = δvr + δBr2v, δvt + δBt2v, δvn + δBn2v
    zrminus, ztminus, znminus = δvr - δBr2v, δvt - δBt2v, δvn - δBn2v
    
    # 小波变换参数设置
    wavelet = 'cmor'
    f_c = pywt.central_frequency(wavelet)
    min_period = 32     # 最小周期 0.5 分钟
    max_period = 960     # 最大周期 15 分钟
    scale_min, scale_max = min_period*60*f_c/data['resolution'], max_period*60*f_c/data['resolution']
    scales = np.linspace(scale_min, scale_max, 32)
    print(scales)
    # 计算周期（分钟）
    periods = scales * data['resolution'] / f_c / 60
    
    print(f"  尺度范围: {scale_min:.2f} - {scale_max:.2f}")
    print(f"  周期范围: {periods[-1]:.1f} - {periods[0]:.1f} 分钟")
    print(f"  时间点数: {len(time)}")
    
    # 磁场小波变换
    WBr, _ = pywt.cwt(Br, scales, wavelet, data['resolution'])
    WBt, _ = pywt.cwt(Bt, scales, wavelet, data['resolution'])
    WBn, _ = pywt.cwt(Bn, scales, wavelet, data['resolution'])
    WB_total = (np.abs(WBr)**2 + np.abs(WBt)**2 + np.abs(WBn)**2)
    
    # 埃尔萨瑟变量小波变换（z+）
    Wzrplus, _ = pywt.cwt(zrplus, scales, wavelet, data['resolution'])
    Wztplus, _ = pywt.cwt(ztplus, scales, wavelet, data['resolution'])
    Wznplus, _ = pywt.cwt(znplus, scales, wavelet, data['resolution'])
    Wzplus_total = (np.abs(Wzrplus)**2 + np.abs(Wztplus)**2 + np.abs(Wznplus)**2)
    
    # 埃尔萨瑟变量小波变换（z-）
    Wzrminus, _ = pywt.cwt(zrminus, scales, wavelet, data['resolution'])
    Wztminus, _ = pywt.cwt(ztminus, scales, wavelet, data['resolution'])
    Wznminus, _ = pywt.cwt(znminus, scales, wavelet, data['resolution'])
    Wzminus_total = (np.abs(Wzrminus)**2 + np.abs(Wztminus)**2 + np.abs(Wznminus)**2)
    
    # 计算σ参数（避免除零）
    fenmu1 = np.where(WB_total > 1e-30, WB_total, 1e-30)
    fenmu2 = np.where((Wzplus_total + Wzminus_total) > 1e-30, 
                      (Wzplus_total + Wzminus_total), 1e-30)
    
    σm = -2 * np.imag((np.conj(WBt) * WBn)) / fenmu1
    σc = (Wzplus_total - Wzminus_total) / fenmu2
    σr = 2 * np.real((np.conj(Wzrplus) * Wzrminus + 
                      np.conj(Wztplus) * Wztminus +
                      np.conj(Wznplus) * Wznminus)) / fenmu2
    
    print("小波变换计算完成！")
    
    # ==================== 绘图部分 ====================
    print("正在绘图...")
    
    time_dt = Time(time, format='jd').datetime
    
    # 创建3个子图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    plt.subplots_adjust(top=0.95, bottom=0.08, hspace=0.15, 
                       left=0.1, right=0.92)
    
    # 创建网格用于 pcolormesh
    T, P = np.meshgrid(time_dt, periods)
    
    # 子图1：σ_m
    im1 = axes[0].pcolormesh(T, P, σm, cmap='jet', shading='auto', 
                             vmin=-1, vmax=1)
    axes[0].set_yscale('log')
    axes[0].set_ylim(periods[-1], periods[0])
    axes[0].set_ylabel('Period (min)', fontsize=12)
    axes[0].set_title('$\sigma_m$', fontsize=14, pad=10)
    axes[0].yaxis.set_major_locator(LogLocator(base=2, numticks=8))
    axes[0].yaxis.set_major_formatter(LogFormatterMathtext(base=2))
    axes[0].tick_params(labelbottom=False)
    axes[0].grid(True, alpha=0.2, linestyle='--')
    
    # 子图2：σ_c
    im2 = axes[1].pcolormesh(T, P, σc, cmap='jet', shading='auto', 
                             vmin=-1, vmax=1)
    axes[1].set_yscale('log')
    axes[1].set_ylim(periods[-1], periods[0])
    axes[1].set_ylabel('Period (min)', fontsize=12)
    axes[1].set_title('$\sigma_c$', fontsize=14, pad=10)
    axes[1].yaxis.set_major_locator(LogLocator(base=2, numticks=8))
    axes[1].yaxis.set_major_formatter(LogFormatterMathtext(base=2))
    axes[1].tick_params(labelbottom=False)
    axes[1].grid(True, alpha=0.2, linestyle='--')
    
    # 子图3：σ_r
    im3 = axes[2].pcolormesh(T, P, σr, cmap='jet', shading='auto', 
                             vmin=-1, vmax=1)
    axes[2].set_yscale('log')
    axes[2].set_ylim(periods[-1], periods[0])
    axes[2].set_ylabel('Period (min)', fontsize=12)
    axes[2].set_title('$\sigma_r$', fontsize=14, pad=10)
    axes[2].set_xlabel('Time (UTC)', fontsize=12)
    axes[2].yaxis.set_major_locator(LogLocator(base=2, numticks=8))
    axes[2].yaxis.set_major_formatter(LogFormatterMathtext(base=2))
    axes[2].grid(True, alpha=0.2, linestyle='--')
    
    # 添加颜色条
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation='vertical', 
                         pad=0.02, aspect=30)
    cbar2 = fig.colorbar(im2, ax=axes[1], orientation='vertical', 
                         pad=0.02, aspect=30)
    cbar3 = fig.colorbar(im3, ax=axes[2], orientation='vertical', 
                         pad=0.02, aspect=30)
    
    cbar1.set_label('$\sigma_m$', fontsize=10)
    cbar2.set_label('$\sigma_c$', fontsize=10)
    cbar3.set_label('$\sigma_r$', fontsize=10)
    cbar1.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar2.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar3.set_ticks([-1, -0.5, 0, 0.5, 1])
    
    # 添加ICME和SFR标记（如果有）
    if icme_start is not None and icme_end is not None:
        icme_dt_start = Time(icme_start, format='jd').datetime
        icme_dt_end = Time(icme_end, format='jd').datetime
        for ax in axes:
            ax.axvspan(icme_dt_start, icme_dt_end, alpha=0.2, color='gray')
    
    if sfr_start is not None and sfr_end is not None:
        sfr_dt_start = Time(sfr_start, format='jd').datetime
        sfr_dt_end = Time(sfr_end, format='jd').datetime
        for ax in axes:
            ax.axvspan(sfr_dt_start, sfr_dt_end, alpha=0.2, color='yellow')
    
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.show()
    print("绘图完成！")


# 使用示例：
# draw_wavelet_3sigma(data)
# draw_wavelet_3sigma(data, start=2459000.0, end=2459005.0)
# draw_wavelet_3sigma(data, icme_start=2459001.0, icme_end=2459002.0)
def draw_3waveletsigma_cmor(values0, start, end, line_color=False):
    time0 = values0['time_data']
    time_mask = (time0 >= start) & (time0 <= end)
    if not np.any(time_mask):
        print('⚠  时间区间内无有效数据，无法绘图')
        return

    dt = values0['resolution']
    N_target = np.sum(time_mask)
    scale_num = 64
    
    wavelet = 'cmor'
    f_c = pywt.central_frequency(wavelet)
    
    # ========== 你的尺度计算方式 ==========
    min_period = 32   
    max_period = 960    
    freqs = np.logspace(np.log10(1/(min_period*60)), np.log10(1/(max_period*60)), scale_num)
    scales = f_c / (freqs * dt)
    print(scales)
    # ========== 扩展窗口（减少边界效应）==========
    support_sigma = 4.0
    s_max = np.max(scales)
    half_support = support_sigma * s_max / 86400.0
    start_extended = start - half_support
    end_extended = end + half_support
    
    ext_mask = (time0 >= start_extended) & (time0 <= end_extended)
    if np.sum(ext_mask) < 2:
        print('⚠  扩展后数据点过少，无法绘图')
        return
    
    # ========== 提取扩展窗口数据 ==========
    time_ext = time0[ext_mask]
    B_ext = values0['b_data'][ext_mask]      # (N, 3)
    V_ext = values0['v_data'][ext_mask]      # (N, 3)
    nep_ext = values0['nep_data'][ext_mask] * 1e6  # (N,)
    
    # ========== 计算阿尔芬速度（时变密度）==========
    Ca_ext = 1e-9 * 1e-3 / np.sqrt(4 * np.pi * 1e-7 * 1.6726e-27 * nep_ext)
    Ca_ext = Ca_ext[:, np.newaxis]  # (N, 1)
    
    # 减去均值（扰动分量）
    V_mean = np.mean(V_ext, axis=0)  # (3,)
    B_mean = np.mean(B_ext, axis=0)  # (3,)
    
    V_fluct = V_ext - V_mean  # (N, 3)
    B_fluct = B_ext - B_mean  # (N, 3)
    
    # ========== 埃尔萨瑟变量 ==========
    Z_plus = V_fluct + Ca_ext * B_fluct   # (N, 3)
    Z_minus = V_fluct - Ca_ext * B_fluct  # (N, 3)
    
    # ========== 小波变换 ==========
    # 磁场小波变换
    wave_B = np.zeros((3, scale_num, len(time_ext)), dtype=complex)
    for i in range(3):
        wave_B[i], freq_cwt = pywt.cwt(B_ext[:, i], scales, wavelet, sampling_period=dt)
    
    # z+ 小波变换
    wave_zplus = np.zeros((3, scale_num, len(time_ext)), dtype=complex)
    for i in range(3):
        wave_zplus[i], _ = pywt.cwt(Z_plus[:, i], scales, wavelet, sampling_period=dt)
    
    # z- 小波变换
    wave_zminus = np.zeros((3, scale_num, len(time_ext)), dtype=complex)
    for i in range(3):
        wave_zminus[i], _ = pywt.cwt(Z_minus[:, i], scales, wavelet, sampling_period=dt)
    
    # ========== 计算总功率 ==========
    power_B = np.sum(np.abs(wave_B)**2, axis=0)           # (scale_num, N)
    power_zplus = np.sum(np.abs(wave_zplus)**2, axis=0)   # (scale_num, N)
    power_zminus = np.sum(np.abs(wave_zminus)**2, axis=0) # (scale_num, N)
    
    # ========== 计算三个归一化参数 ==========
    # σ_m: 磁场椭圆度
    denom_B = power_B
    sigma_m = np.where(denom_B > 0, 
                       2 * np.imag(np.conj(wave_B[1]) * wave_B[2]) / denom_B, 
                       np.nan)
    
    # σ_c: 归一化交叉螺旋度
    denom_Z = power_zplus + power_zminus
    sigma_c = np.where(denom_Z > 0, 
                       (power_zplus - power_zminus) / denom_Z, 
                       np.nan)
    
    # σ_r: 归一化剩余螺旋度
    cross_term = (np.conj(wave_zplus[0]) * wave_zminus[0] +
                  np.conj(wave_zplus[1]) * wave_zminus[1] +
                  np.conj(wave_zplus[2]) * wave_zminus[2])
    sigma_r = np.where(denom_Z > 0, 
                       2 * np.real(cross_term) / denom_Z, 
                       np.nan)
    
    # ========== 绘图 ==========
    s_mask = (time_ext >= start) & (time_ext <= end)
    time_select = Time(time_ext[s_mask], format='jd').datetime
    
    # 计算周期（分钟）用于y轴
    periods = scales * dt / f_c / 60  # 分钟
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    plt.subplots_adjust(top=0.95, bottom=0.08, hspace=0.15, left=0.12, right=0.92)
    
    # 创建网格
    T, P = np.meshgrid(time_select, periods)
    
    # 子图1：σ_m
    im1 = axes[0].pcolormesh(T, P, sigma_m[:, s_mask], 
                             cmap='jet', shading='auto', vmin=-1, vmax=1)
    axes[0].set_yscale('log')
    axes[0].set_ylim(periods[-1], periods[0])
    axes[0].set_ylabel('Period (min)', fontsize=12)
    axes[0].set_title(r'$\sigma_m$', fontsize=14)
    axes[0].tick_params(labelbottom=False)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # 子图2：σ_c
    im2 = axes[1].pcolormesh(T, P, sigma_c[:, s_mask], 
                             cmap='jet', shading='auto', vmin=-1, vmax=1)
    axes[1].set_yscale('log')
    axes[1].set_ylim(periods[-1], periods[0])
    axes[1].set_ylabel('Period (min)', fontsize=12)
    axes[1].set_title(r'$\sigma_c$', fontsize=14)
    axes[1].tick_params(labelbottom=False)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # 子图3：σ_r
    im3 = axes[2].pcolormesh(T, P, sigma_r[:, s_mask], 
                             cmap='jet', shading='auto', vmin=-1, vmax=1)
    axes[2].set_yscale('log')
    axes[2].set_ylim(periods[-1], periods[0])
    axes[2].set_ylabel('Period (min)', fontsize=12)
    axes[2].set_title(r'$\sigma_r$', fontsize=14)
    axes[2].set_xlabel('Time (UTC)', fontsize=12)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    
    # 添加颜色条
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation='vertical', pad=0.02)
    cbar2 = fig.colorbar(im2, ax=axes[1], orientation='vertical', pad=0.02)
    cbar3 = fig.colorbar(im3, ax=axes[2], orientation='vertical', pad=0.02)
    cbar1.set_label(r'$\sigma_m$', fontsize=10)
    cbar2.set_label(r'$\sigma_c$', fontsize=10)
    cbar3.set_label(r'$\sigma_r$', fontsize=10)
    
    # 添加参考线（如果需要）
    if line_color:
        from datetime import datetime
        front = julian.to_jd(datetime(2024, 9, 26, 9, 36, 0), fmt='jd')
        rear = julian.to_jd(datetime(2024, 9, 26, 10, 9, 0), fmt='jd')
        front1 = julian.to_jd(datetime(2024, 9, 26, 5, 40, 0), fmt='jd')
        rear1 = julian.to_jd(datetime(2024, 9, 26, 11, 40, 0), fmt='jd')
        
        for ax in axes:
            ax.axvline(Time(front, format='jd').datetime, color='red', linestyle='--', linewidth=1.5)
            ax.axvline(Time(rear, format='jd').datetime, color='red', linestyle='--', linewidth=1.5)
            ax.axvline(Time(front1, format='jd').datetime, color='black', linestyle='-', linewidth=1.5)
            ax.axvline(Time(rear1, format='jd').datetime, color='black', linestyle='-', linewidth=1.5)
    
    plt.suptitle(f'Wavelet Analysis: {start} - {end}', fontsize=14)
    plt.show()
    
    print("绘图完成！")
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import julian

from astropy.time import Time
from datetime import datetime
from pyrfu.pyrf.wavelet import wavelet
def draw_3waveletsigma_cmorphy(values0, start, end, line_color=True):
    """
    使用pyrfu wavelet计算:
    σ_m (magnetic helicity)
    σ_c (cross helicity)
    σ_r (residual energy)
    """

    # ==============================
    # 时间窗口检查
    # ==============================
    time0 = values0['time_data']
    time_mask = (time0 >= start) & (time0 <= end)

    if not np.any(time_mask):
        print('⚠ 时间区间内无有效数据，无法绘图')
        return

    dt = values0['resolution']
    scale_num = 64
    wavelet_width=2.2

    min_period = 30
    max_period = 900

    # 转频率Hz（低->高）
    freqs = np.logspace(
        np.log10(1 / (max_period * 60)),
        np.log10(1 / (min_period * 60)),
        scale_num
    )

    # ==============================
    # 扩展时间窗口减少边缘效应
    # ==============================
    min_freq = np.min(freqs)
    support_sigma = 4.0
    half_support_days = support_sigma / min_freq / 86400.0

    start_extended = start - half_support_days
    end_extended = end + half_support_days

    ext_mask = (time0 >= start_extended) & (time0 <= end_extended)

    if np.sum(ext_mask) < 10:
        print('⚠ 扩展后数据点过少，无法绘图')
        return

    # ==============================
    # 数据提取
    # ==============================
    time_ext = time0[ext_mask]
    time_xr = Time(time_ext, format='jd').datetime

    B_ext = values0['b_data'][ext_mask].astype(np.float64)
    V_ext = values0['v_data'][ext_mask].astype(np.float64)
    nep_ext = values0['nep_data'][ext_mask].astype(np.float64) * 1e6

    # ==============================
    # 阿尔芬速度
    # ==============================
    Ca_ext = 1e-12 / np.sqrt(
        4 * np.pi * 1e-7 * 1.6726e-27 * nep_ext
    )
    Ca_ext = Ca_ext[:, np.newaxis]

    # ==============================
    # 去均值（扰动）
    # ==============================
    B_mean = np.mean(B_ext, axis=0)
    V_mean = np.mean(V_ext, axis=0)

    B_fluct = B_ext - B_mean
    V_fluct = V_ext - V_mean

    # ==============================
    # Elsässer变量
    # ==============================
    Z_plus = V_fluct + Ca_ext * B_fluct
    Z_minus = V_fluct - Ca_ext * B_fluct

    # ==============================
    # 初始化小波数组
    # ==============================
    wave_B = np.zeros((3, scale_num, len(time_ext)), dtype=np.complex128)
    wave_zplus = np.zeros((3, scale_num, len(time_ext)), dtype=np.complex128)
    wave_zminus = np.zeros((3, scale_num, len(time_ext)), dtype=np.complex128)

    # ==============================
    # 分量逐个小波变换
    # ==============================
    for i in range(3):

        # ---------- B ----------
        da_B = xr.DataArray(
            B_fluct[:, i],
            coords={"time": time_xr},
            dims=["time"]
        )

        wt_B = wavelet(
            da_B,
            f=[np.min(freqs), np.max(freqs)],
            n_freqs=scale_num,
            wavelet_width=wavelet_width,
            cut_edge=False,
            return_power=False
        )

        wave_B[i] = wt_B.data.T

        # ---------- Z+ ----------
        da_Zp = xr.DataArray(
            Z_plus[:, i],
            coords={"time": time_xr},
            dims=["time"]
        )

        wt_Zp = wavelet(
            da_Zp,
            f=[np.min(freqs), np.max(freqs)],
            n_freqs=scale_num,
            wavelet_width=wavelet_width,
            cut_edge=False,
            return_power=False
        )

        wave_zplus[i] = wt_Zp.data.T

        # ---------- Z- ----------
        da_Zm = xr.DataArray(
            Z_minus[:, i],
            coords={"time": time_xr},
            dims=["time"]
        )

        wt_Zm = wavelet(
            da_Zm,
            f=[np.min(freqs), np.max(freqs)],
            n_freqs=scale_num,
            wavelet_width=wavelet_width,
            cut_edge=False,
            return_power=False
        )

        wave_zminus[i] = wt_Zm.data.T

    # ==============================
    # 获取频率和周期
    # ==============================
    freq_cwt = wt_B.frequency.data
    periods = 1.0 / freq_cwt / 60.0

    # ==============================
    # 功率谱
    # ==============================
    power_B = np.sum(np.abs(wave_B) ** 2, axis=0)
    power_zplus = np.sum(np.abs(wave_zplus) ** 2, axis=0)
    power_zminus = np.sum(np.abs(wave_zminus) ** 2, axis=0)

    # 防止除零
    power_B[power_B == 0] = np.nan
    denom_Z = power_zplus + power_zminus
    denom_Z[denom_Z == 0] = np.nan

    # ==============================
    # σ_m
    # ==============================
    sigma_m = 2 * np.imag(
        np.conj(wave_B[1]) * wave_B[2]
    ) / power_B

    # ==============================
    # σ_c
    # ==============================
    sigma_c = (power_zplus - power_zminus) / denom_Z

    # ==============================
    # σ_r
    # ==============================
    cross_term = (
        np.conj(wave_zplus[0]) * wave_zminus[0] +
        np.conj(wave_zplus[1]) * wave_zminus[1] +
        np.conj(wave_zplus[2]) * wave_zminus[2]
    )

    sigma_r = 2 * np.real(cross_term) / denom_Z

    # ==============================
    # 裁剪目标区间
    # ==============================
    s_mask = (time_ext >= start) & (time_ext <= end)
    time_select = Time(time_ext[s_mask], format='jd').datetime

    # ==============================
    # 网格
    # ==============================
    T, P = np.meshgrid(time_select, periods)

    # ==============================
    # 绘图
    # ==============================
    fig, axes = plt.subplots(3, 1, figsize=(13, 9))
    plt.subplots_adjust(
        top=0.95,
        bottom=0.08,
        hspace=0.18,
        left=0.12,
        right=0.92
    )

    datasets = [sigma_m, sigma_c, sigma_r]
    titles = [r'$\sigma_m$', r'$\sigma_c$', r'$\sigma_r$']

    for idx, ax in enumerate(axes):
        im = ax.pcolormesh(
            T,
            P,
            datasets[idx][:, s_mask],
            cmap='jet',
            shading='auto',
            vmin=-1,
            vmax=1
        )

        ax.set_yscale('log')
        ax.set_ylim(periods[0], periods[-1])
        ax.set_ylabel('Period (min)', fontsize=12)
        ax.set_title(titles[idx], fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)

        if idx < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time (UTC)', fontsize=12)

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label(titles[idx], fontsize=10)

    # ==============================
    # 参考线
    # ==============================
    if line_color:
        front = julian.to_jd(datetime(2024, 9, 26, 9, 36, 0), fmt='jd')
        rear = julian.to_jd(datetime(2024, 9, 26, 10, 9, 0), fmt='jd')
        front1 = julian.to_jd(datetime(2024, 9, 26, 5, 40, 0), fmt='jd')
        rear1 = julian.to_jd(datetime(2024, 9, 26, 11, 40, 0), fmt='jd')

        for ax in axes:
            ax.axvline(
                Time(front, format='jd').datetime,
                color='red', linestyle='--', linewidth=1.5
            )
            ax.axvline(
                Time(rear, format='jd').datetime,
                color='red', linestyle='--', linewidth=1.5
            )
            ax.axvline(
                Time(front1, format='jd').datetime,
                color='black', linestyle='-', linewidth=1.5
            )
            ax.axvline(
                Time(rear1, format='jd').datetime,
                color='black', linestyle='-', linewidth=1.5
            )

    plt.show()
    print('绘图完成！')

# ============================================================================
if __name__ == "__main__":
    """
    #对磁场、等离子体、超热电子数据拼接
    start_date = "2021-04-29"
    end_date = "2021-04-29"
    
    mag_data = mag_data_done(start_date, end_date)
    print("磁场数据总点数:", len(mag_data["time_B"]))
    plasma_data = plasma_data_done(start_date, end_date)
    print("等离子体数据总点数:", len(plasma_data["time_p"]))
    strahl_data = strahl_data_done(start_date, end_date)
    print("超热电子数据总点数:", len(strahl_data["time_s"]))
    
    data = mag_data | plasma_data | strahl_data
    
    pkl_save(data,save_pkl = True)
    """
    with open("E:/science/20240926/data_yuchuli.pkl",'rb')as f:
        data = pickle.load(f)
    start_time = julian.to_jd(datetime(2024, 9, 26, 9, 40, 0), fmt = 'jd')
    end_time = julian.to_jd(datetime(2024, 9, 26, 9, 50, 59), fmt = 'jd')
    #cut_start = julian.to_jd(datetime(2024, 9, 26, 9, 18, 0), fmt='jd')
    #cut_end   = julian.to_jd(datetime(2024, 9, 26, 10, 40, 0), fmt='jd')
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
     
        spike_done(data, plasma_mask)#去除毛刺
        #draw_spike(data, start_time, end_time)#对比毛刺去除前后图像
        insert_data(data, start_time, end_time)#插值为同轴同分辨率数据
        draw_8yuchuli(data, start_time, end_time)
        draw_4wave(data, start_time, end_time)
        #draw_12xiangguan(data, start_time, end_time)
    #calculate_num(data)#计算关键数据复现三个小波变换图
    #draw_3waveletsigma_cmorphy(data, start_time, end_time, False)
    #draw_wavelet_3sigma(data, start_time, end_time, title='Wavelet Analysis')
    #draw_3waveletsigma_cmor(data, start_time, end_time, line_color=False)
        #draw_9byangle(data, start_time, end_time)
        #draw_7tuandong(data, start_time, end_time)
        #draw_xiaobobysegment(data, start_time, end_time)
        #draw_3xiaoboandangle(data, start_time, end_time)
        #MVAfind_draw(data, start_time, end_time)
        #draw_MVA_step(data, start_time, end_time)
        #draw_MVALMN(data, start_time, end_time)
        #draw_MVALMN(data, start_time, end_time, cut_windows=[(cut_start, cut_end)])
