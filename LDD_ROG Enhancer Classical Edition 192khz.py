# -*- coding: utf-8 -*-
# Classical Edition

import sys
import os
import time
import gc
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa

INTERNAL_SR = 192000
BLOCK_SEC = 15.0  # 每个分块的时长
PAD_SEC = 1.0

def butter_filter(data, cutoff, fs, btype, order=4):   # 滤波
    nyq = 0.5 * fs
    if btype in ['bandpass', 'bandstop']:
        if isinstance(cutoff, (int, float)):
            cutoff = [max(10, cutoff * 0.9), min(fs/2 - 10, cutoff * 1.1)]
        normal_cutoff = np.array(cutoff) / nyq
    else:
        normal_cutoff = cutoff / nyq
    
    normal_cutoff = np.clip(normal_cutoff, 0.001, 0.999)
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    return signal.filtfilt(b, a, data)

def get_envelope(y, frame_size=2048):   # 确保非负值
    hop = frame_size // 2
    if y.ndim > 1:
        # 如果是立体声，取平均值
        y_mono = np.mean(y, axis=0)
    else:
        y_mono = y
    
    # 计算RMS
    rmse = librosa.feature.rms(y=y_mono, frame_length=frame_size, hop_length=hop, center=True)
    envelope = signal.resample(np.maximum(rmse[0], 1e-10), len(y_mono))
    
    # 轻微平滑包络
    envelope = np.clip(envelope, 1e-10, None)
    
    # 如果原始是立体声，复制包络到两个声道
    if y.ndim > 1:
        envelope = np.stack([envelope, envelope])
    
    return envelope

def analyze_classical_profile(y, sr):   # 特征分析
    frame_len = int(0.05 * sr)
    hop = frame_len // 2
    
    if y.ndim > 1:
        y_mono = np.mean(y, axis=0)
    else:
        y_mono = y
    
    # 确保音频长度足够
    if len(y_mono) < frame_len * 2:
        return 0.7, 0.5, 0.5
    
    # 动态分析
    num_frames = (len(y_mono) - frame_len) // hop + 1
    if num_frames <= 0:
        return 0.7, 0.5, 0.5
    
    rms_values = []
    peak_values = []
    
    for i in range(num_frames):
        start = i * hop
        frame = y_mono[start:start + frame_len]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        peak = np.max(np.abs(frame))
        rms_values.append(rms)
        peak_values.append(peak)
    
    rms_values = np.array(rms_values)
    peak_values = np.array(peak_values)
    
    # 去除静音帧
    valid_idx = rms_values > 0.001
    if np.sum(valid_idx) < 10:
        return 0.7, 0.5, 0.5
    
    rms_values = rms_values[valid_idx]
    peak_values = peak_values[valid_idx]
    
    # 动态范围计算
    mean_rms = np.mean(rms_values)
    std_rms = np.std(rms_values)
    cv = std_rms / (mean_rms + 1e-10)
    dynamic_score = np.clip((cv - 0.05) / 0.8, 0, 1)
    
    # 瞬态分析
    crest_factor = peak_values / (rms_values + 1e-10)
    transient_score = np.clip(np.percentile(crest_factor, 75) / 8, 0, 1)
    
    # 频谱平衡分析
    if len(y_mono) > 1000:
        try:
            freqs = np.fft.rfftfreq(len(y_mono), 1/sr)
            fft = np.abs(np.fft.rfft(y_mono))
            low_energy = np.sum(fft[freqs < 2000])
            high_energy = np.sum(fft[freqs > 8000])
            spectral_balance = np.clip(high_energy / (low_energy + high_energy + 1e-10), 0, 1)
        except:
            spectral_balance = 0.5
    else:
        spectral_balance = 0.5
    
    return dynamic_score, transient_score, spectral_balance

def gamma_detail_lift(signal_in, gamma=0.85, boost_db=3.0):   # 提升细节
    linear_boost = 10 ** (boost_db / 20.0)
    abs_signal = np.abs(signal_in)
    sign_signal = np.sign(signal_in)
    threshold = 0.0005
    
    abs_signal = np.maximum(abs_signal, 1e-15)
    lifted = np.power(abs_signal, gamma)
    
    output = sign_signal * lifted * linear_boost * 0.7
    
    # 平滑阈值
    gate = (abs_signal > threshold).astype(float)
    
    # 应用轻微平滑
    if gate.shape[-1] > 100:
        window_size = 51
        window = signal.windows.hann(window_size)
        window = window / np.sum(window)
        
        if signal_in.ndim == 1:
            smooth_gate = np.convolve(gate, window, mode='same')
        else:
            smooth_gate = np.zeros_like(gate)
            for i in range(gate.shape[0]):
                smooth_gate[i] = np.convolve(gate[i], window, mode='same')
    else:
        smooth_gate = gate
    
    output = output * smooth_gate
    
    # 防止数值溢出
    output = np.clip(output, -1.0, 1.0)
    
    return output

def create_crossfade_blend(y_orig, hf_ext, sr, blend_low=12000, blend_high=18000):  # 频段混合
    blend_low = max(100, blend_low)
    blend_high = min(sr/2 - 100, blend_high)
    
    if blend_high <= blend_low:
        blend_high = blend_low + 1000
    
    orig_transition = butter_filter(y_orig, [blend_low, blend_high], sr, 'bandpass', order=4)
    ext_transition = butter_filter(hf_ext, [blend_low, blend_high], sr, 'bandpass', order=4)
    
    # 分析原始高频质量
    if orig_transition.ndim > 1:
        orig_mono = np.mean(orig_transition, axis=0)
    else:
        orig_mono = orig_transition
    
    # 计算谐波质量
    if len(orig_mono) > 100:
        try:
            fft_orig = np.abs(np.fft.rfft(orig_mono))
            peaks, _ = signal.find_peaks(fft_orig, height=np.max(fft_orig)*0.1)
            harmonic_ratio = len(peaks) / len(fft_orig) * 1000 if len(peaks) > 0 else 0
        except:
            harmonic_ratio = 0
    else:
        harmonic_ratio = 0
    
    # 基于谐波质量决定混合权重
    if harmonic_ratio > 5:
        base_orig_weight = 0.8
    elif harmonic_ratio > 2:
        base_orig_weight = 0.6
    else:
        base_orig_weight = 0.4
    
    # 创建权重数组
    if orig_transition.ndim > 1:
        orig_weight = np.ones_like(orig_transition) * base_orig_weight
        ext_weight = np.ones_like(orig_transition) * (1 - base_orig_weight)
    else:
        orig_weight = np.ones_like(orig_transition) * base_orig_weight
        ext_weight = np.ones_like(orig_transition) * (1 - base_orig_weight)
    
    # 获取包络
    env = get_envelope(orig_mono, frame_size=4096)
    if orig_transition.ndim > 1:
        env = np.stack([env, env])
    
    # 确保包络数值稳定
    env = np.clip(env, 0, 1)
    
    # 应用包络平滑
    smoothed_weight = orig_weight * env + orig_weight * (1 - env) * 0.5
    smoothed_weight = np.clip(smoothed_weight, 0, 1)
    
    blended_transition = orig_transition * smoothed_weight + ext_transition * (1 - smoothed_weight)
    
    return blended_transition

def create_air_band(y_orig, sr, center_freq=16000, bandwidth=6000):  # 空气频段
    # 计算频带边界
    low_freq = max(100, center_freq - bandwidth/2)
    high_freq = min(sr/2 - 100, center_freq + bandwidth/2)
    
    if high_freq <= low_freq:
        high_freq = low_freq + 1000
    
    # 提取空气频段
    if high_freq > low_freq:
        air_band = butter_filter(y_orig, [low_freq, high_freq], sr, 'bandpass', order=4)
    else:
        air_band = butter_filter(y_orig, center_freq, sr, 'highpass', order=4)
    
    # 轻微饱和增加温暖感
    air_saturated = np.tanh(air_band * 1.2)
    
    # 获取包络
    if air_saturated.ndim > 1:
        air_mono = np.mean(air_saturated, axis=0)
    else:
        air_mono = air_saturated
    
    env = get_envelope(air_mono, frame_size=8192)
    
    if air_saturated.ndim > 1:
        env = np.stack([env, env])
    
    # 确保包络非负，然后延展包络
    env = np.maximum(env, 1e-10)
    env_extended = np.sqrt(env)  # 现在env保证为正数
    
    # 添加极低电平的噪声
    noise_shape = air_saturated.shape
    noise = np.random.normal(0, 0.0003, noise_shape)
    
    if high_freq > low_freq:
        noise = butter_filter(noise, [low_freq, high_freq], sr, 'bandpass', order=2)
    else:
        noise = butter_filter(noise, center_freq, sr, 'highpass', order=2)
    
    # 组合结果
    result = air_saturated * env_extended + noise * 0.3
    
    # 防止数值溢出
    result = np.clip(result, -1.0, 1.0)
    
    return result

def create_modulation_layer(y_orig, sr, low_freq=5000, high_freq=20000):
    """
    创建5-20kHz调制层
    通过轻微的频率调制增强高频感知
    """
    # 提取目标频段
    target_band = butter_filter(y_orig, [low_freq, high_freq], sr, 'bandpass', order=4)
    
    # 创建调制信号（低频正弦波，模拟磁带/电子管效应）
    t = np.arange(target_band.shape[1]) / sr
    if target_band.ndim > 1:
        t = t.reshape(1, -1)
    
    # 使用多个低频调制频率
    mod_freq1 = 0.5  # 0.5Hz - 非常慢的调制
    mod_freq2 = 2.0  # 2.0Hz - 轻微颤音效果
    
    modulation1 = 1.0 + 0.02 * np.sin(2 * np.pi * mod_freq1 * t)  # 1%的缓慢调制
    modulation2 = 1.0 + 0.01 * np.sin(2 * np.pi * mod_freq2 * t)  # 0.5%的快速调制
    
    # 组合调制效果
    modulation = modulation1 * modulation2
    
    # 应用调制
    modulated = target_band * modulation
    
    # 根据原始信号包络调整调制强度
    if target_band.ndim > 1:
        target_mono = np.mean(target_band, axis=0)
    else:
        target_mono = target_band
    
    env = get_envelope(target_mono, frame_size=4096)
    if target_band.ndim > 1:
        env = np.stack([env, env])
    
    # 调制强度随原始信号动态变化
    modulated = modulated * (0.3 + env * 0.7)  # 30%基础调制 + 70%随动态变化
    
    return modulated

def add_spectral_tilt_fir(y_signal, sr, tilt_start_freq=10000, tilt_amount=-0.2):   # 频谱倾斜
    # 确保参数在合理范围内
    tilt_amount = np.clip(tilt_amount, -0.3, 0.3)
    
    # 设计倾斜滤波器
    if tilt_amount < 0:
        # 高频衰减滤波器
        b = [1.0, -abs(tilt_amount) * 0.7]
        a = [1.0, -abs(tilt_amount) * 0.3]
    else:
        # 高频提升滤波器
        b = [1.0 + tilt_amount * 0.7, 0]
        a = [1.0, tilt_amount * 0.3]
    
    # 应用滤波器
    if y_signal.ndim > 1:
        y_tilted = np.zeros_like(y_signal)
        for ch in range(y_signal.shape[0]):
            y_tilted[ch] = signal.filtfilt(b, a, y_signal[ch])
    else:
        y_tilted = signal.filtfilt(b, a, y_signal)
    
    return y_tilted

def check_audio_finite(y, label="音频"):   # 检查音频数据是否都是有限值
    if not np.all(np.isfinite(y)):
        print(f"警告: {label} 包含非有限值")
        print(f"  NaN数量: {np.sum(np.isnan(y))}")
        print(f"  Inf数量: {np.sum(np.isinf(y))}")
        return False
    return True

def fix_audio_values(y):   # 修复非有限值
    if not np.all(np.isfinite(y)):
        # 用附近的有效值替换NaN
        y_fixed = y.copy()
        nan_mask = ~np.isfinite(y_fixed)
        if np.any(nan_mask):
            # 用0替换NaN和Inf
            y_fixed[nan_mask] = 0
            print(f"  已修复 {np.sum(nan_mask)} 个非有限值")
        return y_fixed
    return y

def process_chunk_logic_classical(y_chunk, sr):
    # 0. 检查输入数据
    if not check_audio_finite(y_chunk, "输入音频"):
        y_chunk = fix_audio_values(y_chunk)
    
    # 1. 确保输入是立体声
    if y_chunk.ndim == 1:
        y_chunk = np.stack([y_chunk, y_chunk])
    elif y_chunk.shape[0] == 1:
        y_chunk = np.repeat(y_chunk, 2, axis=0)
    
    # 2. 升频
    try:
        y_up = librosa.resample(y_chunk, orig_sr=sr, target_sr=INTERNAL_SR, res_type='kaiser_fast')
    except Exception as e:
        print(f"升频失败: {e}")
        # 使用线性插值作为后备
        ratio = INTERNAL_SR / sr
        new_len = int(len(y_chunk[0]) * ratio)
        y_up = np.zeros((2, new_len))
        for ch in range(2):
            y_up[ch] = np.interp(
                np.linspace(0, len(y_chunk[ch])-1, new_len),
                np.arange(len(y_chunk[ch])),
                y_chunk[ch]
            )
    
    # 确保升频后仍是立体声
    if y_up.ndim == 1:
        y_up = np.stack([y_up, y_up])
    
    # 3. 分析古典乐特征
    dynamic_score, transient_score, spectral_balance = analyze_classical_profile(y_up, INTERNAL_SR)
    
    # 4. 源提取
    source = butter_filter(y_up, [1300, 20000], INTERNAL_SR, 'bandpass', order=4)
    
    # 5. 谐波发生器
    drive_amount = 2.2 + dynamic_score * 0.5
    saturated_source = np.arctan(source * drive_amount) * (2/np.pi)
    
    # 生成谐波
    h2 = saturated_source ** 2 * 0.8
    h3 = saturated_source ** 3 * 0.4
    h5 = saturated_source ** 5 * 0.2
    
    # 去除直流偏移
    for h in [h2, h3, h5]:
        h -= np.mean(h, axis=-1, keepdims=True)
    
    # 组合谐波
    high_freq_weight = 0.5 + spectral_balance * 0.5
    raw_harmonics = h2 * (0.4 + high_freq_weight*0.2) + h3 * 0.25 + h5 * 0.15
    
    # 6. Gamma提亮
    lifted_harmonics = gamma_detail_lift(raw_harmonics, gamma=0.85, boost_db=3.0)
    
    # 7. 空气感增强层
    air_layer = create_air_band(y_up, INTERNAL_SR, center_freq=18000, bandwidth=2000)
    
    # 8. 瞬态保留层
    transient_extract = butter_filter(y_up, [8000, 18000], INTERNAL_SR, 'bandpass', order=4)
    
    if transient_extract.ndim > 1:
        transient_mono = np.mean(transient_extract, axis=0)
    else:
        transient_mono = transient_extract
    
    transient_env = get_envelope(transient_mono, frame_size=1024)
    
    if transient_extract.ndim > 1:
        transient_env = np.stack([transient_env, transient_env])
    
    transient_layer = transient_extract * transient_env * transient_score * 1.2
    
    # 9. 创建调制并混合所有层
    modulation_layer = create_modulation_layer(y_up, INTERNAL_SR, low_freq=5000, high_freq=20000)
    hf_extension = lifted_harmonics + air_layer * 0.4 + transient_layer * 0.3 + modulation_layer * 0.2
    
    # 10. 频段净化
    hf_extension = butter_filter(hf_extension, 13000, INTERNAL_SR, 'highpass', order=2)
    
    # 11. 智能能量匹配
    ref_band = butter_filter(y_up, [14500, 18500], INTERNAL_SR, 'bandpass', order=4)
    ext_band = butter_filter(hf_extension, [14500, 18500], INTERNAL_SR, 'bandpass', order=4)
    
    rms_orig = np.sqrt(np.mean(ref_band**2) + 1e-10)
    rms_ext = np.sqrt(np.mean(ext_band**2) + 1e-10)
    
    dynamic_gain_factor = 0.6 + (1 - dynamic_score) * 0.4
    base_gain = (rms_orig / (rms_ext + 1e-10)) * 1.0
    gain = base_gain * dynamic_gain_factor
    
    # 限制增益范围
    gain = np.clip(gain, 0.1, 10.0)
    hf_extension *= gain
    
    # 12. 智能频段融合
    transition_blend = create_crossfade_blend(y_up, hf_extension, INTERNAL_SR, 
                                               blend_low=12000, blend_high=18000)
    
    # 重建高频扩展
    hf_extension_low = butter_filter(hf_extension, 12000, INTERNAL_SR, 'lowpass', order=3)
    hf_extension_high = butter_filter(hf_extension, 18000, INTERNAL_SR, 'highpass', order=3)
    
    hf_extension = hf_extension_low + transition_blend + hf_extension_high

    # 应用频谱倾斜
    hf_extension = add_spectral_tilt_fir(hf_extension, INTERNAL_SR, 
                                      tilt_start_freq=10000, 
                                      tilt_amount=-0.18)
    
    # 13. 物理滚降
    sos1 = signal.butter(1, 32000, btype='lowpass', fs=INTERNAL_SR, output='sos')
    hf_extension = signal.sosfilt(sos1, hf_extension)
    
    sos2 = signal.butter(2, 42000, btype='lowpass', fs=INTERNAL_SR, output='sos')
    hf_extension = signal.sosfilt(sos2, hf_extension)
    
    sos3 = signal.butter(3, 52000, btype='lowpass', fs=INTERNAL_SR, output='sos')
    hf_extension = signal.sosfilt(sos3, hf_extension)
    
    sos4 = signal.butter(4, 62000, btype='lowpass', fs=INTERNAL_SR, output='sos')
    hf_extension = signal.sosfilt(sos4, hf_extension)
    
    sos5 = signal.butter(6, 72000, btype='lowpass', fs=INTERNAL_SR, output='sos')
    hf_extension = signal.sosfilt(sos5, hf_extension)
    
    # 14. 最终输出
    final_gain = 1.0 - (spectral_balance * 0.15)
    y_final = y_up + hf_extension * (final_gain * 1.1)
    
    # 轻微限制
    y_final = np.tanh(y_final * 0.95) * 1.05
    
    # 15. 最终检查
    if not check_audio_finite(y_final, "输出音频"):
        y_final = fix_audio_values(y_final)
    
    # 确保数值范围
    y_final = np.clip(y_final, -1.0, 1.0)
    
    return y_final

def process_audio_classical(input_file, output_sr=192000):   # main
    start_t = time.time()
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(os.path.dirname(input_file), 
                               f"{file_name}_LDD_ROG_Enhancer.wav")
    
    print(f" 处理: {file_name}")
    print(f" 内部采样率: {INTERNAL_SR/1000:.0f}kHz → 输出: {output_sr/1000:.0f}kHz")
    print("-" * 60)
    
    try:
        with sf.SoundFile(input_file) as f_in:
            sr_in = f_in.samplerate
            channels = f_in.channels
            total_frames = f_in.frames
            
            # 显示文件信息
            duration = total_frames / sr_in
            print(f"时长: {duration:.1f}s | 声道: {channels} | 输入采样率: {sr_in}Hz")
            
            with sf.SoundFile(output_file, 'w', samplerate=output_sr, 
                            channels=channels, subtype='PCM_24') as f_out:
                
                block_frames = int(BLOCK_SEC * sr_in)
                pad_frames = int(PAD_SEC * sr_in)
                current_pos = 0
                block_idx = 1
                total_blocks = int(np.ceil(total_frames / block_frames))
                
                while current_pos < total_frames:
                    read_start = max(0, current_pos - pad_frames)
                    read_end = min(total_frames, current_pos + block_frames + pad_frames)
                    
                    f_in.seek(read_start)
                    y_raw = f_in.read(read_end - read_start)
                    
                    # 检查读取的数据
                    if len(y_raw) == 0:
                        print(f"\n  读取到空数据块，跳过")
                        current_pos += block_frames
                        continue
                    
                    # 确保数据是二维的
                    if y_raw.ndim == 1:
                        y_raw = y_raw.reshape(1, -1)
                    
                    # 转置为声道×样本格式
                    y_raw = y_raw.T
                    
                    # 处理当前块
                    y_processed_192 = process_chunk_logic_classical(y_raw, sr_in)
                    
                    # 检查处理后的数据
                    if not check_audio_finite(y_processed_192, "处理后的192kHz音频"):
                        y_processed_192 = fix_audio_values(y_processed_192)
                    
                    # 降采样
                    try:
                        y_processed_96 = librosa.resample(
                            y_processed_192, 
                            orig_sr=INTERNAL_SR, 
                            target_sr=output_sr, 
                            res_type='kaiser_fast'
                        )
                    except Exception as e:
                        print(f"\n  降采样失败: {e}")
                        # 使用线性插值作为后备
                        ratio = output_sr / INTERNAL_SR
                        new_len = int(y_processed_192.shape[1] * ratio)
                        y_processed_96 = np.zeros((y_processed_192.shape[0], new_len))
                        for ch in range(y_processed_192.shape[0]):
                            y_processed_96[ch] = np.interp(
                                np.linspace(0, y_processed_192.shape[1]-1, new_len),
                                np.arange(y_processed_192.shape[1]),
                                y_processed_192[ch]
                            )
                    
                    # 检查降采样后的数据
                    if not check_audio_finite(y_processed_96, "降采样后的音频"):
                        y_processed_96 = fix_audio_values(y_processed_96)
                    
                    # 计算有效部分
                    ratio = output_sr / sr_in
                    valid_start_frames_in = 0 if current_pos == 0 else pad_frames
                    cut_head = int(valid_start_frames_in * ratio)
                    actual_valid_len_in = min(block_frames, total_frames - current_pos)
                    keep_len = int(actual_valid_len_in * ratio)
                    
                    # 确保不会超出数组边界
                    if cut_head >= y_processed_96.shape[1]:
                        cut_head = 0
                    if cut_head + keep_len > y_processed_96.shape[1]:
                        keep_len = y_processed_96.shape[1] - cut_head
                    
                    if keep_len <= 0:
                        print(f"\n  有效长度为零，跳过")
                        current_pos += actual_valid_len_in
                        continue
                    
                    y_to_write = y_processed_96[:, cut_head : cut_head + keep_len]
                    
                    # 恢复原始声道数
                    if channels == 1:
                        y_to_write = np.mean(y_to_write, axis=0, keepdims=True)
                    
                    # 写入文件
                    f_out.write(y_to_write.T)
                    
                    # 进度显示
                    progress = min(100, (current_pos + actual_valid_len_in) / total_frames * 100)
                    sys.stdout.write(f"\r 进度: {progress:5.1f}% | 块 {block_idx:3d}/{total_blocks:3d} | {actual_valid_len_in/sr_in:.1f}s/块")
                    sys.stdout.flush()
                    
                    current_pos += actual_valid_len_in
                    block_idx += 1
                    
                    # 清理内存
                    del y_raw, y_processed_192, y_processed_96, y_to_write
                    if block_idx % 3 == 0:
                        gc.collect()
        
        print(f"\n 处理完成! 总耗时: {time.time() - start_t:.1f}s")
        print(f" 输出文件: {os.path.basename(output_file)}")
        
        # 显示输出文件信息
        output_size = os.path.getsize(output_file) / (1024*1024)
        print(f" 文件大小: {output_size:.1f} MB")
        
    except Exception as e:
        print(f"\n 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("\n\n\n 猪上床猪下床猪上吊猪听歌猪离去猪玩原神猪倒下猪哼哼猪吃饭猪外有猪")
    print("                LDD_ROG Enhancer Classical Edition")
    print(" 猪抽卡猪跳舞猪生气猪疑惑猪学习猪拉电线猪跳跃猪干嘛猪开心猪内有猪\n\n\n")
    
    # 处理命令行参数
    files = []
    if len(sys.argv) > 1:
        raw_args = sys.argv[1:]
        combined = ' '.join(raw_args)
        combined = combined.strip().strip('"').strip("'")
        if os.path.exists(combined):
            files = [combined]
        else:
            for arg in raw_args:
                clean_path = arg.strip().strip('"').strip("'")
                if clean_path and os.path.exists(clean_path):
                    files.append(clean_path)
    
    if not files:
        print("\n请拖入古典乐分轨文件（支持多选）或输入路径：")
        manual = input("文件路径: ").strip().strip('"').strip("'")
        if manual and os.path.exists(manual):
            files = [manual]
        else:
            # 尝试分割多个文件
            for path in manual.split():
                clean_path = path.strip().strip('"').strip("'")
                if clean_path and os.path.exists(clean_path):
                    files.append(clean_path)
    
    if files:
        print(f"\n找到 {len(files)} 个文件:")
        for i, f in enumerate(files, 1):
            print(f"  {i}. {os.path.basename(f)}")
        
        print("\n" + "="*70)
        
        for f in files:
            try:
                print(f"\n 开始处理: {os.path.basename(f)}")
                process_audio_classical(f)
            except Exception as e:
                import traceback
                print(f"\n 处理出错: {f}")
                traceback.print_exc()
                print("\n是否继续处理其他文件？ (y/n): ", end="")
                if input().lower() != 'y':
                    break
    else:
        print("没有找到有效的音频文件！")
    
    print("\n 按回车键退出...")
    input()