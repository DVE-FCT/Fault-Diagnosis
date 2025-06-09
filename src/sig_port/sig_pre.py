import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.fft import fft
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 信号处理类
class SigPre:
    def __init__(self, data=None, angle_step=1, rpm=600, fs=50000):
        self.data = data
        self.angle_step = angle_step  # 角度步长，单位：度
        self.rpm = rpm                # 转速（转/分）
        self.fs = fs                  # 采样率（Hz）

    def normalize(self, signal):
        """标准化信号，减少转速引起的幅值变化"""
        mean = np.mean(signal)
        std = np.std(signal)
        return (signal - mean) / std if std != 0 else (signal - mean)

    def envelope(self, signal):
        """希尔伯特包络变换"""
        analytic = hilbert(signal)
        return np.abs(analytic) # 因为后续处理提取了正频谱，过滤的直流成分，所以这里不需要减去直流成分（信号的平均值）
    
    def Cumulative_angle(self, signal):
        """计算信号的累积角度"""
        # 时间序列
        t = np.arange(len(signal)) / self.fs
        # 每秒转过的角度（度/秒）
        deg_per_sec = self.rpm * 360.0 / 60.0
        # 理论瞬时角度 = deg_per_sec * t
        instant_angle = deg_per_sec * t
        # 数值积分累积角度
        cum_angle = cumulative_trapezoid(instant_angle, t, initial=0)
        return cum_angle


    def equal_angle_resample(self, signal):
        """
        对时域信号做希尔伯特包络，并按等角度间隔重采样
        返回：等角度重采样信号, 等角度序列
        """
        cum_angle = self.Cumulative_angle(signal)
        total_angle = cum_angle[-1]
        angles = np.arange(0, total_angle, self.angle_step)

        # 包络
        env = self.envelope(signal)

        # 原始时刻
        t = np.arange(len(signal)) / self.fs
        print(f"采样时间：{t[-1]:.2f} 秒")

        # 每秒转过的角度（度/秒）
        deg_per_sec = self.rpm * 360.0 / 60.0
        print(f"每秒转过的角度：{deg_per_sec:.2f} 度/秒")
        # 角度对应时间：t = θ (deg) / (deg/sec)
        angle_time = angles / deg_per_sec
        print(f"累计角度：{total_angle:.2f} 度" )

        # 三次样条插值
        interp_func = interp1d(t, env, kind='cubic', fill_value='extrapolate')
        resampled = interp_func(angle_time)
        return resampled, angles

    def order_spectrum(self, resampled_signal):
        """
        对等角度重采样信号进行FFT，计算阶次谱
        返回：阶次数组, 幅值谱
        """
        N = len(resampled_signal)
        spectrum = np.abs(fft(resampled_signal)) * 2 / N
        half = int(np.round(N / 2))
        mags = spectrum[0:half] # 取正频谱与直流成分（0hz），过滤掉负频谱、奈奎斯特频率
        # 对应阶次
        orders = np.arange(1, half) * (360 / self.angle_step) / N
        # 前100个阶次
        orders = orders[:100]
        mags = mags[:100]
        return orders, mags

    def process_data(self, input_file):
        """
        读取 .csv/.xlsx 文件，逐行/逐 Sheet 处理信号（前 1024 点）并提取最后一列标签
        返回：包含信号帧、阶次谱以及标签的结果字典
        """
        results = {}

        def handle_signal(row, key):
            # 前 1024 为信号数据，最后一列为标签
            vib = np.asarray(row[:-1], dtype=float)
            label = row[-1]
            # 标准化
            normalized = self.normalize(vib)
            # 包络（在标准化信号上做 Hilbert）
            envelope = self.envelope(normalized)
            # 等角度重采样包络
            angle_domain_envelope, angles = self.equal_angle_resample(normalized)
            # 阶次谱
            orders, mags = self.order_spectrum(angle_domain_envelope)

            # 存储所有中间结果
            results[key] = {
                'normalized': normalized,
                'envelope': envelope,
                'angle_domain_envelope': angle_domain_envelope,
                'angles': angles,
                'orders': orders,
                'mags': mags,
                'label': label
            }
            print(f"处理信号 {key} 完成。")

        # 根据后缀判断
        if input_file.endswith('.csv'):
            print("="*20 + f"处理 CSV: {input_file}" + "="*20 + "\n")
            df = pd.read_csv(input_file, header=0) # 以第一行作为列名
            for idx, row in df.iterrows():
                handle_signal(row.values, f'row_{idx}')

        elif input_file.endswith(('.xlsx', '.xls')):
            print("="*20 + f"处理 Excel: {input_file}" + "="*20 + "\n")
            sheets = pd.read_excel(input_file, sheet_name=None, header=0) # 以第一行作为列名
            for sheet_name, df_sheet in sheets.items():
                for idx, row in df_sheet.iterrows():
                    handle_signal(row.values, f'{sheet_name}_row_{idx}')
        else:
            raise ValueError("不支持的文件格式，请输入 .csv 或 .xlsx .xls")

        return results

    def save_results(self, processed_data, output_file):
        """
        将阶次谱结果及标签保存到 CSV 或 Excel
        参数：
            processed_data (dict): 处理后的数据字典
            output_file (str): 输出文件名
        """
        if not processed_data:
            print("警告：无数据可保存。")
            return

        # 构建 DataFrame：每行 mags + label
        first = next(iter(processed_data.values()))
        orders = first['orders']
        mag_cols = [f'Order_{int(o)}' for o in orders]

        mag_rows = []
        labels = []
        for data in processed_data.values():
            mag_rows.append(data['mags'])
            labels.append(data['label'])

        df_out = pd.DataFrame(mag_rows, columns=mag_cols)
        df_out['label'] = labels

        if output_file.lower().endswith('.csv'):
            df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
        else:
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                df_out.to_excel(writer, sheet_name='orders', index=False)

        print(f"结果已保存到: {output_file}")


# 绘图类 
import os

class SigPlot:
    def __init__(self):
        pass

    # 1. 绘制信号及其分布
    def plot_signal_and_density(self, signal, title, save_dir, file_name):
        """绘制信号的直方图和核密度估计曲线。"""
        plt.figure(figsize=(12, 6))
        sns.histplot(signal, kde=True, stat="density", linewidth=0, bins=50)
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        plt.axvline(mean_val, color='r', linestyle='--', label=f'均值: {mean_val:.2f}')
        plt.axvline(mean_val + std_val, color='g', linestyle='--', label=f'1倍标准差范围')
        plt.axvline(mean_val - std_val, color='g', linestyle='--')
        plt.title(title)
        plt.xlabel("信号幅值")
        plt.ylabel("密度")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 保存图像
        plt.savefig(os.path.join(save_dir, f"{file_name}_signal_distribution.png"))
        plt.close()

    # 2. 绘制信号与包络图
    def plot_envelope(self, original_signal, envelope, title, save_dir, file_name):
        """在同一张图上绘制原始信号和其包络信号。"""
        plt.figure(figsize=(12, 6))
        time_axis = np.arange(len(original_signal))
        plt.plot(time_axis, original_signal, color='c', alpha=0.6, label="时域信号 (标准化后)")
        plt.plot(time_axis, envelope, color='r', label="包络信号")
        plt.title(title)
        plt.xlabel("时间 (采样点)")
        plt.ylabel("幅值")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 保存图像
        plt.savefig(os.path.join(save_dir, f"{file_name}_envelope.png"))
        plt.close()

    # 3. 绘制角域信号图
    def plot_angle_domain_signal(self, angle_signal, title, save_dir, file_name):
        """绘制重采样后的角域信号。"""
        plt.figure(figsize=(12, 6))
        angle_axis = np.arange(len(angle_signal))
        plt.plot(angle_axis, angle_signal)
        plt.title(title)
        plt.xlabel("角度 (采样点)")
        plt.ylabel("包络幅值")
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 保存图像
        plt.savefig(os.path.join(save_dir, f"{file_name}_angle_domain.png"))
        plt.close()

    # 4. 绘制阶次谱图
    def plot_order_spectrum(self, orders, mags, title, save_dir, file_name):
        """绘制最终的阶次谱图。"""
        plt.figure(figsize=(12, 6))
        plt.plot(orders, mags, marker='o', linestyle='-', markersize=4)
        plt.title(title)
        plt.xlabel("阶次 (Order)")
        plt.ylabel("幅值")
        # 标记出幅值最高的几个阶次
        top_indices = np.argsort(mags)[-5:]  # 找到最高的5个点
        for i in top_indices:
            plt.text(orders[i], mags[i], f' {orders[i]}阶', verticalalignment='bottom')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.xlim(0, 50)
        plt.ylim(bottom=0)
        
        # 保存图像
        plt.savefig(os.path.join(save_dir, f"{file_name}_order_spectrum.png"))
        plt.close()


def sig_pre(input_file, output_file, angle_step=None, rpm=None, fs=None, plot=False):
    """
    主函数，用于处理信号、保存结果并根据需要进行绘图。

    参数:
        plot (bool): 如果为True，则对输入文件的第一条信号绘制所有分析图表。
    """
    if angle_step is None or rpm is None or fs is None:
        raise ValueError("请设置角度步长、转速和采样率。")

    processor = SigPre(angle_step=angle_step, rpm=rpm, fs=fs)
    plotter = SigPlot()

    # 创建保存图像的文件夹
    output_dir = os.path.dirname(output_file)  # 获取 output_file 所在的文件夹路径
    file_name = os.path.splitext(os.path.basename(output_file))[0]  # 获取不带扩展名的文件名
    save_dir = os.path.join(output_dir, file_name)  # 创建图像保存路径

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 如果文件夹不存在，创建它

    # 1. 处理数据
    processed_data = processor.process_data(input_file)

    # 2. 如果设置了绘图并且有数据，则只对第一条信号进行绘图
    if plot and processed_data:
        # 获取第一条信号的key和其对应的所有数据
        first_key = next(iter(processed_data))
        data = processed_data[first_key]
        
        print("="*20 + f"正在为信号 '{first_key}' 生成图表" + "="*20 + "\n")
        
        # 调用所有绘图函数并保存图像
        plotter.plot_signal_and_density(data['normalized'], f"信号值分布 - {first_key}", save_dir, file_name)
        plotter.plot_envelope(data['normalized'], data['envelope'], f"信号及其包络 - {first_key}", save_dir, file_name)
        plotter.plot_angle_domain_signal(data['angle_domain_envelope'], f"角域包络信号 - {first_key}", save_dir, file_name)
        plotter.plot_order_spectrum(data['orders'], data['mags'], f"阶次谱 - {first_key}", save_dir, file_name)

    # 3. 保存处理结果
    processor.save_results(processed_data, output_file)


# 调用实例
if __name__ == '__main__':
    input_file = r"C:\Users\lenovo\Desktop\paper\Fault-Diagnosis\data\pre_data\dataset_split_0.xlsx"
    output_file = r"C:\Users\lenovo\Desktop\paper\Fault-Diagnosis\data\pre_data\dataset_split_0_result.csv"
    sig_pre(input_file, output_file, angle_step=1, rpm=600, fs=50000, plot=True)
    # 仿真信号测试
    # fs_sim = 200000
    # t = np.linspace(0, 1, fs_sim, endpoint=False)
    # f_base, harmonics = 50, [100, 150]
    # amps = [1.0, 0.5, 0.3]
    # vib = amps[0] * np.sin(2*np.pi*f_base*t)
    # for amp, f in zip(amps[1:], harmonics):
    #     vib += amp * np.sin(2*np.pi*f*t)
    # rpm_sim = 1000
    # # 处理仿真信号
    # sim_pre = SigPre(rpm=rpm_sim, fs=fs_sim)
    # normalized = sim_pre.normalize(vib)
    # envelope = sim_pre.envelope(normalized)
    # angle_env, _ = sim_pre.equal_angle_resample(normalized)
    # orders, mags = sim_pre.order_spectrum(angle_env)
    # # 绘图并保存
    # sim_dir = os.path.join(os.getcwd(), 'simulated_signal')
    # os.makedirs(sim_dir, exist_ok=True)
    # sim_plot = SigPlot()
    # sim_plot.plot_signal_and_density(normalized, "仿真信号分布", sim_dir, "simulated")
    # sim_plot.plot_envelope(normalized, envelope, "仿真信号包络", sim_dir, "simulated")
    # sim_plot.plot_angle_domain_signal(angle_env, "仿真角域包络", sim_dir, "simulated")
    # sim_plot.plot_order_spectrum(orders, mags, "仿真阶次谱", sim_dir, "simulated")
    # print(f"仿真信号图像已保存到: {sim_dir}")
