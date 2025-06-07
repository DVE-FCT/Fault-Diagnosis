import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union


class DataPre:
    """
    自定义的数据预处理类：
      1. 从 data_pre.json 中读取参数，包括：
         - read: 读取时 nrows、sheet_name
         - frame: 分帧参数（frame_length、overlap、apply_window、export_csv、csv_path）
         - label: 整个信号的标签相关配置（label_map、num_classes、shuffle、export_csv、csv_path）
         - split: 数据集划分参数（ratios、shuffle、export_excel、excel_path）
      2. 接收一个字典 file_label_map，键为文件路径，值为标签名称或索引
      3. read_data(): 读取所有文件原始信号，保存为 numpy 数组列表
      4. frame_data(): 对每个文件分帧，裁剪到最小长度，合并所有帧并（可选）导出 CSV
      5. assign_labels(): 给所有帧打上对应的标签，并可打乱、导出 CSV
      6. split_dataset(): 按比例划分 train/test/val，并可导出到同一个 Excel（3 个 sheet）
      7. 还提供静态方法 save_frames_to_csv/save_numpy/load_numpy 供单独使用
    """

    def __init__(self, file_label_map: Dict[str, Union[str,int]], config_path: str = "data_pre.json"):
        """
        初始化时：读取 JSON 配置文件，将所有配置存到 self.config
        并提取 frame_length、overlap 供后续调用。
        接收一个字典 file_label_map，指定多个信号文件路径及其标签
        """
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"未找到配置文件：{config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 文件路径与标签映射
        if not isinstance(file_label_map, dict) or not file_label_map:
            raise ValueError("file_label_map 必须是非空字典，格式 {filepath: label_value, ...}")
        self.file_label_map = file_label_map

        # 读取配置
        read_cfg = self.config.get("read", {})
        self.nrows = read_cfg.get("nrows", None)
        self.sheet_name = read_cfg.get("sheet_name", None)

        frame_cfg = self.config.get("frame", {})
        self.frame_length = frame_cfg.get("frame_length", 1024)
        self.overlap = frame_cfg.get("overlap", 0.5)

        # 用于存储所有文件的原始数据和 DataFrame
        self.raw_data_list: List[np.ndarray] = []
        self.raw_df_list: List[pd.DataFrame] = []
        self.window = None

        # 创建输出目录
        for cfg in (self.config["frame"], self.config["label"], self.config["split"]):
            path = cfg.get("csv_path") or cfg.get("excel_path")
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)


    def read_data(self) -> List[np.ndarray]:
        """
        根据 JSON 中 read.nrows、read.sheet_name 逐个读取 CSV 或 Excel 文件，
        只保留数值列并转换为 numpy 数组，保存到 self.raw_data_list，返回该列表。
        同时记录所有文件的 DataFrame 到 self.raw_df_list。
        """
        for filepath in self.file_label_map:
            print(f"读取文件 {filepath}...")
            ext = os.path.splitext(filepath)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(filepath, nrows=self.nrows)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(filepath, sheet_name=self.sheet_name, nrows=self.nrows)
            else:
                raise ValueError(f"不支持的文件类型：{ext}。请提供 .csv 或 .xlsx/.xls 文件。")

            data_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            arr = df[data_cols].to_numpy()
            self.raw_df_list.append(df.copy())
            self.raw_data_list.append(arr)

        return self.raw_data_list

    def _compute_window(self) -> np.ndarray:
        """
        生成长度为 self.frame_length 的汉宁窗。如果已经生成且长度匹配，则复用。
        """
        if self.window is None or len(self.window) != self.frame_length:
            self.window = np.hamming(self.frame_length)
            # 在帧上乘了一个汉宁窗（Hanning window），而汉宁窗在首尾两个点本身就是零使得帧首尾为1
            self.window[0] = self.window[-1] = 1.0
        return self.window

    def frame_data(self) -> np.ndarray:
        """
        对所有文件数据按 JSON 中 frame 配置做分帧：
          - 裁剪到最短文件长度
          - frame_length: 每帧长度
          - overlap: 重叠率
          - apply_window: 是否加汉宁窗
          - export_csv: 是否导出展平后的所有帧数据到 CSV（不含标签）
          - csv_path: 如果 export_csv=true，保存文件路径

        返回值：numpy 数组，shape = (total_frames, frame_length, num_channels)
        """
        if not self.raw_data_list:
            raise RuntimeError("请先调用 read_data() 读取原始信号。")
        print("分帧—加窗...")
        frame_cfg = self.config.get("frame", {})
        frame_length = frame_cfg.get("frame_length", self.frame_length)
        overlap = frame_cfg.get("overlap", self.overlap)
        apply_window = frame_cfg.get("apply_window", True)
        export_csv = frame_cfg.get("export_csv", False)
        csv_path = frame_cfg.get("csv_path", None)

        # 找到所有文件的最小样本数
        lengths = [data.shape[0] for data in self.raw_data_list]
        min_len = min(lengths)
        print(f"所有文件样本数：{lengths}，最短长度：{min_len}")

        # 统一裁剪
        trimmed = [data[:min_len] if data.ndim > 1 else data[:min_len].reshape(-1,1)
                   for data in self.raw_data_list]

        # 假设通道数一致
        num_channels = trimmed[0].shape[1]
        step = int(frame_length * (1 - overlap))
        if step <= 0:
            raise ValueError("overlap 设置过大，导致步长 step <= 0，请检查 overlap 值 (<1)。")

        num_frames = (min_len - frame_length) // step + 1
        if num_frames <= 0:
            raise ValueError(f"数据长度 {min_len} 小于帧长 {frame_length}，无法分帧。")

        window = self._compute_window() if apply_window else np.ones(frame_length)

        all_frames = np.zeros((0, frame_length, num_channels))
        for data in trimmed:
            frames = np.zeros((num_frames, frame_length, num_channels), dtype=data.dtype)
            for i in range(num_frames):
                seg = data[i * step: i * step + frame_length]
                if apply_window:
                    seg = seg * window.reshape(-1,1)
                # 保留6位小数
                frames[i] = np.round(seg, 6)
            all_frames = np.concatenate([all_frames, frames], axis=0)

        if export_csv:
            if not csv_path:
                raise ValueError("frame.export_csv=True 时，需要提供 frame.csv_path。")
            flat = all_frames.reshape(all_frames.shape[0], frame_length * num_channels)
            pd.DataFrame(flat).to_csv(csv_path, index=False)
            print(f"所有帧数据 CSV 已导出到 {csv_path}")

        self.frame_length = frame_length
        self.overlap = overlap
        return all_frames

    def assign_labels(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        给所有帧按文件来源打标签：
          - 从 init 时传入的 file_label_map 获取对应标签
          - 从 JSON 中读取 label.label_map、label.num_classes
          - shuffle: 是否打乱帧与标签顺序
          - export_csv: 是否导出展平帧 + 标签 CSV
          - csv_path: 如果 export_csv=true，保存文件路径

        返回：
          - frames_out: (total_frames, frame_length, num_channels)
          - labels_out: (total_frames,) 整数标签数组
        """
        print("给帧打标签...")
        label_cfg = self.config.get("label", {})
        label_map: dict = label_cfg.get("label_map", {})
        num_classes: int = label_cfg.get("num_classes", None)
        shuffle_flag: bool = label_cfg.get("shuffle", True)
        export_csv: bool = label_cfg.get("export_csv", False)
        csv_path: Optional[str] = label_cfg.get("csv_path", None)

        if num_classes is None or num_classes <= 0:
            raise ValueError("请在 JSON 中指定合法的 label.num_classes（正整数）。")

        # 生成标签列表，按照文件顺序生成对应数量
        per_file_count = frames.shape[0] // len(self.file_label_map)
        labels = []
        for raw_value in self.file_label_map.values():
            if isinstance(raw_value, str):
                if raw_value not in label_map:
                    raise KeyError(f"label_value='{raw_value}' 未在 label_map 中找到。")
                idx = label_map[raw_value]
            elif isinstance(raw_value, int):
                idx = raw_value
            else:
                raise TypeError("标签值必须是字符串或整数索引。")
            if not (0 <= idx < num_classes):
                raise ValueError(f"标签索引 {idx} 超出范围 [0, {num_classes-1}]。")
            labels.extend([idx] * per_file_count)

        labels_arr = np.array(labels, dtype=int)
        flat_frames = frames.reshape(frames.shape[0], -1)
        paired = np.concatenate([flat_frames, labels_arr.reshape(-1,1)], axis=1)

        if shuffle_flag:
            perm = np.random.permutation(paired.shape[0])
            paired = paired[perm]

        if export_csv:
            if not csv_path:
                raise ValueError("label.export_csv=True 时，需要提供 label.csv_path。")
            pd.DataFrame(paired).to_csv(csv_path, index=False)
            print(f"带标签数据 CSV 已导出到 {csv_path}")

        frames_flat = paired[:, :-1]
        labels_out = paired[:, -1].astype(int)
        frames_out = frames_flat.reshape(-1, self.frame_length, frames.shape[2])
        return frames_out, labels_out

    def split_dataset(
        self,
        frames: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray]]:
        """
        按 JSON 中 split 配置将 frames+labels 切分为 train/test/val：
        与原代码逻辑一致
        """
        print("划分数据集...")
        split_cfg = self.config.get("split", {})
        ratios: List[float] = split_cfg.get("ratios", [0.7,0.2,0.1])
        shuffle_flag: bool = split_cfg.get("shuffle", True)
        export_excel: bool = split_cfg.get("export_excel", False)
        excel_path: Optional[str] = split_cfg.get("excel_path", None)

        if len(ratios) != 3 or abs(sum(ratios)-1)>1e-6:
            raise ValueError("split.ratios 必须是长度为 3 且和为 1 的列表。")

        N = frames.shape[0]
        flat = frames.reshape(N, -1)
        data_all = np.concatenate([flat, labels.reshape(-1,1)], axis=1)
        if shuffle_flag:
            data_all = data_all[np.random.permutation(N)]

        n_train = int(N * ratios[0])
        n_test = int(N * ratios[1])
        n_val = N - n_train - n_test

        train = data_all[:n_train]
        test = data_all[n_train:n_train+n_test]
        val = data_all[n_train+n_test:]

        def unpack(arr):
            if arr.size==0:
                return np.empty((0,self.frame_length,frames.shape[2])), np.empty((0,),int)
            Xf = arr[:,:-1]
            y = arr[:,-1].astype(int)
            X = Xf.reshape(-1,self.frame_length,frames.shape[2])
            return X,y

        train_set = unpack(train)
        test_set = unpack(test)
        val_set = unpack(val)

        if export_excel:
            if not excel_path:
                raise ValueError("split.export_excel=True 时，需要提供 split.excel_path。")
            with pd.ExcelWriter(excel_path) as writer:
                pd.DataFrame(train).to_excel(writer, sheet_name="train", index=False)
                pd.DataFrame(test).to_excel(writer, sheet_name="test", index=False)
                pd.DataFrame(val).to_excel(writer, sheet_name="val", index=False)
                print("导出数据集到 Excel...")

        return train_set, test_set, val_set

    @staticmethod
    def save_frames_to_csv(
        frames: np.ndarray,
        labels: Optional[np.ndarray] = None,
        csv_path: str = "frames_with_labels_static.csv"
    ):
        """
        单独将 frames（(N, frame_length, num_channels)）及 labels（(N,)）导出到 CSV。
        每行：展平后的帧数据 + 对应标签。如果 labels=None，则仅导出帧数据。
        """
        N, fl, ch = frames.shape
        flat = frames.reshape(N, fl*ch)
        if labels is not None:
            paired = np.concatenate([flat, labels.reshape(-1,1)], axis=1)
            pd.DataFrame(paired).to_csv(csv_path, index=False)
        else:
            pd.DataFrame(flat).to_csv(csv_path, index=False)

    @staticmethod
    def save_numpy(
        data: np.ndarray,
        npy_path: str = "data.npy"
    ):
        """
        将任意 numpy 数组保存为 .npy 文件。
        """
        np.save(npy_path, data)

    @staticmethod
    def load_numpy(
        npy_path: str = "data.npy"
    ) -> np.ndarray:
        """
        从 .npy 文件中加载 numpy 数组。
        """
        return np.load(npy_path)


if __name__ == "__main__":
    # 使用示例：
    sig1_path = "C:/Users/lenovo/Desktop/paper/Fault-Diagnosis/data/JNU/ib600_2.csv"
    sig2_path = "C:/Users/lenovo/Desktop/paper/Fault-Diagnosis/data/JNU/ob600_2.csv"
    sig3_path = "C:/Users/lenovo/Desktop/paper/Fault-Diagnosis/data/JNU/tb600_2.csv"
    files = {sig1_path: "内圈故障", sig2_path: "外圈故障", sig3_path: "滚动体故障"}
    pre = DataPre(file_label_map=files, config_path="C:/Users/lenovo/Desktop/paper/Fault-Diagnosis/configs/data_pre.json")
    pre.read_data()
    frames = pre.frame_data()
    frames_labeled, labels = pre.assign_labels(frames)
    (train_X, train_y), (test_X, test_y), (val_X, val_y) = pre.split_dataset(frames_labeled, labels)
    print(f"train: {train_X.shape}, test: {test_X.shape}, val: {val_X.shape}")
