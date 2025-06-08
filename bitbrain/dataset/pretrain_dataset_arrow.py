import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

import glob
import datasets 

#! 直接从处理好的数据集中读取
class PretrainDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.dataset = self.load_data(data_path)
        print(f"数据路径: {data_path}")
        print(f"文件列表: {os.listdir(data_path)}")

    def load_data(self, data_path):
        # 直接从预处理好的数据集加载
        try:
            dataset = datasets.load_from_disk(data_path)
            print(f"成功加载预处理数据集，包含 {len(dataset)} 个样本")
            return dataset
        except Exception as e:
            raise ValueError(f"无法从 {data_path} 加载预处理数据集: {e}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
        
        # 确保 loss_mask 是张量
        if isinstance(sample['loss_mask'], torch.Tensor):
            loss_mask = sample['loss_mask']
        else:
            loss_mask = torch.tensor(sample['loss_mask'], dtype=torch.long)
        
        # 构建训练数据
        X = input_ids[:-1]  # 去掉最后一个 token 作为输入
        Y = input_ids[1:]   # 去掉第一个 token 作为目标
        loss_mask = loss_mask[1:]  # 对齐预测位置
        
        return X, Y, loss_mask

#! 从多个数据集路径中混合读取数据
class PretrainDataset_mix(Dataset):
    def __init__(self, data_paths, ratios, stopping_strategy="all_exhausted"):
        """
        初始化混合预训练数据集。

        参数:
            data_paths (list[str]): 一个包含多个数据集路径的列表。
                                    每个路径指向一个由 datasets.load_from_disk 可加载的目录。
            ratios (list[float]): 一个列表，包含对应于 data_paths 中每个数据集的采样比例。
                                  总和应为 1.0。如果不为1，将进行归一化。
            stopping_strategy (str, optional): datasets.interleave_datasets 的停止策略。
                                               可选值为 "first_exhausted" 或 "all_exhausted"。
                                               默认为 "all_exhausted"。
        """
        super().__init__()

        # 校验输入参数
        if not data_paths or not ratios:
            raise ValueError("data_paths 和 ratios 列表不能为空。")
        if len(data_paths) != len(ratios):
            raise ValueError("data_paths 和 ratios 列表的长度必须相同。")
        if not all(isinstance(path, str) for path in data_paths):
            raise ValueError("data_paths 列表中的所有元素都必须是字符串（路径）。")
        if not all(isinstance(ratio, (int, float)) for ratio in ratios):
            raise ValueError("ratios 列表中的所有元素都必须是数字。")

        # 检查并归一化ratios
        if abs(sum(ratios) - 1.0) > 1e-6:
            print(f"警告: 提供的比例 {ratios} 总和不为 1。将进行归一化处理。")
            total_ratio = sum(ratios)
            if total_ratio == 0:
                raise ValueError("ratios 的总和不能为零。")
            ratios = [r / total_ratio for r in ratios]
            print(f"归一化后的比例: {ratios}")

        self.data_paths = data_paths
        self.ratios = ratios
        
        # 加载并混合数据集
        self.dataset = self._load_and_interleave_datasets(data_paths, ratios, stopping_strategy)
        
        print(f"成功加载并混合了 {len(data_paths)} 个数据集。")
        print(f"混合后数据集的总样本数: {len(self.dataset)}")

    def _load_and_interleave_datasets(self, data_paths, ratios, stopping_strategy):
        """
        加载所有指定的数据集，并使用指定的比例和停止策略将它们交错混合。
        """
        individual_datasets = [] # 用于存放加载的各个数据集
        for i, path in enumerate(data_paths):
            try:
                # 从磁盘加载单个预处理好的数据集
                ds = datasets.load_from_disk(path)
                print(f"成功加载数据集 '{path}'，包含 {len(ds)} 个样本。")
                individual_datasets.append(ds)
            except Exception as e:
                # 如果加载失败，抛出错误
                raise ValueError(f"无法从路径 '{path}' 加载预处理数据集: {e}")

        if not individual_datasets:
            raise ValueError("未能成功加载任何数据集。")

        # 将ratios转换为浮点数，以防万一是整数
        float_ratios = [float(r) for r in self.ratios]

        try:
            # 使用 datasets.interleave_datasets 进行混合
            # 这个函数会根据 probabilities (即我们的 ratios) 从 individual_datasets 中抽取数据
            mixed_dataset = datasets.interleave_datasets(
                individual_datasets,  # 包含所有待混合数据集的列表
                probabilities=float_ratios,  # 每个数据集被选中的概率列表
                stopping_strategy=stopping_strategy, # 停止策略
                # seed=42  # 如果需要可复现的混合顺序，可以设置一个种子
            )
            return mixed_dataset
        except Exception as e:
            # 如果混合过程中出错，抛出错误
            raise ValueError(f"混合数据集时发生错误: {e}")

    def __len__(self):
        # 返回混合后数据集的总长度
        return len(self.dataset)

    def __getitem__(self, index):
        # 从混合数据集中获取一个样本
        sample = self.dataset[index]
        
        # 将 'input_ids' 转换为 PyTorch 张量
        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
        
        # 确保 'loss_mask' 是 PyTorch 张量
        if isinstance(sample['loss_mask'], torch.Tensor):
            loss_mask = sample['loss_mask']
        else:
            loss_mask = torch.tensor(sample['loss_mask'], dtype=torch.long)
        
        # 构建模型的输入 (X) 和目标 (Y)
        # X 是去掉最后一个 token 的 input_ids
        X = input_ids[:-1]
        # Y 是去掉第一个 token 的 input_ids (作为预测目标)
        Y = input_ids[1:]
        # loss_mask 也需要相应地调整，去掉第一个元素，使其与 Y 对齐
        loss_mask = loss_mask[1:]  
        
        return X, Y, loss_mask