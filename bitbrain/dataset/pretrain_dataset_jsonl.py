import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
# 导入 Hugging Face 和 ModelScope 的数据集库
from datasets import load_dataset as hf_load_dataset 
from modelscope.msdatasets import MsDataset
from loguru import logger
import ujson
import json
import os # 新增：导入os模块，它提供了与操作系统交互的功能，比如文件路径操作
import concurrent.futures

# todo 只能读取单个文件
class PretrainDataset_v1(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.samples 会存储从 load_data 方法加载回来的所有数据
        self.samples = self.load_data(data_path)

    # 我们来修改这个方法
    # 参数 'path' 现在可以是一个指向目录的路径，也可以是一个指向单个 .jsonl 文件的路径
    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = input_ids[:-1].detach().clone().long()
        Y = input_ids[1:].detach().clone().long()
        loss_mask = loss_mask[1:].detach().clone().long()
        return X, Y, loss_mask


# todo 单线程加载多个jsonl文件
class PretrainDataset_v2(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        logger.info(f"开始加载数据，路径: {path}")

        if os.path.isdir(path):
            # 如果是目录，遍历目录下所有jsonl文件
            for filename in os.listdir(path):
                if filename.endswith('.jsonl'):
                    file_path = os.path.join(path, filename)
                    logger.info(f"加载文件: {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = ujson.loads(line.strip())
                            samples.append(data)
        else:
            # 如果是单个文件，直接读取
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = input_ids[:-1].detach().clone().long()
        Y = input_ids[1:].detach().clone().long()
        loss_mask = loss_mask[1:].detach().clone().long()
        return X, Y, loss_mask

# todo 使用多线程加载jsonl文件
# todo 有BUG，启动多线程加载时会卡住
class PretrainDataset_v3(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data_with_multiprocessing(data_path)

    def load_data(self, data_path):
        samples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = ujson.loads(line.strip())
                        samples.append(data)
                    except Exception as e:
                        logger.error(f"Error loading JSON line: {e}")
                        continue
            logger.info(f"成功加载文件: {data_path}")
        return samples
    
    def load_data_with_multiprocessing(self, path: str) -> list[dict]:
        # if os.path.isdir(path):
        #     return self.load_data(path)
        
        if not os.path.isdir(path):
            raise ValueError(f"Path is not a directory: {path}")
        
        jsonl_files = [
            os.path.join(path, f) for f in os.listdir(path)
            if f.endswith('.jsonl')
        ]
        
        if not jsonl_files:
            return []
        
        all_samples = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.cpu_count())) as executor:
            future_to_file =  {
                executor.submit(self.load_data, data_path): data_path
                for data_path in jsonl_files
            }
            for future in concurrent.futures.as_completed(future_to_file):
                try:
                    data = future.result()
                    all_samples.extend(data)
                except Exception as e:
                    logger.error(f"Error loading JSONL file: {e}")
        return all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = input_ids[:-1].detach().clone().long()
        Y = input_ids[1:].detach().clone().long()
        loss_mask = loss_mask[1:].detach().clone().long()
        return X, Y, loss_mask