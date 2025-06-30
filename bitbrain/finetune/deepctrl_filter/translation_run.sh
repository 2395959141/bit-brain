# 全局参数
project_name: 'translation_sft'
dataset_path: '/DATA/disk2/yuhang/.cache/bit_brain_data/step1_unified_format/'
export_path: '/DATA/disk2/yuhang/.cache/bit_brain_data/sft_llamafactory/translation/translation.jsonl'  # 修正拼写错误
text_keys: 'instruction,output'  # 关键修改：翻译数据通常包含双语字段
np: 30

# 处理流程优化
process:
  # 预处理
  - chinese_convert_mapper:
      mode: 't2s'
  - clean_links_mapper:  # 保留链接清理
  - whitespace_normalization_mapper:
  - remove_specific_chars_mapper:
      chars_to_remove: '◆●■►▼▲▴∆▻▷❖♡□'

  # 语言过滤（关键修改）
  - language_id_score_filter:
      lang: ['zh', 'en']
      min_score: 0.9  # 确保语言纯净度

  # 文本质量过滤
  - text_length_filter:
      min_len: 20  # 适当提高最小长度
      max_len: 1000
  - perplexity_filter:
      lang: zh
      max_ppl: 2000  # 放宽困惑度阈值
  - word_repetition_filter:
      lang: zh
      max_ratio: 0.3  # 严格限制重复率

  # 翻译数据专用处理
  - **remove_repeat_sentences_mapper**:  # 新增去重
      min_repeat_sentence_length: 2
  - **sentence_split_mapper**:  # 新增分句处理
      lang: 'zh'
      max_length: 100

  # 去重优化
  - document_simhash_deduplicator:
      hamming_distance: 4
      tokenization: 'character'
