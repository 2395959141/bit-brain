#
#! 参考自项目：Steel-LLM
#! 具体代码：https://github.com/zhanshijinwat/Steel-LLM/blob/main/data/pretrain_data_prepare/step2/run_step2.sh
DJ_PATH=/home/chenyuhang/data-juicer/
TXT_YAML=txt_process.yaml

#python $DJ_PATH/tools/process_data.py --config $CODE_YAML 
python $DJ_PATH/tools/process_data.py --config $TXT_YAML 
