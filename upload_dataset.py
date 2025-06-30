from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = '61ea5049-692b-4936-83e1-0a871e72aaba'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

owner_name = 'hh2395959141'
dataset_name = 'chinese_fineweb_v2_tokenized'

api.upload_folder(
    repo_id=f"{owner_name}/{dataset_name}",
    folder_path='/DATA/disk2/yuhang/.cache/bit_brain_data/pretrain',
    commit_message='feat: first commit',
    repo_type = 'dataset',
    max_workers = 30
)
