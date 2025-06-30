from modelscope.hub.api import HubApi,

api.upload_folder(
    repo_id=f"{owner_name}/{model_name}",
    folder_path='/path/to/your_model_dir',
    commit_message='upload model folder to repo',
)