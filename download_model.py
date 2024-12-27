import warnings
import os
import subprocess
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

def download_model(repo_name='BAAI', model_name='bge-base-zh-v1.5'):
    """
        从ModelScope下载预训练模型到本地'models/'目录。

        参数:
            repo_name (str): 模型仓库名称，默认为'BAAI'。
            model_name (str): 模型名称，默认为'bge-base-zh-v1.5'。
    """
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 检查 models 目录是否存在，不存在则创建
    models_dir = os.path.join(current_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    # 下载模型
    subprocess.run([
        'modelscope', 'download',
        '--model', f'{repo_name}/{model_name}',
        '--local_dir', os.path.join(models_dir, model_name)
    ])

# 测试代码
if __name__ == '__main__':
    # download_model(repo_name='BAAI', model_name='bge-base-zh-v1.5')
    download_model(repo_name='BAAI', model_name='bge-large-zh-v1.5')
    download_model(repo_name='Qwen', model_name='Qwen2.5-7B-Instruct')
    download_model(repo_name='BAAI', model_name='bge-reranker-large')