import warnings
import os
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

def download_model(repo_name='BAAI', model_name='bge-base-zh-v1.5'):
    """
    从ModelScope下载预训练模型到本地目录。

    参数:
        repo_name (str): 模型仓库名称，默认为'BAAI'。
        model_name (str): 模型名称，默认为'bge-base-zh-v1.5'。

    功能:
        - 使用modelscope命令行工具下载指定的模型
        - 将模型保存到'models/{model_name}'目录下
        - 支持自定义仓库名称和模型名称
    """
    import subprocess
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 创建models目录
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