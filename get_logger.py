import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(filename: str) -> logging.Logger:
    """
    根据文件名返回一个配置好的logger实例，用于记录该文件的日志信息
    
    Args:
        filename (str): 调用该函数的文件名
    
    Returns:
        logging.Logger: 配置好的logger实例
    """
    # 创建logger实例
    logger = logging.getLogger(filename)
    
    # 如果logger已经有handler则直接返回
    if logger.handlers:
        return logger
        
    # 设置日志级别
    logger.setLevel(logging.INFO)
    
    # 创建logs目录(如果不存在)
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # 只记录每次运行的日志，如果logs目录存在且不为空,则删除目录下所有文件
    if os.path.exists('logs') and os.listdir('logs'):
        for file in os.listdir('logs'):
            file_path = os.path.join('logs', file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Error deleting {file_path}: {e}')
        
    
    # 从完整路径中提取文件名
    base_filename = os.path.basename(filename)
    log_file = os.path.join('logs', f'{os.path.splitext(base_filename)[0]}.log')
    # 创建文件处理器,按文件大小轮转
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# # 创建一个全局logger实例用于记录通用日志
# logger = get_logger('test')

# # 测试
# if __name__ == '__main__':
#     logger.info('This is a test message')
