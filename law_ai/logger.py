# coding: utf-8
import logging
import sys
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_level = record.levelname
        color = self.COLORS.get(log_level, '')
        
        # 添加时间戳和颜色
        record.levelname = f"{color}[{log_level}]{self.RESET}"
        
        formatter = logging.Formatter(
            '%(levelname)s [%(name)s] %(message)s'
        )
        return formatter.format(record)


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 移除已有的处理器，避免重复
    logger.handlers = []
    
    # 添加控制台处理器
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    formatter = ColoredFormatter()
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger


# 创建全局日志记录器
retriever_logger = setup_logger("Retriever")
chain_logger = setup_logger("Chain")
app_logger = setup_logger("App")
