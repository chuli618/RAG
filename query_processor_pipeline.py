from query_context import QueryContext
from query_processor import QueryProcessor
import copy
import traceback
import json

class QueryProcessorPipeline:
    def __init__(self, processors=None):
        """
        初始化一个查询处理管道。

        :param processors: QueryProcessor对象的列表，按顺序处理QueryContext。
        """
        self.processors = processors if processors is not None else []

    def add_processor(self, processor):
        """
        向管道中添加一个新的处理器。

        :param processor: QueryProcessor对象。
        """
        self.processors.append(processor)

    def process(self, context: QueryContext):
        """
        处理传入的QueryContext对象，按顺序通过所有处理器。

        :param context: QueryContext对象。
        """
        pass