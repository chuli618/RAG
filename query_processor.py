from abc import ABC, abstractmethod
from query_context import QueryContext


class QueryProcessor(ABC):
    """
    QueryProcessor类定义了一个处理查询的抽象类或接口。
    它的主要作用是定义了一个process方法，该方法接受一个QueryContext对象作为参数，并对其进行处理。
    这个类旨在被继承，具体的处理逻辑将在子类中实现。

    该类设计为责任链模式的一部分，允许多个处理器按顺序处理同一个QueryContext对象。
    具体的处理器可能包括但不限于：
    1. 预处理query
    2. 去vector store查询相关文档
    3. 对查询得到的文档进行过滤
    4. 对查询得到的文档进行rerank
    """

    def __init__(self):
        # @Prompt
        # 这里默认为继承类的class name
        self.name = self.__class__.__name__  # 设置name属性为类名
        self.hide_execution_status = False  # 增加默认属性，是否隐藏执行状态，默认为false
        self.strong_dependency = True  # 是否强依赖，默认为true

    @abstractmethod
    def process(self, context: QueryContext) -> None:
        """
        处理给定的QueryContext对象。

        :param context: QueryContext对象，包含了查询的相关信息和数据。
        """
        pass