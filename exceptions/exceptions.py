class BaseSimCLRException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""
    # 核心网络选择失效时出现

class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""
