"""
TensorRT 类型存根（简化版）
手动维护的常用 API 类型提示

使用方法：
1. 将此文件放在项目根目录的 typings/tensorrt.pyi
2. VS Code 会自动识别
"""

from typing import List, Optional, Any, Union
import numpy as np

class Logger:
    class Severity:
        INTERNAL_ERROR: int
        ERROR: int
        WARNING: int
        INFO: int
        VERBOSE: int
    
    def __init__(self, severity: int = ...): ...

class Dims:
    def __init__(self, dims: List[int]): ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> int: ...

class Weights:
    def __init__(self, a: np.ndarray): ...

class ITensor:
    @property
    def shape(self) -> Dims: ...
    
    @property
    def name(self) -> str: ...
    
    @name.setter
    def name(self, value: str) -> None: ...
    
    @property
    def dtype(self) -> DataType: ...

class ILayer:
    @property
    def name(self) -> str: ...
    
    @name.setter
    def name(self, value: str) -> None: ...
    
    def get_output(self, index: int) -> ITensor: ...

class INetworkDefinition:
    def add_input(self, name: str, dtype: DataType, shape: Union[Dims, tuple]) -> ITensor: ...
    def add_constant(self, shape: Dims, weights: Weights) -> ILayer: ...
    def add_matrix_multiply(self, input0: ITensor, op0: MatrixOperation, 
                           input1: ITensor, op1: MatrixOperation) -> ILayer: ...
    def add_elementwise(self, input1: ITensor, input2: ITensor, 
                       op: ElementWiseOperation) -> ILayer: ...
    def add_shuffle(self, input: ITensor) -> IShuffleLayer: ...
    def add_slice(self, input: ITensor, start: Dims, shape: Dims, stride: Dims) -> ILayer: ...
    def add_softmax(self, input: ITensor) -> ISoftMaxLayer: ...
    def add_reduce(self, input: ITensor, op: ReduceOperation, 
                  axes: int, keep_dims: bool) -> ILayer: ...
    def add_activation(self, input: ITensor, type: ActivationType) -> ILayer: ...
    def add_unary(self, input: ITensor, op: UnaryOperation) -> ILayer: ...
    def mark_output(self, tensor: ITensor) -> None: ...

class IShuffleLayer(ILayer):
    @property
    def reshape_dims(self) -> Dims: ...
    
    @reshape_dims.setter
    def reshape_dims(self, dims: Dims) -> None: ...
    
    @property
    def second_transpose(self) -> Permutation: ...
    
    @second_transpose.setter
    def second_transpose(self, perm: Permutation) -> None: ...

class ISoftMaxLayer(ILayer):
    @property
    def axes(self) -> int: ...
    
    @axes.setter
    def axes(self, axes: int) -> None: ...

class Permutation:
    def __init__(self, perm: List[int]): ...

class IBuilderConfig:
    def set_flag(self, flag: BuilderFlag) -> bool: ...
    def set_memory_pool_limit(self, pool: MemoryPoolType, size: int) -> bool: ...
    @property
    def profiling_verbosity(self) -> ProfilingVerbosity: ...
    
    @profiling_verbosity.setter
    def profiling_verbosity(self, verbosity: ProfilingVerbosity) -> None: ...

class Builder:
    def __init__(self, logger: Logger): ...
    def create_network(self, flags: int = ...) -> INetworkDefinition: ...
    def create_builder_config(self) -> IBuilderConfig: ...
    def build_serialized_network(self, network: INetworkDefinition, 
                                 config: IBuilderConfig) -> Optional[bytes]: ...

class ICudaEngine:
    @property
    def num_io_tensors(self) -> int: ...
    
    def get_tensor_name(self, index: int) -> str: ...
    def get_tensor_mode(self, name: str) -> TensorIOMode: ...
    def get_tensor_shape(self, name: str) -> Dims: ...
    def get_tensor_dtype(self, name: str) -> DataType: ...
    def create_execution_context(self) -> IExecutionContext: ...

class IExecutionContext:
    def set_input_shape(self, name: str, shape: Union[Dims, tuple]) -> bool: ...
    def get_tensor_shape(self, name: str) -> Dims: ...
    def set_tensor_address(self, name: str, address: int) -> bool: ...
    def execute_async_v3(self, stream_handle: int) -> bool: ...

class Runtime:
    def __init__(self, logger: Logger): ...
    def deserialize_cuda_engine(self, serialized_engine: bytes) -> ICudaEngine: ...

class NetworkDefinitionCreationFlag:
    EXPLICIT_BATCH: int

class DataType:
    FLOAT: DataType
    HALF: DataType
    INT8: DataType
    INT32: DataType
    BOOL: DataType

class MatrixOperation:
    NONE: MatrixOperation
    TRANSPOSE: MatrixOperation
    VECTOR: MatrixOperation

class ElementWiseOperation:
    SUM: ElementWiseOperation
    PROD: ElementWiseOperation
    MAX: ElementWiseOperation
    MIN: ElementWiseOperation
    SUB: ElementWiseOperation
    DIV: ElementWiseOperation
    POW: ElementWiseOperation

class ReduceOperation:
    SUM: ReduceOperation
    PROD: ReduceOperation
    MAX: ReduceOperation
    MIN: ReduceOperation
    AVG: ReduceOperation

class ActivationType:
    RELU: ActivationType
    SIGMOID: ActivationType
    TANH: ActivationType
    LEAKY_RELU: ActivationType
    ELU: ActivationType
    SELU: ActivationType
    SOFTSIGN: ActivationType
    SOFTPLUS: ActivationType
    CLIP: ActivationType
    HARD_SIGMOID: ActivationType
    SCALED_TANH: ActivationType
    THRESHOLDED_RELU: ActivationType

class UnaryOperation:
    EXP: UnaryOperation
    LOG: UnaryOperation
    SQRT: UnaryOperation
    RECIP: UnaryOperation
    ABS: UnaryOperation
    NEG: UnaryOperation
    SIN: UnaryOperation
    COS: UnaryOperation
    TAN: UnaryOperation
    SINH: UnaryOperation
    COSH: UnaryOperation
    ASIN: UnaryOperation
    ACOS: UnaryOperation
    ATAN: UnaryOperation
    ASINH: UnaryOperation
    ACOSH: UnaryOperation
    ATANH: UnaryOperation
    CEIL: UnaryOperation
    FLOOR: UnaryOperation
    ERF: UnaryOperation
    NOT: UnaryOperation
    SIGN: UnaryOperation
    ROUND: UnaryOperation

class BuilderFlag:
    INT8: BuilderFlag
    FP16: BuilderFlag
    DEBUG: BuilderFlag
    GPU_FALLBACK: BuilderFlag
    STRICT_TYPES: BuilderFlag
    REFIT: BuilderFlag

class TensorIOMode:
    INPUT: TensorIOMode
    OUTPUT: TensorIOMode

class MemoryPoolType:
    WORKSPACE: MemoryPoolType
    DLA_MANAGED_SRAM: MemoryPoolType
    DLA_LOCAL_DRAM: MemoryPoolType
    DLA_GLOBAL_DRAM: MemoryPoolType

class ProfilingVerbosity:
    LAYER_NAMES_ONLY: ProfilingVerbosity
    DETAILED: ProfilingVerbosity
    NONE: ProfilingVerbosity

def nptype(dtype: DataType) -> type: ...
