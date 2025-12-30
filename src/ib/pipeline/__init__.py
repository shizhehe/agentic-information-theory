from .compression_prediction import CompressionPredictionProtocol, create_compression_protocol
from .wildchat_compression import WildchatCompressionProtocol
from .fineweb_compression import FinewebCompressionProtocol

__all__ = [
    "CompressionPredictionProtocol",
    "WildchatCompressionProtocol", 
    "FinewebCompressionProtocol",
    "create_compression_protocol",
]