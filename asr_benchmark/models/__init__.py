from .base import BaseASRModel
from .whisper_model import WhisperModel
from .qwen_model import QwenASRModel
from .revolab_api_model import RevolabAPIModel
from .zipformer_model import ZipformerModel
from .gemini_model import GeminiModel
from .elevenlabs_model import ElevenLabsModel
from .deepgram_model import DeepgramModel
from .assemblyai_model import AssemblyAIModel
from .ilmu_model import IlmuModel
from .qwen3_asr_gguf_model import Qwen3AsrGgufModel
from .qwen_audio_model import QwenAudioModel

MODEL_REGISTRY: dict[str, type[BaseASRModel]] = {
    "whisper": WhisperModel,
    "qwen": QwenASRModel,
    "revolab-api": RevolabAPIModel,
    "zipformer": ZipformerModel,
    "gemini": GeminiModel,
    "elevenlabs": ElevenLabsModel,
    "deepgram": DeepgramModel,
    "assemblyai": AssemblyAIModel,
    "ilmu": IlmuModel,
    "qwen3-asr-gguf": Qwen3AsrGgufModel,
    "qwen-audio": QwenAudioModel,
}

__all__ = [
    "BaseASRModel",
    "WhisperModel",
    "QwenASRModel",
    "RevolabAPIModel",
    "ZipformerModel",
    "GeminiModel",
    "ElevenLabsModel",
    "DeepgramModel",
    "AssemblyAIModel",
    "IlmuModel",
    "Qwen3AsrGgufModel",
    "QwenAudioModel",
    "MODEL_REGISTRY",
]
