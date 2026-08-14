from .base import BaseASRModel
from .whisper_model import WhisperModel
from .qwen_model import QwenASRModel
from .revolab_api_model import RevolabAPIModel
from .gemini_model import GeminiModel
from .elevenlabs_model import ElevenLabsModel
from .deepgram_model import DeepgramModel
from .assemblyai_model import AssemblyAIModel
from .ilmu_model import IlmuModel
from .qwen_audio_model import QwenAudioModel

MODEL_REGISTRY: dict[str, type[BaseASRModel]] = {
    "whisper": WhisperModel,
    "qwen": QwenASRModel,
    "revolab-api": RevolabAPIModel,
    "gemini": GeminiModel,
    "elevenlabs": ElevenLabsModel,
    "deepgram": DeepgramModel,
    "assemblyai": AssemblyAIModel,
    "ilmu": IlmuModel,
    "qwen-audio": QwenAudioModel,
}

__all__ = [
    "BaseASRModel",
    "WhisperModel",
    "QwenASRModel",
    "RevolabAPIModel",
    "GeminiModel",
    "ElevenLabsModel",
    "DeepgramModel",
    "AssemblyAIModel",
    "IlmuModel",
    "QwenAudioModel",
    "MODEL_REGISTRY",
]
