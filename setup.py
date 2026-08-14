from setuptools import setup, find_packages

setup(
    name="asr-benchmark",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "datasets>=2.19.0",
        "soundfile>=0.12.1",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "whisper": ["torch>=2.2.0", "transformers>=4.40.0", "accelerate>=0.29.0"],
        "qwen": ["torch>=2.2.0", "transformers>=4.45.0", "accelerate>=0.29.0", "librosa>=0.10.0"],
        "gemini": ["google-generativeai>=0.7.0"],
        "elevenlabs": ["elevenlabs>=1.0.0"],
        "all": [
            "torch>=2.2.0", "transformers>=4.45.0", "accelerate>=0.29.0",
            "librosa>=0.10.0",
            "google-generativeai>=0.7.0", "elevenlabs>=1.0.0",
        ],
    },
)
