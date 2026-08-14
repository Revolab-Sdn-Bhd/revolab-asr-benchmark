from setuptools import setup, find_packages

# Versions are deliberately unpinned — install the latest of everything.
# requirements/*.txt mirror these lists for pip -r users.
BACKENDS = {
    "whisper": ["torch", "transformers", "accelerate"],
    "qwen": ["torch", "transformers", "accelerate", "librosa"],
    "gemini": ["google-genai"],
    "elevenlabs": ["elevenlabs"],
    "deepgram": ["deepgram-sdk"],
    "assemblyai": ["assemblyai"],
}

setup(
    name="asr-benchmark",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "datasets",
        "soundfile",
        "numpy",
        "requests",
        "python-dotenv",
        "torch",       # torchcodec imports it without declaring it
        "torchcodec",  # datasets>=4 decodes audio through it
    ],
    extras_require={
        **BACKENDS,
        "all": sorted({dep for deps in BACKENDS.values() for dep in deps}),
    },
)
