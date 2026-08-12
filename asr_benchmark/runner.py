"""Core evaluation loop — model-agnostic, model-oriented."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import MODEL_REGISTRY, BaseASRModel
from .normalizer import EnglishTextNormalizer, BasicTextNormalizer, MalayTextNormalizer
from .utils.data import load_dataset_hf
from .utils.manifest import write_manifest
from .utils.metrics import compute_all_metrics, _align

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_type: str           # key in MODEL_REGISTRY
    model_id: str             # HF repo id, local path, or checkpoint name
    language: str = "en"      # BCP-47 tag — drives normalizer + model language hint
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    dataset_name: str
    splits: list[str]         # evaluated in sequence with one model load
    audio_column: str = "audio"
    text_column: str = "sentence"
    normalized_text_column: str | None = None  # dataset's own pre-normalized ref (e.g. "normalized_text")
    metadata_columns: list[str] = field(default_factory=list)  # e.g. ["category"]
    subset: str | None = None
    max_samples: int | None = None
    streaming: bool = True
    category_filter: str | None = None  # only evaluate rows where metadata["category"] matches

    @classmethod
    def single(cls, dataset_name: str, split: str, **kwargs) -> "DatasetConfig":
        return cls(dataset_name=dataset_name, splits=[split], **kwargs)


# ---------------------------------------------------------------------------
# Keep EvalConfig as a thin compatibility shim so existing callers don't break
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    model_type: str
    model_id: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    dataset_name: str = "mozilla-foundation/common_voice_13_0"
    dataset_subset: str | None = "en"
    dataset_split: str = "test"
    audio_column: str = "audio"
    text_column: str = "sentence"
    max_samples: int | None = None
    streaming: bool = True
    language: str = "en"
    output_dir: str = "results"
    batch_size: int = 1

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            model_type=self.model_type,
            model_id=self.model_id,
            language=self.language,
            model_kwargs=self.model_kwargs,
        )

    def to_dataset_config(self) -> DatasetConfig:
        return DatasetConfig(
            dataset_name=self.dataset_name,
            splits=[self.dataset_split],
            audio_column=self.audio_column,
            text_column=self.text_column,
            subset=self.dataset_subset,
            max_samples=self.max_samples,
            streaming=self.streaming,
        )


class Evaluator:
    """
    Model-oriented evaluator: load the model ONCE, run every requested split.

    Usage:
        ev = Evaluator(model_cfg, dataset_cfg, output_dir="results/", batch_size=1)
        summaries = ev.run()   # model loaded once, all splits evaluated
    """

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        output_dir: str = "results",
        batch_size: int = 1,
        # Back-compat: accept legacy EvalConfig as first arg
        config: EvalConfig | None = None,
    ) -> None:
        if config is not None:
            # Legacy single-arg construction
            model_config = config.to_model_config()
            dataset_config = config.to_dataset_config()
            output_dir = config.output_dir
            batch_size = config.batch_size

        self.model_cfg = model_config
        self.dataset_cfg = dataset_config
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size

        lang = model_config.language or "en"
        if lang == "en":
            self._normalizer = EnglishTextNormalizer()
        elif lang in ("ms", "id"):
            self._normalizer = MalayTextNormalizer()
        else:
            self._normalizer = BasicTextNormalizer()
        self._model: BaseASRModel | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        """Load model once, evaluate all splits, return list of summaries."""
        self._load_model()
        summaries = []
        for split in self.dataset_cfg.splits:
            logger.info(
                "=== %s | %s | split=%s ===",
                self.model_cfg.model_id,
                self.dataset_cfg.dataset_name,
                split,
            )
            summary = self._run_split(split)
            summaries.append(summary)
            logger.info(
                "WER=%.2f%%  CER=%.2f%%  S=%.2f%%  I=%.2f%%  D=%.2f%%  RTFx=%.2f  "
                "[%d rows dataset ref, %d rows ours]",
                summary["final_wer"], summary["final_cer"],
                summary["final_substitution_rate"], summary["final_insertion_rate"],
                summary["final_deletion_rate"], summary["rtfx"],
                summary["rows_using_dataset_ref"], summary["rows_using_ours_ref"],
            )
        return summaries

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        mcfg = self.model_cfg
        model_cls = MODEL_REGISTRY[mcfg.model_type]
        logger.info("Loading model: %s (%s)", mcfg.model_id, mcfg.model_type)
        self._model = model_cls(
            model_id=mcfg.model_id,
            language=mcfg.language,
            **mcfg.model_kwargs,
        )
        logger.info("Model loaded.")

    def _manifest_path(self, split: str) -> Path:
        mcfg = self.model_cfg
        dcfg = self.dataset_cfg
        safe_model = mcfg.model_id.replace("/", "__")
        safe_dataset = dcfg.dataset_name.replace("/", "__")
        subset_tag = f"_{dcfg.subset}" if dcfg.subset else ""
        fname = f"{safe_model}__{safe_dataset}{subset_tag}__{split}.jsonl"
        return self.output_dir / fname

    def _run_split(self, split: str) -> dict[str, Any]:
        dcfg = self.dataset_cfg
        mcfg = self.model_cfg

        # Pull the dataset's normalized_text column via metadata if configured
        meta_cols = list(dcfg.metadata_columns)
        if dcfg.normalized_text_column and dcfg.normalized_text_column not in meta_cols:
            meta_cols = meta_cols + [dcfg.normalized_text_column]
        if dcfg.category_filter and "category" not in meta_cols:
            meta_cols = meta_cols + ["category"]

        data_iter = load_dataset_hf(
            dataset_name=dcfg.dataset_name,
            split=split,
            subset=dcfg.subset,
            audio_column=dcfg.audio_column,
            text_column=dcfg.text_column,
            metadata_columns=meta_cols,
            max_samples=dcfg.max_samples,
            streaming=dcfg.streaming,
        )

        records: list[dict[str, Any]] = []
        audio_batch: list = []
        sr_batch: list[int] = []
        dur_batch: list[float] = []
        ref_batch: list[str] = []
        meta_batch: list[dict] = []

        def _flush() -> None:
            results = self._model.transcribe_batch(audio_batch, sr_batch, dur_batch)
            for result, ref, meta in zip(results, ref_batch, meta_batch):
                if result.skipped:
                    continue
                pred_norm = self._normalizer(result.prediction)

                # --- Normalize both references ---
                ref_ours = self._normalizer(ref)
                ref_dataset = (
                    self._normalizer(meta[dcfg.normalized_text_column])
                    if dcfg.normalized_text_column and dcfg.normalized_text_column in meta
                    else None
                )

                # --- Per-row alignment for both refs ---
                stats_ours = _align(ref_ours.split(), pred_norm.split())

                if ref_dataset is not None:
                    stats_dataset = _align(ref_dataset.split(), pred_norm.split())
                    # WER proxy per row: errors / ref_length (lower = better for model)
                    wer_ours = stats_ours.errors / max(stats_ours.ref_length, 1)
                    wer_dataset = stats_dataset.errors / max(stats_dataset.ref_length, 1)
                    if wer_dataset <= wer_ours:
                        stats_final = stats_dataset
                        ref_final = ref_dataset
                        ref_source = "dataset"
                    else:
                        stats_final = stats_ours
                        ref_final = ref_ours
                        ref_source = "ours"
                else:
                    stats_final = stats_ours
                    ref_final = ref_ours
                    ref_source = "ours"

                record: dict[str, Any] = {
                    "reference": ref,
                    "reference_normalized_ours": ref_ours,
                    "prediction": result.prediction,
                    "prediction_normalized": pred_norm,
                    "audio_length_s": result.audio_length_s,
                    "transcription_time_s": result.transcription_time_s,
                    "rtfx": result.rtfx,
                    # Final (per-row best) reference and its alignment stats
                    "reference_normalized_final": ref_final,
                    "ref_source": ref_source,
                    "num_ref_words": stats_final.ref_length,
                    "word_hits": stats_final.hits,
                    "word_substitutions": stats_final.substitutions,
                    "word_insertions": stats_final.insertions,
                    "word_deletions": stats_final.deletions,
                    # Individual scores for traceability
                    "wer_ours": round(stats_ours.errors / max(stats_ours.ref_length, 1) * 100, 2),
                    "wer_dataset": round(stats_dataset.errors / max(stats_dataset.ref_length, 1) * 100, 2) if ref_dataset is not None else None,
                    "model_id": mcfg.model_id,
                    "dataset": dcfg.dataset_name,
                    "subset": dcfg.subset,
                    "split": split,
                }
                if ref_dataset is not None:
                    record["reference_normalized_dataset"] = ref_dataset
                    record["reference_normalized_dataset_raw"] = meta[dcfg.normalized_text_column]
                # Store category and other metadata
                for k, v in meta.items():
                    if k != dcfg.normalized_text_column:
                        record[k] = v
                records.append(record)
            audio_batch.clear(); sr_batch.clear(); dur_batch.clear()
            ref_batch.clear(); meta_batch.clear()

        for audio, sr, dur, ref, meta in data_iter:
            if dcfg.category_filter and meta.get("category") != dcfg.category_filter:
                continue
            audio_batch.append(audio)
            sr_batch.append(sr)
            dur_batch.append(dur)
            ref_batch.append(ref)
            meta_batch.append(meta)
            if len(audio_batch) >= self.batch_size:
                _flush()
                logger.info("  [%s] processed %d samples", split, len(records))

        if audio_batch:
            _flush()

        # --- Corpus-level metrics (aggregated from per-row final stats) ---
        # WER/CER: computed from the per-row winning reference
        final_refs = [r["reference_normalized_final"] for r in records]
        preds      = [r["prediction_normalized"] for r in records]
        metrics    = compute_all_metrics(final_refs, preds)

        total_audio = sum(r["audio_length_s"] for r in records)
        total_time  = sum(r["transcription_time_s"] for r in records)
        rtfx        = round(total_audio / total_time, 2) if total_time > 0 else float("inf")

        n_dataset = sum(1 for r in records if r["ref_source"] == "dataset")
        n_ours    = len(records) - n_dataset

        manifest_path = self._manifest_path(split)
        write_manifest(manifest_path, records)
        logger.info("Manifest → %s", manifest_path)
        logger.info(
            "WER=%.2f%%  CER=%.2f%%  S=%.2f%%  I=%.2f%%  D=%.2f%%  RTFx=%.2f  "
            "[rows: %d via dataset ref, %d via ours]",
            metrics["wer"], metrics["cer"],
            metrics["substitution_rate"], metrics["insertion_rate"], metrics["deletion_rate"],
            rtfx, n_dataset, n_ours,
        )

        return {
            "model_id": mcfg.model_id,
            "dataset": dcfg.dataset_name,
            "subset": dcfg.subset,
            "split": split,
            "num_samples": len(records),
            "final_wer": metrics["wer"],
            "final_cer": metrics["cer"],
            "final_substitution_rate": metrics["substitution_rate"],
            "final_insertion_rate": metrics["insertion_rate"],
            "final_deletion_rate": metrics["deletion_rate"],
            "word_substitutions": metrics["word_substitutions"],
            "word_insertions": metrics["word_insertions"],
            "word_deletions": metrics["word_deletions"],
            "word_hits": metrics["word_hits"],
            "num_ref_words": metrics["num_ref_words"],
            "char_substitutions": metrics["char_substitutions"],
            "char_insertions": metrics["char_insertions"],
            "char_deletions": metrics["char_deletions"],
            "rtfx": rtfx,
            "total_audio_s": round(total_audio, 2),
            "total_transcription_s": round(total_time, 2),
            "rows_using_dataset_ref": n_dataset,
            "rows_using_ours_ref": n_ours,
        }
