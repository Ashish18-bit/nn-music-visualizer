"""
features/cache.py
─────────────────
Persistent cache for extracted FeatureVectors.

Saves each batch as a .npz (numpy compressed archive) so features
don't need to be recomputed across runs.  The cache key is derived
from the audio file path + extractor config hash.

Usage
-----
cache = FeatureCache(save_dir="features/cache")

# Save
cache.save(feature_vectors, key="song_abc")

# Load
vectors = cache.load("song_abc")   # returns List[FeatureVector] or None
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from features.extractor import FeatureVector

logger = logging.getLogger(__name__)


class FeatureCache:
    """
    Disk-backed cache for feature vectors using numpy .npz format.

    Each cache entry stores:
      - vectors matrix  : shape (N, D) float32
      - feature_map     : JSON string
      - metadata arrays : segment indices, start/end times, source labels
    """

    def __init__(self, save_dir: str | Path = "features/cache") -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ── public ──────────────────────────────────────────────

    def save(
        self, vectors: List[FeatureVector], key: str
    ) -> Path:
        """
        Persist a list of FeatureVectors to disk.

        Parameters
        ----------
        vectors : list of FeatureVector objects
        key     : cache key (used as filename stem, sanitised)

        Returns
        -------
        Path to the saved .npz file
        """
        if not vectors:
            raise ValueError("Cannot save an empty feature list")

        path = self._key_to_path(key)

        matrix = np.stack([fv.vector for fv in vectors])          # (N, D)
        indices = np.array([fv.segment_idx for fv in vectors])    # (N,)
        starts = np.array([fv.start_sec for fv in vectors])       # (N,)
        ends = np.array([fv.end_sec for fv in vectors])           # (N,)
        sources = np.array([fv.source for fv in vectors])         # (N,) object

        # feature_map is the same for all vectors in a batch
        fmap_json = json.dumps(vectors[0].feature_map)

        np.savez_compressed(
            path,
            matrix=matrix,
            indices=indices,
            starts=starts,
            ends=ends,
            sources=sources,
            feature_map_json=np.array(fmap_json),
        )

        logger.info("Cached %d vectors → %s", len(vectors), path)
        return path

    def load(self, key: str) -> Optional[List[FeatureVector]]:
        """
        Load a previously cached feature list.

        Returns None if the key does not exist in the cache.
        """
        path = self._key_to_path(key)
        if not path.exists():
            return None

        data = np.load(path, allow_pickle=True)
        matrix = data["matrix"]             # (N, D)
        indices = data["indices"]
        starts = data["starts"]
        ends = data["ends"]
        sources = data["sources"]
        fmap_json = str(data["feature_map_json"])

        # Rebuild feature_map — stored as {name: [start, end]}
        raw_fmap = json.loads(fmap_json)
        feature_map: Dict[str, Tuple[int, int]] = {
            k: tuple(v) for k, v in raw_fmap.items()
        }

        vectors = [
            FeatureVector(
                vector=matrix[i].astype(np.float32),
                feature_map=feature_map,
                segment_idx=int(indices[i]),
                start_sec=float(starts[i]),
                end_sec=float(ends[i]),
                source=str(sources[i]),
            )
            for i in range(len(matrix))
        ]

        logger.info("Loaded %d vectors from cache (%s)", len(vectors), path.name)
        return vectors

    def exists(self, key: str) -> bool:
        """Return True if the cache entry exists."""
        return self._key_to_path(key).exists()

    def delete(self, key: str) -> bool:
        """Delete a cache entry. Returns True if it existed."""
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()
            logger.info("Deleted cache entry: %s", path.name)
            return True
        return False

    def clear_all(self) -> int:
        """Delete every .npz file in the cache directory. Returns count deleted."""
        deleted = 0
        for p in self.save_dir.glob("*.npz"):
            p.unlink()
            deleted += 1
        logger.info("Cleared %d cache entries", deleted)
        return deleted

    def list_keys(self) -> List[str]:
        """Return all available cache keys (file stems)."""
        return [p.stem for p in sorted(self.save_dir.glob("*.npz"))]

    # ── static helpers ───────────────────────────────────────

    @staticmethod
    def make_key(audio_path: str | Path, extractor_config: Optional[dict] = None) -> str:
        """
        Generate a deterministic cache key from a file path + config.

        Using a hash means the cache is automatically invalidated when
        the extractor configuration changes.
        """
        stem = Path(audio_path).stem
        if extractor_config:
            config_hash = hashlib.md5(
                json.dumps(extractor_config, sort_keys=True).encode()
            ).hexdigest()[:8]
            return f"{stem}_{config_hash}"
        return stem

    # ── private ─────────────────────────────────────────────

    def _key_to_path(self, key: str) -> Path:
        """Sanitise key and return the full .npz path."""
        safe_key = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self.save_dir / f"{safe_key}.npz"
