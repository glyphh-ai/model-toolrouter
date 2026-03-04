"""
Deploy models from directory to PostgreSQL on runtime startup.

Discovers model directories in custom_models/,
reads their JSONL data, encodes with the model's encoder, and
stores as glyphs in the runtime database.

Called from main.py lifespan. No .glyphh files needed.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

logger = logging.getLogger(__name__)

RUNTIME_ROOT = Path(__file__).parent.parent
CUSTOM_MODELS_DIR = RUNTIME_ROOT / "custom_models"


def _load_jsonl(data_dir: Path) -> list[dict]:
    """Load all JSONL files from a data directory."""
    entries = []
    if not data_dir.exists():
        return entries
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


async def deploy_model_to_db(
    model_dir: Path,
    org_id: str,
    model_manager: Any,
    session_factory: Any,
) -> int:
    """Deploy a single model's data to the database.

    1. Load encoder config from encoder.py via loader.load_model()
    2. Check DB for existing deployment (same version + glyph count -> skip)
    3. Read JSONL data from data/
    4. Convert entries via entry_to_record_fn
    5. Encode each record with model's Encoder
    6. Store as glyphs in PostgreSQL scoped to (org_id, model_id)
    7. Serialize encoder_config + version to ModelConfig row

    Returns number of glyphs created.
    """
    from domains.models.loader import load_model as load_model_def
    from domains.models.storage import GlyphStorage
    from domains.models.db_models import ModelConfig
    from glyphh.encoder import Encoder
    from glyphh.core.types import Concept

    loaded = load_model_def(model_dir)
    model_id = loaded.model_id

    if not loaded.has_custom_encoder or loaded.encoder_config is None:
        logger.debug(f"Skipping {model_id}: no custom encoder")
        return 0

    data_dir = model_dir / "data"
    entries = _load_jsonl(data_dir)
    if not entries:
        logger.debug(f"Skipping {model_id}: no JSONL data")
        return 0

    manifest_version = loaded.manifest.version or ""

    # Check if already deployed with same version and count
    async with session_factory() as session:
        storage = GlyphStorage(session)
        existing_count = await storage.count_glyphs(org_id, model_id)

        result = await session.execute(
            select(ModelConfig).where(
                ModelConfig.org_id == org_id,
                ModelConfig.model_id == model_id,
            )
        )
        db_config = result.scalar_one_or_none()
        db_version = db_config.model_version if db_config else None

    # Skip if same version AND same glyph count
    if db_version == manifest_version and existing_count >= len(entries):
        logger.info(
            f"Model {org_id}/{model_id} v{manifest_version} already deployed "
            f"({existing_count} glyphs), skipping"
        )
        return existing_count

    # Create encoder from model's config
    encoder = Encoder(loaded.encoder_config)

    # Convert and encode
    entry_to_record = loaded.entry_to_record_fn
    created = 0

    async with session_factory() as session:
        storage = GlyphStorage(session)

        # Clear existing data if re-deploying
        if existing_count > 0:
            logger.info(
                f"Re-deploying {org_id}/{model_id}: "
                f"clearing {existing_count} old glyphs "
                f"(version {db_version} -> {manifest_version})"
            )
            await storage.delete_model_data(org_id, model_id)

        for entry in entries:
            try:
                if entry_to_record:
                    record = entry_to_record(entry)
                    concept_text = record["concept_text"]
                    metadata = record["metadata"]
                    attrs = record["attributes"]
                else:
                    concept_text = entry.get("question", entry.get("text", ""))
                    metadata = {k: v for k, v in entry.items() if k != "question"}
                    attrs = {"text": concept_text}

                concept = Concept(
                    name=f"entry_{created}",
                    attributes=attrs,
                    metadata=metadata,
                )
                glyph = encoder.encode(concept)
                embedding = glyph.global_cortex.data.astype(float).tolist()

                await storage.create_glyph(
                    org_id=org_id,
                    model_id=model_id,
                    concept_text=concept_text,
                    embedding=embedding,
                    metadata={**metadata, "record_type": "pattern"},
                )
                created += 1

            except Exception as e:
                logger.warning(f"Failed to encode entry in {model_id}: {e}")
                continue

        await session.commit()

    # Serialize encoder config to ModelConfig DB row
    encoder_config_dict = None
    if hasattr(loaded.encoder_config, "to_dict"):
        encoder_config_dict = loaded.encoder_config.to_dict()

    async with session_factory() as session:
        result = await session.execute(
            select(ModelConfig).where(
                ModelConfig.org_id == org_id,
                ModelConfig.model_id == model_id,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            existing.model_version = manifest_version
            existing.meta_name = loaded.manifest.name
            existing.model_path = str(model_dir)
            existing.encoder_config = encoder_config_dict
            existing.updated_at = datetime.utcnow()
        else:
            config = ModelConfig(
                org_id=org_id,
                model_id=model_id,
                model_path=str(model_dir),
                model_version=manifest_version,
                meta_name=loaded.manifest.name,
                short_description=loaded.manifest.description,
                long_description="",
                encoder_config=encoder_config_dict,
            )
            session.add(config)
        await session.commit()

    logger.info(
        f"Deployed {org_id}/{model_id} v{manifest_version}: "
        f"{created} glyphs from {len(entries)} entries"
    )
    return created


async def deploy_all_models(
    model_manager: Any,
    session_factory: Any,
) -> dict[str, int]:
    """Deploy eligible models to the database on startup.

    Scans custom_models/ (org_id="custom") and deploys all found models.
    Also checks GLYPHH_DEV_MODEL_DIR env var for a single inline model path
    (used by `glyphh dev` to deploy without touching custom_models/).
    Returns dict of model_id -> glyph count.
    """
    import os
    from domains.models.loader import load_model as load_model_def

    results = {}

    for base_dir, default_org, always_deploy in [
        (CUSTOM_MODELS_DIR, "custom", True),
    ]:
        if not base_dir.exists():
            continue
        for model_dir in sorted(base_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            try:
                loaded = load_model_def(model_dir)

                # For core models, only deploy if load_on_startup is True
                if not always_deploy and not loaded.manifest.load_on_startup:
                    logger.debug(
                        f"Skipping {loaded.model_id}: load_on_startup is false"
                    )
                    continue

                count = await deploy_model_to_db(
                    model_dir=model_dir,
                    org_id=default_org,
                    model_manager=model_manager,
                    session_factory=session_factory,
                )
                if count > 0:
                    results[model_dir.name] = count
            except Exception as e:
                logger.error(f"Failed to deploy model {model_dir.name}: {e}")

    # Inline model path from `glyphh dev` — deploy without touching custom_models/
    dev_model_dir = os.environ.get("GLYPHH_DEV_MODEL_DIR", "").strip()
    if dev_model_dir:
        from pathlib import Path as _Path
        dev_path = _Path(dev_model_dir)
        if dev_path.is_dir():
            try:
                count = await deploy_model_to_db(
                    model_dir=dev_path,
                    org_id="local-dev-org",
                    model_manager=model_manager,
                    session_factory=session_factory,
                )
                if count > 0:
                    results[dev_path.name] = count
                    logger.info(f"Deployed dev model {dev_path.name}: {count} glyphs")
            except Exception as e:
                logger.error(f"Failed to deploy dev model {dev_path.name}: {e}")
        else:
            logger.warning(f"GLYPHH_DEV_MODEL_DIR not found: {dev_model_dir}")

    return results


async def register_model_encoders(
    model_manager: Any,
    session_factory: Any,
) -> None:
    """Register model encoders in the model manager for query/listener use.

    For each model directory with a custom encoder, creates an Encoder
    and registers a LoadedModel in Model_Manager._models with:
    - encoder
    - similarity_calculator
    - encode_query_fn
    """
    from domains.models.loader import discover_models
    from glyphh.encoder import Encoder
    from domains.models.manager import LoadedModel as ManagerLoadedModel
    from shared.sdk_adapter import get_sdk_adapter

    adapter = get_sdk_adapter()

    # Collect all models to register: custom_models/ + optional inline dev model
    all_models = list(discover_models(CUSTOM_MODELS_DIR, None))

    dev_model_dir = os.environ.get("GLYPHH_DEV_MODEL_DIR", "").strip()
    if dev_model_dir:
        from pathlib import Path as _Path
        dev_path = _Path(dev_model_dir)
        if dev_path.is_dir():
            try:
                dev_loaded = load_model_def(dev_path, source="dev")
                # Avoid duplicates if this model is also in custom_models/
                if not any(m.model_id == dev_loaded.model_id for m in all_models):
                    all_models.append(dev_loaded)
            except Exception as e:
                logger.warning(f"Could not load dev model for encoder registration: {e}")

    for loaded in all_models:
        if not loaded.has_custom_encoder or loaded.encoder_config is None:
            continue

        if loaded.source == "core":
            org_id = "glyphh"
        elif loaded.source == "dev":
            org_id = "local-dev-org"
        else:
            org_id = "custom"
        model_id = loaded.model_id

        # Skip if already registered
        existing = await model_manager.get_model(org_id, model_id)
        if existing is not None:
            continue

        try:
            encoder = Encoder(loaded.encoder_config)
            similarity_calculator = adapter.create_similarity_calculator()

            # Minimal model proxy — no GlyphhModel needed
            class DirectoryModel:
                def __init__(self, name, version, config):
                    self.name = name
                    self.version = version
                    self.encoder_config = config

            sdk_model_proxy = DirectoryModel(
                loaded.manifest.name,
                loaded.manifest.version or "0.1.0",
                loaded.encoder_config,
            )

            manager_model = ManagerLoadedModel(
                org_id=org_id,
                model_id=model_id,
                model_path=str(loaded.model_dir),
                sdk_model=sdk_model_proxy,
                encoder=encoder,
                similarity_calculator=similarity_calculator,
                loaded_at=datetime.utcnow(),
                meta_name=loaded.manifest.name,
                short_description=loaded.manifest.description,
                long_description="",
                encode_query_fn=loaded.encode_query_fn,
            )

            model_manager._models[(org_id, model_id)] = manager_model
            logger.info(f"Registered encoder for {org_id}/{model_id}")

        except Exception as e:
            logger.error(f"Failed to register encoder for {model_id}: {e}")
