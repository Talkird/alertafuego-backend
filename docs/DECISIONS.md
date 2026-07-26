# Decision Log

### [2026-07-25] — Use a fixed EPSG:4326 grid, not GOES-19's native projection, for patch/mask geometry

**Context:** `model/data_pipeline/patches.py` needs a square region of exactly `patch_size_px` pixels around a lat/lon, used identically for both the GOES-19 patch and the VIIRS-derived label mask. Using the GOES image's own native projection (its geostationary fixed grid) in `ee.Geometry.buffer()`/`.reproject()` produced a region of ~4 billion pixels instead of ~1000 — the native grid's coordinate units are not plain meters the way `buffer()` assumes.
**Decision:** Define a single fixed projection, `ee.Projection("EPSG:4326").atScale(GOES_SCALE_METERS)`, and reproject the GOES image onto it before sampling; use the same projection to reproject the label mask. `patches.target_projection()` is the shared source of truth.
**Alternatives considered:**
- Use `goes_image.select(band).projection()` (the image's own native grid) directly.
- Query the native projection's declared linear unit from Earth Engine metadata and convert distances manually.
**Rationale:** A plain, known CRS+scale applied identically to both patch and mask guarantees pixel-grid alignment without needing to reverse-engineer GOES's native grid semantics. Simpler and more robust for phase 1 than special-casing the satellite's fixed grid.
**Consequences:** `ee.Geometry.buffer()` called with a projection that has an embedded scale (via `atScale`) interprets the distance argument in that grid's *pixel* units, not meters — `compute_patch_region` passes `patch_size_px / 2`, not `patch_size_px / 2 * GOES_SCALE_METERS`. Also: extracted patches come out at `patch_size_px + 1` per side (e.g. 33x33 for a 32x32 request) due to Earth Engine's own bbox rounding — this is consistent and off-by-one, not random. Phase 2's PyTorch `Dataset`/loader will need to crop or pad samples to a uniform shape before batching, since patch dimensions are not guaranteed identical across samples.

### [2026-07-25] — Apply per-band scale/offset calibration to GOES-19 CMI bands

**Context:** GOES-19 MCMIPF's `CMI_C01`..`CMI_C16` bands are documented as calibrated reflectance/brightness-temperature, but Earth Engine actually stores them as raw uint16 digital numbers (`PixelType: int, 0-65535`). Sampled patch values (e.g. ~7895-9404 for band 7) were nonsensical as physical units until this was caught during smoke testing.
**Decision:** Fetch each band's `{band}_scale`/`{band}_offset` image properties and apply `physical = raw * scale + offset` in `patches.extract_patch()` before returning the patch array.
**Alternatives considered:** Ship raw DN values and defer calibration to the PyTorch dataset loader in phase 2.
**Rationale:** Calibrating at export time means the exported `.npz` dataset is self-contained and physically meaningful (verified: band 7 values land at ~300-320K over a real fire-adjacent detection, consistent with expected brightness temperature); deferring it would let uncalibrated data leak into training if the loader step were ever skipped or reused elsewhere.
**Consequences:** The masked-pixel sentinel had to move from the raw domain to the physical domain: `RAW_FILL_VALUE=0` is used as `sampleRectangle`'s default (valid for the raw uint16 range), then masked positions are overwritten with `PATCH_FILL_VALUE=-1.0` after calibration (safe since calibrated reflectance/brightness-temperature is never negative).

### [2026-07-25] — Cap VIIRS FeatureCollection queries at 4000 features per call

**Context:** `matching.fetch_viirs_detections()` merges per-image fire-pixel `FeatureCollection`s across a date range and calls `.getInfo()` once. Earth Engine hard-caps `FeatureCollection.getInfo()` at 5000 elements per query; a one-week window over all of Argentina in dry season (September) exceeded that on both VIIRS collections.
**Decision:** Apply `.limit(MAX_VIIRS_FEATURES_PER_QUERY)` with `MAX_VIIRS_FEATURES_PER_QUERY = 4000` per collection per query, logging a warning when the cap is hit.
**Alternatives considered:**
- Paginate/chunk the query by day instead of capping.
- Fail loudly instead of silently truncating.
**Rationale:** Capping keeps the pipeline usable without a pagination rewrite for phase 1; a warning makes truncation visible instead of silent.
**Consequences:** Large bbox/date-range pulls in high fire-activity periods will silently drop detections beyond the cap. A real full-scale export needs to chunk by smaller date ranges (e.g. per day) per query to get a complete pull, not one call over a full week+countrywide bbox.

### [2026-07-25] — Build the wildfire model as PyTorch pixel-segmentation trained on a statically-exported GOES-19/VIIRS dataset

**Context:** Before writing any code for the AI model (this is a from-scratch `model/` directory), three foundational choices had to be made: which deep learning framework, how to frame the detection problem, and how training data gets from Earth Engine into the model.
**Decision:** Use PyTorch; frame detection as per-pixel segmentation (predict a fire-probability mask over a GOES-19 patch, not a single fire/no-fire label per patch); build the dataset by exporting it once to local disk via a CLI script (`model/scripts/export_dataset.py`), rather than querying Earth Engine live during training.
**Alternatives considered:**
- TensorFlow/Keras (has more native Earth Engine export tooling, e.g. TFRecord export).
- Patch-level binary classification instead of per-pixel segmentation.
- On-the-fly data loading from Earth Engine during each training epoch.
**Rationale:** PyTorch chosen for flexibility/ecosystem fit. Segmentation chosen because the end product is a fire map, not a single label per region, and is more spatially precise. Static export chosen so training doesn't depend on Earth Engine's quota/latency on every epoch and so the dataset is reproducible/inspectable offline.
**Consequences:** This phase (phase 1) covers only building the exported dataset (`model/data_pipeline/`, `model/scripts/export_dataset.py`) — verified working end-to-end against live Earth Engine. Model architecture (e.g. U-Net) and the training loop are phase 2, not yet started. Changing framework or framing later requires re-deriving the export format too, since patch/mask shapes are chosen to fit this framing.

### [2026-07-25] — Export one `.npz` file per sample instead of a single HDF5/TFRecord dataset

**Context:** `model/data_pipeline/dataset_builder.py` needs to persist thousands of GOES-19 patch + VIIRS mask pairs to local disk in a format a PyTorch `Dataset` can load.
**Decision:** Save each sample as its own `.npz` file (`patch`, `mask` arrays) under `model/dataset/{train,val,test}/`, plus a single `manifest.csv` indexing all of them (`filename, lat, lon, goes_time, has_fire, split`).
**Alternatives considered:**
- Single HDF5 file containing all samples.
- GeoTIFF per band per sample.
- Sharded TFRecord/WebDataset files.
**Rationale:** Per-sample files survive a crash mid-export without corrupting already-written data (export makes many sequential Earth Engine calls and can fail partway through, as happened repeatedly during smoke testing); no new dependency beyond `numpy`; maps directly onto `torch.utils.data.Dataset.__getitem__` via `np.load` per item.
**Consequences:** At much larger dataset sizes (many tens of thousands of samples) this will have worse I/O throughput than a sharded format — an explicitly deferred future optimization, not a phase-1 concern.

### [2026-07-25] — Split train/val/test by contiguous date ranges, not random shuffling

**Context:** Fire events cluster in time (a single fire can be detected across several consecutive days/overpasses), so a random shuffle risks putting near-duplicate detections of the same event into both train and a validation/test split, inflating apparent model performance.
**Decision:** `dataset_builder.assign_temporal_splits()` assigns each sample to train/val/test purely by comparing its GOES capture date against two CLI-supplied cutoff dates (`--train-end-date`, `--val-end-date`) — earliest dates go to train, latest to test.
**Alternatives considered:** Random shuffle split (e.g. `sklearn.train_test_split`).
**Rationale:** Prevents temporal leakage between splits; standard practice for any dataset with time-correlated events.
**Consequences:** Split cutoffs must be chosen deliberately per export run relative to the actual date range being pulled — a short pull can end up with an empty val/test split otherwise (observed during smoke testing with a 1-week pull).

### [2026-07-25] — Restrict negative (no-fire) samples to Argentine land, away from known fire locations

**Context:** `model/data_pipeline/negative_sampling.py` needs to generate "no fire" training examples. Naively sampling random points/times across the full bbox risks trivial negatives (ocean/no-data pixels) and false negatives (points near a real, possibly-undetected fire).
**Decision:** Restrict negative locations to Argentina's land boundary (FAO GAUL admin-0 polygon intersected with the bbox) and reject any candidate within `min_negative_distance_km` (default 50km) of a known positive detection; negative timestamps are drawn from real GOES-19 capture times in the same date range as the positives.
**Alternatives considered:** Uniform random sampling over the full bbox with no land/distance filtering.
**Rationale:** Land-only sampling avoids the model learning "ocean == no fire" as a shortcut; the distance rejection avoids mislabeling areas near real fire activity as negative.
**Consequences:** Negative sampling does client-side rejection sampling in a bounded loop (`MAX_SAMPLING_ATTEMPTS`) — if `min_negative_distance_km` is set very high relative to available land area, it can return fewer negatives than requested (logged as a warning, not an error). Verified working in the smoke test: negatives landed at real Argentine land coordinates (Neuquén, Mendoza, Catamarca, La Pampa) with all-zero masks.

### [2026-07-25] — No generic data-source abstraction layer in `model/data_pipeline/`

**Context:** `model/data_pipeline/` only ever needs to combine one imagery source (GOES-19) with one label source (VIIRS, across two satellites). It would be possible to build a generic "imagery source"/"label source" interface now, anticipating future satellite products.
**Decision:** Keep each pipeline stage (`matching`, `patches`, `labels`, `negative_sampling`, `dataset_builder`) as a plain module with concrete functions tied to GOES-19/VIIRS specifically — no abstract base classes or plugin registration.
**Alternatives considered:** Define abstract `ImagerySource`/`LabelSource` interfaces so a future satellite product could be swapped in.
**Rationale:** YAGNI — there is currently exactly one imagery source and one label source; a generic interface today would be speculative and untested against any second real use case.
**Consequences:** Adding a second imagery or label source later requires refactoring these modules to extract a real interface once that second source's actual requirements are known, rather than assuming today's guessed interface is right.

### [2026-07-25] — Chunk multi-day exports by day internally in `export_dataset.py`

**Context:** A real export needs to cover weeks or months, but every Earth Engine query in the pipeline (VIIRS detections, GOES capture-time list for negatives) is capped at 5000 elements — a single week in dry season already hit that cap. `write_manifest`/`save_sample` also overwrite on every invocation (`open(..., "w")`, index restarting at 1), so running the CLI multiple times by hand for sub-ranges would silently clobber earlier output rather than accumulate.
**Decision:** `export_dataset.main()` internally splits the requested `[--start-date, --end-date)` into `CHUNK_DAYS=1`-day windows, calls `build_positive_samples`/`build_negative_samples` per day, and accumulates all samples in memory before writing `manifest.csv` once at the end.
**Alternatives considered:**
- Make the user run the CLI once per sub-range manually.
- Make `save_sample`/`write_manifest` append-aware (continue numbering, append CSV rows) so multiple invocations could accumulate safely.
**Rationale:** A single CLI invocation that chunks internally is simpler for the user (one command for an arbitrary date range) and avoids needing cross-run state management (tracking the last used sample index, deduplicating manifest rows on rerun, etc.).
**Consequences:** `--limit` is now a budget spent across chunks (decremented as chunks are processed, loop stops early once exhausted) rather than applied to a single query. A multi-month real export means many sequential per-day Earth Engine round-trips — expect this to take a long time (each active fire-day chunk took ~30-60s in testing); a full dry season (months) should be timed on a short trial range first rather than launched blind.

### [2026-07-25] — Cap and randomly subsample detections per day-chunk (default 200)

**Context:** Running a real 1-month export with no `--limit` stalled for hours on a single high-activity day: one day had thousands of VIIRS detections (near the 4000-per-satellite query cap), and each detection needs several sequential Earth Engine round-trips (match to GOES, extract patch, sample mask) — with no per-day bound, one outbreak day made the whole export impractically slow. The prior truncation (`detections[:limit]`) was also a latent bug: Earth Engine returns detections in scan order, so slicing the first N would have geographically biased the dataset toward whatever region happens to be scanned first.
**Decision:** `PipelineConfig.max_detections_per_chunk` (default 200) bounds how many detections `build_positive_samples()` will process per call; when a window has more, it randomly subsamples (seeded, `random.Random(seed).sample(...)`) down to that cap instead of taking the first N.
**Alternatives considered:**
- No cap, let the export run however long a high-activity day takes.
- Truncate to the first N detections returned (the original, order-biased approach).
**Rationale:** Bounds worst-case per-day runtime regardless of real-world fire activity; random sampling avoids the geographic bias a positional slice would introduce; 200/day keeps a full month's worst case on the order of hours rather than potentially a full day for one outbreak.
**Consequences:** On very high-activity days, most real detections are discarded rather than exported — the dataset undersamples the most extreme fire days relative to their true detection density. If that matters later (e.g. training needs more extreme-event coverage), `max_detections_per_chunk` is the knob to raise, trading dataset size/diversity for export runtime.

### [2026-07-26] — Pin `requirements.txt` to the CUDA build of torch

**Context:** `requirements.txt` listed plain `torch`, which resolves to the CPU-only wheel by default on Windows/PyPI. Training ran on CPU despite the development machine having an RTX 4060 - confirmed via `torch.cuda.is_available() == False` with the CPU wheel installed. The GPU driver (596.49) is recent enough to support CUDA 12.6.
**Decision:** Reinstalled `torch==2.13.0+cu126` from `https://download.pytorch.org/whl/cu126`, and pinned `requirements.txt` accordingly with an `--extra-index-url` line so a fresh `pip install -r requirements.txt` gets the CUDA build automatically instead of silently falling back to CPU.
**Alternatives considered:**
- Keep CPU-only training (works, just much slower - ~29s/epoch on the full ~9488-sample train set observed on CPU).
- Pin a different CUDA version (cu121/cu124 wheels were not available for the installed Python 3.14 interpreter at time of writing; cu126 was the newest available match).
**Rationale:** GPU training is materially faster and this machine has a GPU sitting unused; pinning the exact `+cu126` build (rather than leaving `torch` unpinned) makes the dependency reproducible instead of silently CPU-only on a clean install.
**Consequences:** `requirements.txt` now assumes an NVIDIA GPU + driver capable of CUDA 12.6 - a machine without one (or `pip install -r requirements.txt` on a CPU-only CI box) needs to override this line back to plain `torch` or the CPU wheel explicitly. `model/scripts/train_model.py` still auto-detects the device (`cuda` if available else `cpu`), so the training code itself has no hard GPU dependency - only the pinned wheel does.

### [2026-07-26] — Serve detections on-demand, synchronously, from two fixed-bbox endpoints

**Context:** Phase 3 needed to decide how `backend/` exposes the trained model: whether requests trigger a live Earth Engine fetch + inference right then, or read from a periodically-refreshed cache, and what geographic area(s) a request covers.
**Decision:** Two synchronous `GET` endpoints, `/detect/demo` (small fixed bbox) and `/detect/argentina` (the full country bbox already used in the data pipeline) - each call fetches the latest available GOES-19 image and runs inference in the same HTTP request, no caching between calls.
**Alternatives considered:**
- Background scheduler + cache, endpoints just read the last computed result.
- A single endpoint with a client-supplied bbox instead of two fixed ones.
**Rationale:** Simplest correct implementation for this phase; a scheduler is explicitly deferred until it's clear the on-demand latency (see below) is actually a problem in practice, rather than building it preemptively.
**Consequences:** `/detect/argentina` takes ~50-55s per call (measured, see below) - acceptable for manual/demo use but too slow for something expecting snappy responses; revisit the background/cache option if that becomes a real requirement. Every call re-fetches from Earth Engine - no staleness risk, but also no request-coalescing if multiple clients call at once (each pays the full cost independently).

### [2026-07-26] — Fetch large bboxes in spatial chunks, stitched by absolute geographic position

**Context:** `Image.sampleRectangle()` caps at 262,144px/call (see the 2026-07-25 DQF-calibration-adjacent entry above). The full Argentina bbox is ~1.85M pixels at GOES's 2000m scale - ~7x over that cap - so `/detect/argentina` cannot fetch in one call.
**Decision:** `model/inference/raster_fetch.compute_chunk_grid()` splits the bbox into ~480px chunks sized using degrees-per-pixel read directly from Earth Engine's own projection transform (`patches.degrees_per_pixel()`) rather than computed/assumed manually. `model/inference/tiling.assemble_raster()` then pastes each fetched chunk into one canvas by its own absolute geographic bbox (computing each chunk's pixel offset from the overall bbox's north/west edges), rather than concatenating chunks by grid row/col order.
**Alternatives considered:**
- Compute chunk degree-spacing from geodesic meters-per-degree formulas (`111,320 * cos(lat)`), correcting for latitude.
- Stitch by concatenating the chunk grid in row/col order (`np.concatenate` per row, then rows together).
**Rationale:** Every time this codebase reasoned about Earth Engine's projection/geometry units from first principles it got the first attempt wrong (native GOES projection units, `buffer()` distance units - both documented above) - so this fetch asks Earth Engine for its actual transform instead of re-deriving degree math, and places each chunk from its own known bbox instead of trusting a row/col concatenation order not to have an off-by-one or a north/south flip.
**Consequences:** Verified empirically end-to-end: `/detect/argentina` fetched 12 chunks covering the full bbox and found 17 detections clustered sensibly in known agricultural-fire regions (Chaco/Formosa), with no visible chunk-boundary artifacts, in ~50s total.

### [2026-07-26] — Emit one detection per pixel above threshold, no clustering

**Context:** `service.run_detection()` needed to decide the output granularity: one entry per fire-positive pixel, or grouped/clustered per contiguous fire front.
**Decision:** Per-pixel: each `(lat, lon, probability)` above `seg_threshold` (reused from `model.training.config`, not re-decided here) is its own entry in the response.
**Alternatives considered:** Cluster adjacent positive pixels into one detection per contiguous fire front.
**Rationale:** Simplest option that adds no new logic beyond thresholding; matches the per-pixel segmentation framing already established in phase 2. Clustering is a real product improvement but wasn't asked for and would need a real decision about grouping distance/algorithm.
**Consequences:** A single large fire front spanning many adjacent 2km pixels will appear as many separate detections in the response, not one. `/detect/argentina` on a high-activity day could return a long flat list. Revisit if consumers of this API need one alert per fire event rather than per pixel.

### [2026-07-26] — Persist detections to Postgres/PostGIS (Supabase), via SQLAlchemy + GeoAlchemy2 + Alembic

**Context:** Detections computed by `/detect/demo`/`/detect/argentina` were previously stateless - returned once, never stored. The user wants to build a Nuxt frontend later to visualize fires on a map, which needs a persisted history, not just the current live call. Postgres+PostGIS on Supabase was already chosen over MongoDB earlier (structured, geospatial data) with the frontend calling this FastAPI backend directly (no Supabase client anywhere).
**Decision:** Added `sqlalchemy`, `psycopg[binary]`, `geoalchemy2`, `alembic` as dependencies. `backend/app/db.py`/`models.py`/`crud.py` own the DB (engine, ORM model, insert logic), mirroring how `model/` owns Earth Engine/ML concerns. The `detections` table stores `location` as a real PostGIS `geometry(Point,4326)` (via GeoAlchemy2, GiST-indexed) rather than plain `lat`/`lon` floats, plus `probability`, `image_time`, `detected_at` (DB-side `now()`), the source `bbox_*`, and `threshold`. The two detect endpoints changed from `GET` to `POST` (confirmed with the user) since they now have a real side effect (a DB write), and persist via `save_detections()` as their last step before returning the same response shape as before.
**Alternatives considered:**
- Plain `lat`/`lon` float columns instead of a PostGIS geometry column.
- Keep the endpoints as `GET` despite the new side effect.
- A separate background job persists instead of the on-demand endpoints themselves.
**Rationale:** A real geometry column unlocks `ST_Within`/`ST_DWithin`/bbox-intersection queries with GiST indexing directly - the exact reason Postgres+PostGIS was picked over Mongo - without a later migration+backfill once the frontend's map queries materialize. `POST` is correct REST semantics for an action with a side effect, and costs nothing to fix now since no frontend depends on the old contract yet.
**Consequences:** Confirmed end-to-end against a real Supabase instance: `alembic upgrade head` created the table + GiST index + bootstrapped the `postgis` extension in one step; a live `/detect/argentina` call's 1 detection landed in Postgres with matching `probability`/`image_time`/`bbox`/`threshold` and correct `POINT(lon lat)` geometry (verified via `ST_AsText`). Two real Alembic+GeoAlchemy2 autogenerate bugs were hit and fixed by hand (documented so they don't need rediscovering): (1) autogenerate references `geoalchemy2.types.Geometry` in the migration but doesn't add the `import geoalchemy2` line; (2) autogenerate emits a redundant `op.create_index(...)` for the spatial index that duplicates the one GeoAlchemy2's own DDL event already creates from `spatial_index=True`, causing a `DuplicateTable` error - the explicit `create_index`/`drop_index` calls were removed from the migration. No read/list endpoint was added this phase (explicit user scope: "simplemente la persistencia por ahora") - verification was via direct SQL, not a new API. No deduplication of repeated detections across calls - accepted as a known limitation.
