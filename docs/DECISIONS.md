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

### [2026-07-28] — Filter detections to Argentina's real polygon, not the bounding rectangle

**Context:** `argentina_bbox` (`west=-73.6, south=-55.1, east=-53.6, north=-21.8`) is a rectangle, and a rectangle around Argentina's actual (irregular) shape necessarily includes slivers of Paraguay, Bolivia, Chile, Brazil, and Uruguay near the borders. This was caught in production: a real stored detection (id=1, `lat=-23.26, lon=-56.27`) landed outside Argentina - confirmed via point-in-polygon test against the real FAO GAUL country boundary already used for negative sampling (`data_pipeline/negative_sampling.py: load_argentina_land_geometry`).
**Decision:** Added `model/inference/land_filter.py`: `load_argentina_polygon(bbox)` fetches Argentina's real land boundary **once** (reusing `load_argentina_land_geometry`, converted to a `shapely` geometry via `.getInfo()` + `shapely.geometry.shape()`) and stores it on `InferenceContext` (loaded once at startup, like the model/norm stats); `filter_to_polygon()` then does a local point-in-polygon test (`shapely`, no extra Earth Engine call per request) on every detection before it's returned/persisted, in `service.run_detection()`.
**Alternatives considered:**
- Do the point-in-polygon test via an Earth Engine call per request (e.g. `FeatureCollection.filterBounds`) instead of caching a local `shapely` polygon.
- Only filter the `/detect/argentina` endpoint (skip it for `/detect/demo`, since that bbox doesn't touch a border).
**Rationale:** Caching the polygon once and filtering locally with `shapely` avoids adding another sequential Earth Engine round-trip to an already slow endpoint (~50s for `/detect/argentina`). Filtering unconditionally (regardless of which bbox was requested) is simpler than branching per-endpoint and is always correct if a bbox is ever changed to sit near a border later.
**Consequences:** Verified against real data: re-running `/detect/argentina` dropped 33 of 74 raw detections that fell outside Argentina's real border (a large fraction - much of the agricultural burning activity in this run was concentrated right along the Paraguay border). This does **not** retroactively clean the bad row already in the DB from before the fix (`id=1`) - existing bad data needs a manual/one-off cleanup, this only prevents new bad rows going forward. The same rectangle-vs-real-border issue equally affects phase 1's training data export (`matching.fetch_viirs_detections` also just uses the bbox) - not fixed here since it doesn't affect model correctness (fire physics doesn't respect political borders) and was out of scope for this fix, but noted for whoever revisits dataset export/labeling accuracy later.

### [2026-07-28] — Add `GET /detections`: read-only endpoint over stored data

**Context:** The only detection-related endpoints were `POST /detect/demo`/`POST /detect/argentina` - both trigger a live ~5-55s Earth Engine scan and a DB write as a side effect. A frontend (Nuxt) needs to display already-persisted detections (map + a "logs" table view matching the DB schema) without re-triggering a live scan on every page load, which would also insert duplicate rows on every visit.
**Decision:** Added `backend/app/routers/detections.py` (`GET /detections`, plural - distinct from the `/detect/*` action endpoints), backed by `crud.list_detections()` (SQLAlchemy + PostGIS `ST_MakeEnvelope`/`ST_Intersects` for optional bbox filtering, plus optional `since`/`until` on `image_time`, `limit` capped at 5000). Response schema `StoredDetection` mirrors the `detections` table columns directly (`id, lat, lon, probability, image_time, detected_at, bbox, threshold`), converting the stored geometry back to `lat`/`lon` via `geoalchemy2.shape.to_shape()`.
**Alternatives considered:** Reuse `DetectionResponse` (the action-endpoint response shape) for reads too.
**Rationale:** A dedicated `StoredDetection` schema matches what the user explicitly described wanting in the frontend (a logs table "siguiendo el schema de la base de datos") - `DetectionResponse` is shaped around one live scan (`image_time`/`bbox`/`threshold` once, `detections` list nested inside), whereas reads need each row's own `id`/`image_time`/`bbox`/`threshold`/`detected_at` independently, since rows come from many different past scans.
**Consequences:** This is the endpoint the frontend should actually build against for the map and logs table - the `/detect/*` endpoints remain for triggering a fresh scan (e.g. a manual "refresh now" button or a future scheduled job), not for routine reads. No pagination beyond `limit`/`since`/`until` (no cursor/offset) - fine at current data volume, revisit if the table grows large.

### [2026-09-29] — Write dataset exports day by day, resume after crashes, and retry failed days

**Context:** A full-month export (April 2025) crashed after ~30 minutes when Earth Engine rejected a single call with a 403 "Not signed up for Earth Engine or project is not registered", after hundreds of identical calls had succeeded in the same run (the project was confirmed working again minutes later). Because `export_dataset.main()` held every sample in memory and only wrote files at the end, the 3 finished days were lost; a month takes ~6.5h (~12.5 min per capped day, measured), so any single failure wasted hours.
**Decision:** `export_dataset.py` now saves each day's `.npz` files and appends its rows to `manifest.csv` as soon as the day is built, records the day in `completed_days.txt` inside `--output-dir`, and on start skips days already listed there (continuing the sample numbering from the manifest). A day that raises `ee.EEException` or `OSError` is rebuilt from scratch up to 4 times, waiting 60s/120s/180s between attempts.
**Alternatives considered:**
- Retry each individual Earth Engine call (wrapping every `getInfo()`/`computePixels` in matching, patches, labels and negative sampling).
- Rely on Earth Engine's built-in retries (`googleapiclient` `num_retries`, which only retries 429/5xx, not this 403).
- Keep the all-at-end write and export shorter ranges (e.g. one week) per run into separate folders.
**Rationale:** Day-level retry covers every Earth Engine call in one place instead of touching four modules; rebuilding a failed day costs at most ~12 min, negligible against a 6h run. Per-day writes bound the loss from a hard crash to the day in progress and make "re-run the same command" the recovery procedure, instead of manually splitting ranges.
**Consequences:** `write_manifest` (overwrite) became `append_manifest` (append, header only when the file is new), so re-running into an existing `--output-dir` now adds to it instead of clobbering it — this reverses the "overwrites on every invocation" behavior noted in the 2026-07-25 chunking entry. Each export (e.g. each month) must still use its own `--output-dir` to keep datasets separate, and a resumed run must reuse the same split dates. If a crash lands between appending a day's manifest rows and recording it in `completed_days.txt` (a sub-millisecond window), that day is re-exported on resume and its rows appear twice; not handled.

### [2026-10-01] — Keep each dataset export in its own folder and train on a merged manifest; add a 2-minute Earth Engine request deadline

**Context:** A second month (April 2025) was exported alongside September 2025, but `train_model.py` reads a single `--dataset-dir/manifest.csv`, and September's manifest (exported on Windows) stores backslash paths that don't resolve on macOS/Linux. Separately, the April export hung for 3+ hours on one Earth Engine HTTPS request (open connection, 0% CPU): requests have no timeout by default, so the day-level retry never fired.
**Decision:** Every export lives in `model/dataset/<YYYY-MM>/` (September moved from the root into `model/dataset/2025-09/`), `export_dataset.py` now requires `--output-dir`, and a new `model/scripts/merge_datasets.py` writes `model/dataset/manifest.csv` listing all samples from every subfolder with POSIX paths relative to `model/dataset`, without copying files. `export_dataset.py` also calls `ee.data.setDeadline(2 min)` after initializing Earth Engine.
**Alternatives considered:**
- Add a `--manifest`/multiple `--dataset-dir` option to `train_model.py` instead of merging manifests.
- Copy or hardlink all months into one renumbered dataset folder.
- A 5-minute deadline, or setting the deadline in `ee_client.init_earth_engine()` (shared with the API server).
**Rationale:** A merged manifest needs no change to the training code (its default `--dataset-dir model/dataset` keeps working) and costs no disk space; keeping months separate makes each one independently resumable and removable. Measured: with a deadline, googleapiclient retries a timed-out call 5 times and then raises `TimeoutError` (an `OSError`), which the day-level retry already catches — so the worst case per stalled call is roughly 6× the deadline; 2 minutes bounds that to ~12 min, while every call in the pipeline normally returns in seconds. Setting it only in the export script keeps the API server's behavior unchanged.
**Consequences:** `merge_datasets` must be re-run after each new month, before training. Each month keeps its own temporal train/val/test split, so val/test now span every month. Running `export_dataset` without `--output-dir` is an error instead of writing into `model/dataset/`. Training center-crops patches, so the 33×33 patches some exports produce are fine.

### [2026-10-12] — Region alerts: point + radius areas, checked after each detection run, emailed through SES

**Context:** Users should be able to pick an area and be told when a fire appears there. Detection runs only happen on demand (`POST /detect/*`, ~1 minute for all of Argentina). There's no scheduler, and adding one was deferred.
**Decision:** `alert_areas` stores a per-user circle (Geography `POINT` center + `radius_km` up to 200 + `min_probability`, default 0.65). Right after `save_detections()`, `backend/app/alerts.py` matches the new rows with `ST_DWithin`, records each match in `alert_notifications`, and sends the area's owner one plain-text email per area per run through Amazon SES (`ALERT_EMAIL_FROM`). The owner's email is copied from the JWT `email` claim on create and edit.
**Alternatives considered:**
- Free-drawn polygons or administrative regions (provinces and departments) as alert areas.
- A queue or separate worker for matching and sending.
- Resend or another email API instead of SES, or WhatsApp or Telegram.
- Looking up the user's email through the Supabase admin API at send time.
**Rationale:**
- Point + radius is the simplest UX ("my farm, 25 km") and maps directly onto `ST_DWithin` on Geography, in meters.
- Matching is one indexed query over a single run's detections, so a queue would add infrastructure without a need.
- SES fits the existing AWS deploy, and the Lambda role only needs `ses:SendEmail`. WhatsApp may come later.
- Copying the email from the JWT avoids giving the backend a Supabase service key.
- The 0.65 default sits above the model's 0.5 detection threshold. At test F1 = 0.33, false alarms are the main risk to alerts being trusted.
**Consequences:**
- Alerts fire only when someone triggers a run, until scheduled detection exists. Once a scheduler calls `/detect/argentina`, alerts work with no further changes.
- Cooldown: a detection doesn't alert an area again if that area already alerted within 5 km and 6 h (by `image_time`), and neighbouring pixels in one run collapse into one alert. A long-burning fire re-alerts every 6 h.
- Email failures are logged and leave `emailed_at` null. They never fail the detection run. Without `ALERT_EMAIL_FROM`, alerts are recorded but not emailed.
- SES starts in sandbox mode (only verified recipients, 200 emails a day). Production access must be requested before real users sign up (`terraform/README.md`, step 5). The sender is verified by hand because the CI deploy role has no SES permissions.
- Observed while testing: almost all stored detections sit at the same position inside their 32-px inference tile (rows 15–16, columns 18–23). That suggests a positional bias in the model, possibly from training patches centered on the VIIRS fire. If so, reported locations can be tens of km off, which matters for small alert radii. Not addressed here.

### [2026-10-12] — Load IGN provinces and departments into a `regions` table for per-region stats

**Context:** The dashboard needed detections per province and department, and nothing in the database knew Argentina's administrative boundaries.
**Decision:** `python -m backend.scripts.load_regions` downloads IGN's WFS layers `ign:provincia` (24) and `ign:departamento` (529), simplifies them to ~100 m, and replaces the `regions` table. `GET /stats/regions` joins detections with `ST_Contains`. `GET /stats/hourly` (Argentina time, UTC-3) and `GET /stats/reports` were added alongside it.
**Alternatives considered:**
- Commit the GeoJSON (~250 MB raw) to the repo.
- Reverse-geocode each detection at insert time into province and department columns.
- Compute the stats in the frontend from `GET /detections`.
**Rationale:**
- IGN is the official source, and the same IGN tiles are already used for the basemap.
- Simplifying to ~100 m is far below the ~2 km GOES pixel, and keeps the table small.
- Joining at query time means re-loading boundaries never requires backfilling detections.
- Stats in the frontend would be capped by `/detections`' row limit (default 500).
**Consequences:**
- The loader has to be run once per database. It has been run on the Supabase DB.
- Detections outside every region (offshore) are not counted in `/stats/regions`.

### [2026-10-13] — Add Telegram alerts and a confirmed alternate alert email, stored in a user_settings table

**Context:** Users asked for an account page where they choose how alerts reach them: Telegram, and an email other than their login's. Each `alert_areas` row copies the account email from the JWT, and there was nowhere to keep per-user settings.
**Decision:**
- A `user_settings` table (one row per user) holds an email on/off switch, a confirmed `alert_email`, a pending address with its 24-hour confirmation token, and the connected Telegram chat.
- Telegram goes through a bot. A user connects by opening a one-time `t.me/<bot>?start=<token>` link (30 minutes) from their account page and pressing Start. The bot receives updates on `POST /telegram/webhook`, checked with a secret header.
- Each alert records `emailed_at` and `telegram_sent_at` separately.
**Alternatives considered:**
- Let users type any address, or a Telegram username, without confirmation.
- Store the alternate address on each alert area.
- Ask for a phone number now, for later WhatsApp or SMS alerts.
- Poll Telegram (`getUpdates`) in production instead of a webhook.
- Per-user defaults for new zones (built, then dropped at the user's request before merge).
**Rationale:**
- Without confirmation, anyone could point alerts at someone else's inbox. A bot can only message a chat after the user presses Start, so the deep link both confirms and connects.
- Only token hashes are stored, so a database leak doesn't expose working links.
- The backend runs on Lambda behind a public function URL: a webhook needs no always-on process, polling would.
- WhatsApp and SMS cost money per message, and WhatsApp needs Meta approval. The phone field is shown as "Próximamente" in the frontend; nothing stores it yet.
**Consequences:**
- Telegram needs `TELEGRAM_BOT_TOKEN`, `TELEGRAM_BOT_USERNAME` and `TELEGRAM_WEBHOOK_SECRET`, plus a one-time `python -m backend.app.telegram set-webhook <URL>`. Without them the account page shows Telegram as unavailable.
- Local runs use `python -m backend.app.telegram poll`, which removes the webhook; run set-webhook again afterwards.
- A user who blocks the bot is disconnected automatically on the next message.
- While SES is in sandbox mode, confirmation and alert emails only reach SES-verified addresses.
- Account deletion is not implemented: removing the Supabase `auth.users` row needs the service-role key or a privilege the backend doesn't use yet.

### [2026-10-13] — Require an API key to run detections

**Context:** `POST /detect/demo` and `POST /detect/argentina` were open to anyone. Each call runs about a minute of Lambda, uses Earth Engine quota, writes detections and runs alert matching (emails, Telegram). The frontend never calls them.
**Decision:** Both endpoints require the `DETECT_API_KEY` secret in an `X-API-Key` header, compared in constant time. They answer 401 without it, and 503 when the backend has no key configured (fail closed). Every other endpoint is unchanged: user data needs a Supabase JWT, and detections, stats and the anonymous report list stay public.
**Alternatives considered:**
- An API key shipped in the frontend, so that "only the app" can call the API.
- Supabase login for `/detect` (any signed-up user could run detections).
- A Lambda reserved-concurrency cap.
- CloudFront + AWS WAF rate limiting.
**Rationale:**
- A key in the frontend is readable by any visitor in the browser's developer tools, so it can't restrict callers. Only a secret kept on servers can.
- Signing up is open, so a login check wouldn't limit who runs detections.
- The account's Lambda concurrency limit is 10, and AWS keeps at least 10 unreserved, so a reservation can't be set. The account limit already caps simultaneous runs at 10.
- WAF costs a few dollars a month and needs CloudFront in front of the function URL. That isn't warranted yet.
**Consequences:**
- Running a detection by hand needs the header (`curl -H "X-API-Key: ..."`, or **Authorize** in `/docs`). The future scheduler must hold the key.
- New GitHub secret `DETECT_API_KEY`, passed to Lambda through Terraform. It is required, so the deploy fails until it's set.
- Local `.env` needs `DETECT_API_KEY` to run detections.

### [2026-10-19] — Run detection on a schedule, in a separate Lambda, once per GOES-19 image

**Context:** Nothing ran `/detect/argentina` automatically: detections and alerts only happened when someone called it by hand (the last run was on 2026-09-29), which falls short of the system's near-real-time goal. A run takes ~50 s and loads the model; the API Lambda serves the app.
**Decision:**
- A second Lambda, `alertafuego-detector`, from the same image, invoked every 10 minutes (GOES-19's full-disk cadence) by EventBridge Scheduler. The Lambda Web Adapter forwards the scheduler's event as a POST to `/internal/scheduled-detection`, a route mounted only when `DETECTION_WORKER=1`, so it doesn't exist on the public API.
- A `detection_runs` table, one row per processed image (`image_time` unique), with status, counts and timings. A run first checks, without loading the model, whether another run is in progress or the latest image was already processed, and stops if so.
- The schedule ships disabled, enabled after a manual run confirms the deployed model's output.
**Alternatives considered:**
- An EventBridge rule calling the public `/detect/argentina` with the API key. Runs would occupy the instances serving the app.
- Reserved concurrency of 1 to prevent overlapping runs. The account's limit is 10 concurrent executions and AWS keeps at least 10 unreserved, so none can be reserved.
**Rationale:** A separate function keeps detection runs from slowing the app (the API's cold start stays ~2 s, without torch). Deduplicating by image avoids storing the same detections twice when the schedule fires before Earth Engine has a new image. Overlap is prevented in the database because concurrency can't be reserved.
**Consequences:**
- Re-apply `terraform/bootstrap` once: the deploy role needs `scheduler:*` on the project's schedules.
- `detection_runs` measures the automatic operation (latency from capture to detection, run duration, failures), usable as results.
- Cost: ~144 runs a day of ~50 s at 3 GB, roughly within Lambda's free tier to a few dollars a month (estimate); also 144 runs a day of Earth Engine requests.
- With the model's 0.08 threshold, the number of detections per run over the whole country needs checking before enabling the schedule.

### [2026-10-19] — Scheduled detection every 20 minutes (kept disabled for now)

**Context:** The first manual run of the detector (2026-10-19) processed the 01:30 UTC GOES-19 image: 27 detections over Argentina (threshold 0.08, confirming b-shift is deployed; 10 at probability >= 0.65), in 199.7 s (the detector's first run, including the new image's cold start and loading the model), peaking at 2.1 GB. The image was ~37 minutes old when the run started: Earth Engine's delay in publishing GOES-19, which the system can't shorten. A fire reached the map 41 minutes after the capture.
**Decision:** Run the schedule every 20 minutes rather than every 10, and log each run's result (`logging.basicConfig` at INFO). The schedule stays disabled (`detection_schedule_enabled = false`): checking the run against NOAA's FDCF fire product and VIIRS showed misplaced detections (fixed: fetched chunks were south-up) and, in daytime scenes, false positives with no relation to VIIRS fires. It is enabled once the model's full-scene precision is acceptable.
**Alternatives considered:**
- Every 10 minutes, GOES-19's cadence. With Earth Engine's ~40-minute delay, alerts would arrive ~40-50 minutes after a capture instead of ~40-60, at twice the runs: at ~3.3 minutes a run, about USD 35 a month of Lambda versus about half that (estimates from Lambda's public pricing).
**Rationale:** Earth Engine's delay dominates the latency, so halving the runs costs little in timeliness and halves the cost and Earth Engine usage.
**Consequences:**
- Measure real run durations in `detection_runs` for a day; if runs are short (~1 minute), go to every 10 minutes (`detection_schedule`).
- Half the GOES-19 images are skipped; a fire visible in only one 10-minute image can be missed.

### [2026-10-22] — Train and pick the threshold for fire recall (target 95%), not F1

**Context:** Recall is the thesis's primary metric (target >95%): a missed fire is worse than a false alarm, and precision is the accepted trade-off. Training picked the checkpoint and threshold by strict pixel F1, and no metric measured whether each fire was found. Measured on val for the production `b-shift` checkpoint: fire recall (share of VIIRS fires, one per positive sample, with a prediction within 1 pixel) is 76.7% at threshold 0.65 (tolerant pixel precision 92.1%), 81.8% at the deployed 0.08 and 84.0% at 0.02. Lowering the threshold alone can't reach 95%.
**Decision:**
- New metrics: `fire_recall` and `false_alarm_rate` (fire-free patches with any prediction) per sample (`metrics.fire_counts`), and F2 of tolerant precision and fire recall.
- Loss: Dice replaced by Tversky with false negatives weighted 0.7 and false positives 0.3 (`--fn-weight`; 0.5 is Dice). BCE stays.
- Checkpoint: the epoch with the best val F2 over all threshold candidates (was F1 at a fixed 0.5).
- Threshold: the highest candidate whose val fire recall reaches `--recall-target` (default 0.95), else the one with the highest fire recall. Candidates now go down to 0.005.
- Training and evaluation use Apple MPS when there is no CUDA.
**Alternatives considered:**
- Only lowering the threshold on the current model: tops out at ~84%.
- Strict pixel recall as the target: GOES's 2 km pixels can't place VIIRS's 375 m fires exactly, so it understates what an alert needs.
- BCE `pos_weight` or focal loss instead of Tversky.
- Picking the checkpoint by fire recall alone: an epoch that predicts fire everywhere would win.
**Rationale:** Fire recall measures what an alert misses. Tversky penalizes misses directly. F2 favors recall without rewarding degenerate predictions. The threshold rule spends precision only as far as the target needs.
**Consequences:**
- The new model must be trained (~1.5 h on the desktop GPU) and compared with `b-shift` on fire recall. 95% isn't guaranteed: some VIIRS fires are likely too small or brief for GOES-19 to see at all.
- If no threshold reaches the target, the lowest-threshold result is saved, which can mean many false alarms. Check `false_alarm_rate` in `results.json` before deploying.
- More false alarms in production. Scheduled detection is already disabled over daytime false positives, so a lower threshold makes that worse.
- `metrics.csv` gains `val_threshold`, `val_fire_recall` and `val_f2`; old checkpoints' files keep the old columns.

### [2026-10-22] — Measure on full scenes: production finds ~10% of VIIRS fires; patch metrics hid a shift leak

**Context:** Every metric so far came from 32x32 patches centered within 8 pixels of a VIIRS fire, or on random fire-free land. Production runs the model over all of Argentina in non-overlapping 32x32 tiles. Stage 1 of the recall plan (`tesis/docs/plan-recall-no-export.md`) measures the production setting.
**Decision:**
- `model/scripts/evaluate_scenes.py`: `fetch` caches 30 full Argentina scenes from Earth Engine once (`model/scene_eval/`, gitignored, ~800 MB), at GOES-19 images matched to VIIRS overpasses on val/test days, one day and one night per month plus 6 random (17 day, 13 night). Each scene stores the 16-band raster from the production fetch code, NOAA's FDCF Mask on the same grid, and every VIIRS detection within ±12 h. `score` runs offline with the production tiling.
- Scene scoring inside Argentina's polygon. Fire pixels are VIIRS detections (confidence ≥ nominal) within 15 min of the image. `fire_recall`: fire pixels with a prediction within 1 pixel. `cluster_recall`: groups of adjacent fire pixels with any of them found. A false detection is a prediction with no VIIRS detection of any confidence within 1 pixel and 12 h. That is an upper bound, since VIIRS misses fires under clouds.
- VIIRS is queried in 3-hour slices, filtered by acquisition time on the server, and split into quarters when a query hits Earth Engine's 4,000-feature cap. `matching.fetch_viirs_detections` caps whole days in collection order: on 2025-09-25 it returned 6,128 of 17,908 detections.
- `evaluate_model --by-stratum` and `results.json`'s `test_by_stratum` score per UTC hour and day/night (sun above the horizon at the sample).
- `merge_datasets --holdout-months 2025-07 2026-01` writes `manifest_holdout.csv`: those two months (winter, summer) are the whole test set; the other months' test days join val; nothing moves into train (train 94,349 / val 26,400 / test 24,798). Used with `--manifest` in train_model and evaluate_model.
**Results (b-shift, the production model):**

Full scenes, 30 scenes (day 17, night 13):

| Threshold | Fire pixel recall | Cluster recall | Precision (VIIRS ±12 h) | False pixels / scene | False clusters / scene |
|---|---|---|---|---|---|
| 0.02 | 13.2% (day 15.1, night 7.5) | 9.2% | 12.0% | 130 | 83 |
| 0.08 (deployed) | 9.9% (day 11.1, night 6.2) | 6.7% | 11.6% | 92 | 63 |
| 0.65 | 5.6% (day 6.3, night 3.7) | 3.4% | 11.5% | 37 | 32 |

a-baseline (trained without shifts) is about the same: 8.0% fire pixel recall and 56 false clusters per scene at 0.08.

Recall by the fire pixel's distance to the nearest tile edge (0.08): 2% on the edge, 8-21% from 3 to 13 pixels in, 62% at 14 and 95% at the center (15). The center rings hold only 21 and 76 fire pixels.

Recall by VIIRS fire strength (FRP summed per GOES pixel), b-shift at 0.08 vs NOAA's FDCF (codes 10-15, 30-35): 0-5 MW 2.0% vs 1.4%; 5-10 MW 2.3% vs 4.3%; 10-20 MW 3.9% vs 8.1%; 20-50 MW 8.3% vs 16.9%; 50-100 MW 17.8% vs 33.9%; over 100 MW 46.2% vs 75.5%.

Patches, test, threshold 0.08, by hour: fire-free patches with any prediction 9.3-13.0% at 15-17 UTC (12-14 local) vs 0.6-4.5% at other hours. Fire recall day 86.3%, night 75.9%.
**Findings:**
- **Shift leak.** `dataset.shift` fills the uncovered border with zeros, so the border's width tells the model where the fire is. On val fire patches, b-shift finds 83.5% of fires shifted with a zero border, as in training, and 19.3% when the same shift wraps real pixels around instead. The model relies on "fire at the patch center", which production tiles don't provide. It explains the tile-center pattern above and false positives clustering at tile centers. Every patch metric so far overstates production, including the 77-84% fire recall that motivated `feat/recall-first`.
- **GOES-19 can't see most VIIRS fires.** Even NOAA's operational product finds 1-8% of fire pixels under 20 MW and 75% over 100 MW. 95% recall against all VIIRS fires is out of reach for any GOES-based detector. A reachable target needs a fire-strength floor (e.g. "fires over 100 MW").
- Overlapping tiles (stride 16, keeping each tile's central 16x16) didn't help b-shift on two scenes (recall 0-10%, about 4x the false detections), so the fix belongs in training, not tiling.
**Alternatives considered:** Keep patch metrics as the headline (they don't reflect production); score against VIIRS over the whole bbox without the ±12 h confirmation (counts fires from other overpasses as false).
**Consequences:**
- Full-scene metrics are the headline from now on; patch metrics only compare training runs.
- Stage 3 has to remove the shift leak (augment without a zero border) before any other training change is worth measuring.
- Report recall per fire-strength band, and set the thesis target on fires GOES can physically see.
- The 30-scene set is fixed (`scenes.csv`, seed 0) so later models are scored on the same scenes.

### [2026-10-22] — FDCF baseline: NOAA's fire product beats b-shift on full scenes, and is published 28 minutes sooner

**Context:** Stage 2 of the recall plan. NOAA's GOES-19 fire product (`NOAA/GOES/19/FDCF`) is on the same 2 km grid as the imagery the model uses, and is already cached for the 30 evaluation scenes. It is the operational baseline any GOES model has to beat, and a candidate to combine with the model for recall.
**Decision:**
- `evaluate_scenes score` also scores FDCF and "model OR FDCF" on the same scenes and metrics. An FDCF fire pixel is a Mask code in 10-15 (processed, saturated, cloud-contaminated, high, medium and low probability fire) or 30-35 (the same, temporally filtered). FDCF has no threshold.
- Publishing delay measured from Earth Engine's ingestion timestamp (`system:version` minus `system:time_start`) over 414 images per collection (2026-10-19 to 2026-10-22). It matches the age of the newest images checked directly (41 and 11 minutes).
**Results (30 scenes):**

| Detector | Fire pixel recall (day / night) | Cluster recall | Precision (VIIRS ±12 h) | False pixels / scene | False clusters / scene |
|---|---|---|---|---|---|
| FDCF | 17.3% (21.4 / 5.0) | 12.2% | 70.9% | 14 | 5.3 |
| b-shift, 0.08 (deployed) | 9.9% (11.1 / 6.2) | 6.7% | 11.6% | 92 | 63 |
| b-shift, 0.65 | 5.6% (6.3 / 3.7) | 3.4% | 11.5% | 37 | 32 |
| b-shift 0.08 OR FDCF | 19.3% (23.4 / 7.4) | 14.2% | 27.5% | 106 | 68 |
| b-shift 0.65 OR FDCF | 18.1% (22.3 / 5.8) | 13.1% | 41.9% | 51 | 37 |

By fire strength (from the stage 1 entry), FDCF finds 1-8% of fire pixels under 20 MW, 17% at 20-50 MW, 34% at 50-100 MW and 75% over 100 MW, about twice the model in every band above 5 MW.

Publishing delay after capture: GOES-19 imagery (MCMIPF, what the model reads) median 40 min (p10 31, p90 56, max 92); FDCF median 12 min (p10 12, p90 14, max 26).

Per scene, threshold 0.08:

| Scene (UTC) | Day/night | VIIRS fire px | Recall: model / FDCF / OR | False clusters: model / FDCF |
|---|---|---|---|---|
| 2025-04-25 04:50 | night | 78 | 0% / 0% / 0% | 49 / 0 |
| 2025-04-28 17:40 | day | 528 | 6% / 13% / 16% | 152 / 1 |
| 2025-05-30 05:20 | night | 57 | 7% / 0% / 7% | 8 / 0 |
| 2025-05-30 19:20 | day | 15 | 0% / 0% / 0% | 3 / 0 |
| 2025-06-27 17:20 | day | 313 | 4% / 13% / 15% | 66 / 3 |
| 2025-06-28 05:00 | night | 40 | 10% / 0% / 10% | 4 / 0 |
| 2025-07-25 04:30 | night | 8 | 0% / 0% / 0% | 22 / 0 |
| 2025-07-28 17:40 | day | 28 | 4% / 0% / 4% | 37 / 0 |
| 2025-08-28 18:20 | day | 1396 | 10% / 18% / 20% | 94 / 10 |
| 2025-08-29 05:10 | night | 316 | 1% / 2% / 2% | 107 / 0 |
| 2025-09-25 17:30 | day | 784 | 8% / 21% / 22% | 128 / 10 |
| 2025-09-26 05:10 | night | 362 | 2% / 3% / 3% | 16 / 0 |
| 2025-09-26 19:10 | day | 118 | 37% / 68% / 75% | 52 / 25 |
| 2025-10-27 17:50 | day | 47 | 0% / 4% / 4% | 61 / 4 |
| 2025-10-28 19:00 | day | 75 | 0% / 8% / 8% | 32 / 1 |
| 2025-10-30 06:10 | night | 19 | 0% / 0% / 0% | 14 / 0 |
| 2025-11-25 04:20 | night | 113 | 32% / 33% / 35% | 69 / 1 |
| 2025-11-28 19:10 | day | 15 | 0% / 0% / 0% | 48 / 3 |
| 2025-11-29 06:30 | night | 0 | - / - / - | 12 / 0 |
| 2025-11-30 17:00 | day | 12 | 8% / 8% / 8% | 45 / 18 |
| 2025-11-30 18:30 | day | 33 | 18% / 30% / 30% | 35 / 8 |
| 2025-12-25 05:20 | night | 61 | 18% / 16% / 23% | 20 / 11 |
| 2025-12-27 18:30 | day | 132 | 43% / 66% / 67% | 140 / 8 |
| 2025-12-29 18:00 | day | 114 | 32% / 46% / 46% | 168 / 11 |
| 2026-01-29 05:50 | night | 109 | 10% / 3% / 10% | 56 / 0 |
| 2026-01-31 18:00 | day | 186 | 17% / 23% / 28% | 187 / 26 |
| 2026-02-25 05:40 | night | 70 | 0% / 0% / 0% | 69 / 0 |
| 2026-02-26 19:30 | day | 7 | 0% / 0% / 0% | 67 / 11 |
| 2026-03-26 04:50 | night | 65 | 8% / 0% / 8% | 7 / 0 |
| 2026-03-28 17:10 | day | 7 | 0% / 29% / 29% | 134 / 7 |

**Findings:**
- FDCF finds about 1.7x the model's fires with about 1/12 of its false clusters. Today's model adds nothing over FDCF by day; OR-ing them adds 2 points of recall and 63 false clusters per scene.
- At night both find little (5-6%). The model finds some night fires FDCF misses (2025-06-28, 2026-01-29, 2026-03-26), the only place it adds recall.
- FDCF is in Earth Engine ~28 minutes before the imagery the model needs. An FDCF-based alert could reach users ~30 minutes sooner than today's detection.
- FDCF night precision is low (17.7%) on few pixels: 13.6 false pixels per scene but 0.9 false clusters, so a few persistent hot spots, likely industrial.
**Alternatives considered:**
- Count only FDCF's high-confidence codes (10, 11, 13, 30, 31, 33): left for later; the full set is FDCF's own definition of a fire pixel.
- Fetch FDCF per test sample for patch-level comparison: deferred (needs an export), and the patch metrics are unreliable anyway (see the stage 1 entry).
**Consequences:**
- The thesis can report FDCF as the operational baseline on the same 30 scenes.
- Until a retrained model beats FDCF on full scenes, the system can't claim the model improves on NOAA's product. The best production option today, for recall and latency, is FDCF alone, or FDCF plus the model at night only.
- Any retrained model is compared against this table (same scenes, same metrics).

### [2026-10-26] — Stage 3: predict whole scenes, day/night thresholds, and retrain without the shift leak or the hour shortcut

**Context:** Stage 3 of the recall plan. The stage 1 entry found that production tiles only find fires at tile centers (shift leak) and that fire-free samples at 16-19 UTC are scarce (17-18 UTC: 90-94% of train samples are fire). This entry covers what changed without training and prepares one desktop training run.
**Decision:**
- `predictor.predict_raster`: the model runs on the whole raster in 512 px blocks with a 64 px margin (same output as one pass, ~5 s on CPU) instead of 32x32 tiles. Used by `service.run_detection` and `evaluate_scenes`. Optional flip/rotation averaging (TTA, `--tta`, ~29 s on CPU).
- Day/night thresholds: `ModelConfig.night_threshold` (optional; old checkpoints keep one threshold), applied per pixel by sun position in `run_detection`; each detection stores its own threshold. `evaluate_scenes calibrate` picks both by best F2 on the val-day scenes.
- Training, behind options (defaults unchanged until the run confirms them):
  - `--random-crop --crop-size 24 --max-shift 0`: crop 24x24 at a random position inside the stored 33x33 patch (val/test: fixed per sample) instead of shifting with a zero-filled border. No pixel is invented, so nothing marks the fire's position. On 1,478 val fire patches, none loses its fire to the crop.
  - `--balance-hours`: a weighted sampler (`strata.hour_balance_weights`) gives fire and fire-free samples the same weight within each UTC hour, at most 10x per sample. Hours with only one kind keep weight 1: 12 of 24 hours have no fire samples, and 07 and 20 UTC (3 and 21 fires) stay unbalanced at the cap. The effective fire share drops from 50% to 33% of train draws, and 943 fire-free samples at 18 UTC are drawn ~8x per epoch, so augmentation carries more weight.
**Results (b-shift, no retraining):**

Test-day scenes (19), calibrated thresholds day 0.0002 / night 0.0005: fire pixel recall 54.3% (day 61.2, night 27.0), cluster recall 48.9%, precision 33%, 155 false clusters per scene. With TTA: 55.9%, precision 36%, 140 false clusters. Same scenes with tiles at 0.08: 8.8% recall, 8.7% precision, 67 false clusters. FDCF: 14.8%, 70%, 5.

Threshold sweep on the val-day scenes (11; day 5, night 6), whole-scene prediction:

| Threshold | Fire pixel recall (day / night) | Cluster recall | Precision (VIIRS ±12 h) | False clusters / scene (day / night) |
|---|---|---|---|---|
| 0.0002 | 58.4% (64.5 / 45.9) | 51.7% | 36.0% | 229 (257 / 205) |
| 0.0005 | 52.3% (57.1 / 42.6) | 43.8% | 46.5% | 97 (91 / 103) |
| 0.001 | 46.4% (51.9 / 35.0) | 37.8% | 53.5% | 51 (39 / 60) |
| 0.002 | 38.4% (42.0 / 31.0) | 29.6% | 62.2% | 27 (18 / 34) |
| 0.005 | 26.9% (29.4 / 21.8) | 19.3% | 74.5% | 11 (7 / 15) |
| 0.01 | 20.9% (22.1 / 18.6) | 14.1% | 81.2% | 6.2 (3.4 / 8.5) |
| 0.02 | 17.3% (18.4 / 15.3) | 11.3% | 88.5% | 2.8 (1.4 / 4.0) |
| 0.08 (deployed) | 12.9% (13.5 / 11.8) | 8.3% | 93.3% | 0.8 (0.2 / 1.3) |
| 0.65 | 6.4% (7.0 / 5.1) | 3.2% | 98.3% | 0.1 |
| FDCF | 20.8% (27.0 / 8.3) | 14.0% | 71.3% | 5.6 (10.0 / 2.0) |

**Findings:**
- Predicting the whole raster removes most of the tile damage without retraining: at the deployed 0.08, precision goes from 11.6% (tiles, all 30 scenes, stage 1) to 93.3% (val scenes).
- On val scenes b-shift now matches FDCF: at 0.01 it finds the same share of fire pixels (20.9% vs 20.8%) with about the same false clusters (6.2 vs 5.6). At night it finds 2-5x FDCF's recall at every threshold up to 0.02.
- The F2 calibration picks the lowest thresholds (~150-230 false clusters per scene), too many for alerts. The operating point should come from a false-alarm budget, not F2.
**Alternatives considered:**
- Reflect-pad or wrap the shifted border: no zeros, but the mirror axis or seam still marks the shift. Rejected for the random crop, which adds no synthetic pixels and costs 8 px of context (24 vs 32).
- Downsample fire samples at 16-19 UTC to balance hours: discards ~26,000 fire samples. Rejected for weighting.
**Consequences:**
- Pending, user decision: the operating point (false clusters per scene the system accepts).
- Pending, desktop run: train with `--random-crop --crop-size 24 --max-shift 0 --balance-hours` on `manifest_holdout.csv`, calibrate on val scenes, and compare with b-shift on the same scenes, per hour, and with and without TTA. If it wins, make these options the defaults.
- `evaluate_model` still scores patches with the zero-border shift; patch metrics remain for comparing runs only.
