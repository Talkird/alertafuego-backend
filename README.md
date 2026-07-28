# alertafuego-backend

Sistema de detección de incendios casi en tiempo real para Argentina: combina
imágenes satelitales GOES-19 con un modelo de segmentación entrenado con focos de
calor VIIRS, servido desde un backend FastAPI que persiste detecciones en
Postgres/PostGIS.

El razonamiento detrás de cada decisión de arquitectura (por qué Postgres y no Mongo,
por qué esta forma de parche, por qué este umbral, bugs encontrados y cómo se
resolvieron, etc.) está en [`docs/DECISIONS.md`](docs/DECISIONS.md) — este README es
la guía operativa de "cómo correr esto hoy", no el historial de decisiones.

## Estado actual (2026-07-28)

- ✅ Fase 1 — pipeline de datos (`model/data_pipeline/`): exporta dataset GOES-19/VIIRS.
- ✅ Fase 2 — entrenamiento (`model/training/`): modelo entrenado, checkpoint en
  `model/checkpoints/` (F1=0.33 / IoU=0.20 en test, entrenado con un mes de datos —
  ver limitaciones abajo).
- ✅ Fase 3 — inferencia servida (`model/inference/` + `backend/app/`): dos endpoints
  on-demand que corren el modelo contra la imagen GOES-19 más reciente.
- ✅ Fase 4 — persistencia: las detecciones se guardan en Postgres/PostGIS (Supabase).
- ✅ Detecciones restringidas al polígono real de Argentina (no al rectángulo del bbox
  — evita falsos positivos en Paraguay/Bolivia/Chile/Brasil/Uruguay).
- ✅ `GET /detections` — endpoint de lectura sobre lo ya persistido (el que debería
  usar el frontend para el mapa/tabla de logs).
- ⏳ Pendiente: ampliar el dataset de entrenamiento a más de un mes/estación,
  clustering de detecciones, autenticación, deduplicar detecciones repetidas entre
  corridas.

## Setup

1. Instalar dependencias: `pip install -r requirements.txt`
2. Autenticar Earth Engine localmente (una sola vez): `earthengine authenticate`
3. Completar `.env`:
   ```
   EARTH_ENGINE_PROJECT_ID="tu-proyecto-de-gee"
   DATABASE_URL="postgresql+psycopg://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres"
   FRONTEND_ORIGINS="http://localhost:3000"
   ```
   `DATABASE_URL` es la conexión **directa** de Supabase (puerto `5432`), no el
   transaction pooler (puerto `6543`) — este backend es un proceso persistente, no
   serverless. Nota el sufijo `+psycopg` (sin eso SQLAlchemy busca psycopg2, que no
   está instalado).
4. Si es la primera vez que se apunta a esta base de datos, aplicar las migraciones:
   ```
   alembic upgrade head
   ```
   Esto crea la extensión `postgis` y la tabla `detections` (con índice espacial GiST).

## Pipeline de datos (fase 1)

Exporta un dataset de entrenamiento: busca detecciones VIIRS, las empareja con la
captura GOES-19 más cercana en tiempo, extrae parches de 16 bandas calibradas.

```
python -m model.scripts.export_dataset \
    --start-date 2025-09-01 --end-date 2025-10-01 \
    --train-end-date 2025-09-24 --val-end-date 2025-09-27
```

Genera `model/dataset/{train,val,test}/sample_XXXXXX.npz` (`patch` `(16,H,W)` float32,
`mask` `(H,W)` uint8) + `model/dataset/manifest.csv`. `--limit N` acota la cantidad de
muestras para pruebas rápidas. Rango largo de fechas se procesa internamente en chunks
diarios (hay un tope de 5000 elementos por query de Earth Engine).

## Entrenamiento (fase 2)

```
python -m model.scripts.train_model --epochs 30
```

Args útiles: `--dataset-dir`, `--checkpoint-dir` (default `model/checkpoints`),
`--batch-size`, `--lr`, `--limit` (subset chico para smoke test). Corre en GPU si hay
CUDA disponible (`torch.cuda.is_available()`), si no cae a CPU automáticamente. Guarda
`model_best.pt`, `norm_stats.json`, `metrics.csv` y `training_curves.png` en
`checkpoint_dir`.

## Backend / API (fases 3-4)

```
python -m uvicorn backend.app.main:app --reload
```

(Correr desde la raíz del repo — no `uvicorn backend.app.main:app` a secas — para que
los imports `model.*`/`backend.*` resuelvan.)

Al arrancar carga el modelo entrenado y autentica Earth Engine una sola vez (no por
request). Endpoints:

- `POST /detect/demo` — corre el modelo sobre un bbox chico y fijo (La Rioja/Catamarca/
  San Juan), responde en segundos.
- `POST /detect/argentina` — corre sobre todo el país (fetch en chunks espaciales por
  el límite de píxeles de Earth Engine), tarda ~50-55s.

Ambos: buscan la imagen GOES-19 más reciente, corren el modelo (filtrando resultados
al polígono real de Argentina, no al rectángulo del bbox), **guardan las detecciones
en Postgres**, y devuelven el mismo resultado en la respuesta HTTP (`image_time`,
`bbox`, `threshold`, `chunk_count`, `detections: [{lat, lon, probability}]`).

- `GET /detections` — lectura pura sobre lo ya guardado, sin llamar a Earth Engine,
  rápido. Filtros opcionales por query params: `west/south/east/north` (bbox),
  `since`/`until` (rango de `image_time`, ISO datetime), `limit` (default 500, tope
  5000). Devuelve cada fila con forma `{id, lat, lon, probability, image_time,
  detected_at, bbox, threshold}` — **este es el endpoint que debería usar el
  frontend** para el mapa y la tabla de logs, no `/detect/*` (esos disparan un scan
  en vivo con efecto secundario, no son para lectura rutinaria).

Docs interactivas en `/docs`. Ver esquema de la tabla `detections` y por qué se
modeló así en `docs/DECISIONS.md` (entradas de persistencia, fase 4).

## Limitaciones conocidas

- Dataset de entrenamiento cubre un solo mes (septiembre 2025) — el modelo tiene señal
  real pero no vio otras estaciones/regímenes de fuego. Reentrenar con más datos es la
  mejora de mayor impacto pendiente.
- Sin clustering: cada píxel de fuego es una detección separada, no agrupadas por
  foco/evento.
- Sin autenticación en ningún endpoint.
- Días con actividad de incendios muy alta recortan detecciones a un máximo de 200 por
  chunk de exportación (ver `MAX_DETECTIONS_PER_CHUNK` en `docs/DECISIONS.md`).
