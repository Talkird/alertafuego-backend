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
   # Clave para correr detecciones (POST /detect/*), ej. `openssl rand -hex 32`:
   DETECT_API_KEY="una-clave-larga-y-aleatoria"
   # Opcional, para emails de alertas (ver "Alertas por zona"):
   ALERT_EMAIL_FROM="alertas@tu-dominio"
   FRONTEND_URL="http://localhost:3000"
   # Opcional, para alertas por Telegram (ver "Alertas por Telegram"):
   TELEGRAM_BOT_TOKEN="123456:ABC..."
   TELEGRAM_BOT_USERNAME="AlertaFuegoBot"
   TELEGRAM_WEBHOOK_SECRET="un-string-aleatorio"
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
    --train-end-date 2025-09-24 --val-end-date 2025-09-27 \
    --output-dir model/dataset/2025-09
```

Cada exportación (un mes) va en su propia carpeta: genera
`<output-dir>/{train,val,test}/sample_XXXXXX.npz` (`patch` `(16,H,W)` float32, `mask`
`(H,W)` uint8) + `<output-dir>/manifest.csv`. `--output-dir` es obligatorio. `--end-date`
es exclusivo (primer día del mes siguiente). `--limit N` acota la cantidad de muestras
para pruebas rápidas. Rango largo de fechas se procesa internamente en chunks diarios
(hay un tope de 5000 elementos por query de Earth Engine).

Cada día se guarda en disco apenas termina y queda registrado en
`completed_days.txt` dentro de `--output-dir`: si la exportación se corta, volver a
correr el mismo comando (mismas fechas y carpeta) retoma desde el día siguiente. Los
errores de Earth Engine, de red y los requests colgados (timeout de 2 minutos)
reintentan el día hasta 4 veces.

Después de exportar un mes, combinar todos los meses en un único manifest para
entrenar:

```
python -m model.scripts.merge_datasets
```

Escribe `model/dataset/manifest.csv` con todas las muestras de `model/dataset/*/`
(sin copiar archivos). Cada mes conserva su propio split train/val/test.

## Entrenamiento (fase 2)

```
python -m model.scripts.train_model --epochs 30
```

Usa el manifest combinado (`model/dataset/manifest.csv`, ver `merge_datasets`
arriba). Args útiles: `--dataset-dir`, `--checkpoint-dir` (default `model/checkpoints`),
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

Piden la clave de `DETECT_API_KEY` en el header `X-API-Key` (401 sin ella, 503 si el
backend no la tiene configurada): cada corrida gasta Lambda y cuota de Earth Engine,
escribe en la base y puede disparar alertas. En `/docs`, cargarla con **Authorize**.

```
curl -X POST -H "X-API-Key: $DETECT_API_KEY" <URL backend>/detect/argentina
```

- `GET /detections` — lectura pura sobre lo ya guardado, sin llamar a Earth Engine,
  rápido. Filtros opcionales por query params: `west/south/east/north` (bbox),
  `since`/`until` (rango de `image_time`, ISO datetime), `limit` (default 500, tope
  5000). Devuelve cada fila con forma `{id, lat, lon, probability, image_time,
  detected_at, bbox, threshold}` — **este es el endpoint que debería usar el
  frontend** para el mapa y la tabla de logs, no `/detect/*` (esos disparan un scan
  en vivo con efecto secundario, no son para lectura rutinaria).

### Alertas por zona

Un usuario autenticado define zonas de alerta (punto + radio en km, hasta 200, y una
probabilidad mínima, default 0.65). Después de cada corrida de `/detect/*`, las
detecciones nuevas dentro de una zona activa generan una alerta y un email al dueño.
Un mismo foco no vuelve a alertar a la misma zona si ya hubo una alerta a menos de
5 km en las 6 horas anteriores (por `image_time`).

- `GET/POST /alert-areas`, `PATCH/DELETE /alert-areas/{id}` — zonas del usuario.
- `GET /alerts` — historial de alertas del usuario, cada una con su detección.

Los emails salen por Amazon SES desde `ALERT_EMAIL_FROM` y enlazan a
`FRONTEND_URL/alertas`. Sin `ALERT_EMAIL_FROM` en `.env` no se envían (se loguea un
warning) pero las alertas se registran igual. Configuración de SES en
`terraform/README.md`, paso 5.

### Cuenta y canales de aviso

- `GET/PATCH /account` — canales del usuario (`email_alerts` prende o apaga el email).
- `POST /account/alert-email` — recibir los emails en otra dirección. Se le envía un
  enlace (vence en 24 h) a `FRONTEND_URL/confirmar-email`, que llama a
  `POST /account/alert-email/confirm` (sin login: el token alcanza). Sin
  `ALERT_EMAIL_FROM`, el enlace se loguea en vez de enviarse.
- `DELETE /account/alert-email` vuelve al email de la cuenta;
  `DELETE /account/alert-email/pending` cancela un cambio sin confirmar.

### Alertas por Telegram

Un bot de Telegram envía las alertas a quien lo conecte desde **Mi cuenta**:
`POST /account/telegram/link` devuelve un enlace de un solo uso
(`t.me/<bot>?start=<token>`, vence en 30 min); al tocar *Iniciar* en Telegram, el bot
recibe `/start <token>` y vincula ese chat. `/stop` en el chat lo desvincula, igual
que `DELETE /account/telegram`. `POST /account/telegram/test` envía un mensaje de
prueba. Si el usuario bloquea el bot, se desvincula solo.

Telegram entrega los mensajes al bot por webhook (`POST /telegram/webhook`, que
rechaza llamadas sin `TELEGRAM_WEBHOOK_SECRET`). Como no puede llegar a `localhost`,
en local se usa polling:

```
python -m backend.app.telegram poll                       # local (borra el webhook)
python -m backend.app.telegram set-webhook <URL backend>  # producción
```

Crear el bot y configurarlo en AWS: `terraform/README.md`, paso 6.

### Estadísticas

- `GET /stats/regions?kind=province|department` — detecciones por provincia o
  departamento.
- `GET /stats/hourly` — detecciones por hora del día (hora argentina).
- `GET /stats/reports` — reportes de usuarios por veredicto y categoría.

Todas aceptan `since`/`until`. `/stats/regions` necesita los límites del IGN cargados
en la tabla `regions` (una vez por base de datos, re-correrlo los reemplaza):

```
python -m backend.scripts.load_regions
```

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
