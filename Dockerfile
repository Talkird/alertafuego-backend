# AWS Lambda container image for the AlertaFuego backend (FastAPI + torch inference).
# Uses the AWS Lambda Web Adapter so the app runs as a plain uvicorn server -
# no Mangum handler, main.py is untouched.
FROM python:3.12-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter
ENV PORT=8080
ENV AWS_LWA_READINESS_CHECK_PATH=/

WORKDIR /app

COPY requirements-lambda.txt ./
RUN pip install --no-cache-dir -r requirements-lambda.txt

COPY backend ./backend
COPY model ./model

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
