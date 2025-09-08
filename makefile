.PHONY: train serve docker-build docker-run


train:
python -m src.train


serve:
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload


docker-build:
docker build -f docker/Dockerfile -t deepfake-detector:latest .


docker-run:
docker run --rm -p 8080:8080 -v $(PWD)/app/weights:/app/app/weights deepfake-detector:latest