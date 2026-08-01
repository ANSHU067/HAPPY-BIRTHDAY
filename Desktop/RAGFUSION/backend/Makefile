.PHONY: run test compile docker-up docker-down

run:
	python -m uvicorn main:app --reload

test:
	pytest

compile:
	python -m compileall .

docker-up:
	docker compose up --build

docker-down:
	docker compose down
