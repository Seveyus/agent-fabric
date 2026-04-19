dev-up:
	docker compose -f infra/compose/docker-compose.yml up -d --build

dev-down:
	docker compose -f infra/compose/docker-compose.yml down -v

migrate:
	docker compose -f infra/compose/docker-compose.yml exec api alembic upgrade head

seed:
	docker compose -f infra/compose/docker-compose.yml exec api python infra/scripts/seed_demo_data.py

logs:
	docker compose -f infra/compose/docker-compose.yml logs -f api worker

reset:
	rm -rf storage/raw/* storage/extracted/* storage/artifacts/* storage/reports/*
