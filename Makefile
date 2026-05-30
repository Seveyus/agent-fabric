dev-up:
	docker compose -f infra/compose/docker-compose.yml up -d --build

dev-down:
	docker compose -f infra/compose/docker-compose.yml down -v

db-wait:
	docker compose -f infra/compose/docker-compose.yml exec postgres sh -lc 'until pg_isready -U agentfabric -d agentfabric; do echo "waiting for postgres"; sleep 2; done'

migrate: db-wait
	docker compose -f infra/compose/docker-compose.yml exec api alembic upgrade head

seed:
	docker compose -f infra/compose/docker-compose.yml exec api python infra/scripts/seed_demo_data.py

logs:
	docker compose -f infra/compose/docker-compose.yml logs -f api worker

reset:
	rm -rf storage/raw/* storage/extracted/* storage/artifacts/* storage/reports/*
