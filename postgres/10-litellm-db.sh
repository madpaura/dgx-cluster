#!/bin/bash
# LiteLLM manages its schema with Prisma, which owns the whole `public` schema
# of whatever database it is pointed at: on first migration it drops the tables
# it does not know about. Sharing a database with dgxctl therefore destroys the
# node inventory, deployments, clusters and audit log — quietly, minutes after
# a successful-looking start.
#
# So LiteLLM gets its own database in the same Postgres instance.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE litellm OWNER $POSTGRES_USER;
EOSQL
echo "created database 'litellm' (kept separate from '$POSTGRES_DB')"
