from neo4j import GraphDatabase, Driver
from contextlib import contextmanager
from typing import Generator
import logging

from config import get_settings

logger = logging.getLogger(__name__)


class Neo4jConnection:
    _driver: Driver | None = None

    @classmethod
    def get_driver(cls) -> Driver:
        if cls._driver is None:
            settings = get_settings()
            cls._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
                max_connection_pool_size=50,
            )
            cls._driver.verify_connectivity()
            logger.info("✅ Connected to Neo4j Aura")
        return cls._driver

    @classmethod
    def close(cls):
        if cls._driver:
            cls._driver.close()
            cls._driver = None
            logger.info("Neo4j connection closed")

    @classmethod
    @contextmanager
    def session(cls) -> Generator:
        driver = cls.get_driver()
        session = driver.session(database="neo4j")
        try:
            yield session
        finally:
            session.close()

    @classmethod
    def run_query(cls, query: str, parameters: dict = None) -> list:
        with cls.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
