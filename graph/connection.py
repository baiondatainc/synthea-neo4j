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
            cls._driver = cls._create_driver()
        return cls._driver

    @classmethod
    def _create_driver(cls) -> Driver:
        settings = get_settings()
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            max_connection_pool_size=50,
            connection_timeout=30,
            max_transaction_retry_time=30,
            keep_alive=True,
        )
        driver.verify_connectivity()
        logger.info("✅ Connected to Neo4j Aura")
        return driver

    @classmethod
    def _reset_driver(cls):
        """Close and recreate the driver — called on connection failure."""
        try:
            if cls._driver:
                cls._driver.close()
        except Exception:
            pass
        cls._driver = None
        cls._driver = cls._create_driver()

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
            try:
                session.close()
            except Exception:
                pass

    @classmethod
    def run_query(cls, query: str, parameters: dict = None, _retry: bool = True) -> list:
        try:
            with cls.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            # SSL EOF / SessionExpired — reconnect once and retry
            if _retry and any(
                k in str(e) for k in ("SessionExpired", "SSLEOFError", "EOF", "ServiceUnavailable")
            ):
                logger.warning(f"Neo4j connection lost, reconnecting... ({e})")
                cls._reset_driver()
                return cls.run_query(query, parameters, _retry=False)
            raise