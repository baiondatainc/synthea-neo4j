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

        # bolt:// for local Docker, neo4j+s:// for Aura cloud
        # Controlled entirely by NEO4J_URI in .env — no code change needed
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            max_connection_pool_size=50,
            connection_timeout=30,
            max_transaction_retry_time=30,
            keep_alive=True,
        )
        driver.verify_connectivity()
        logger.info(f"✅ Connected to Neo4j at {settings.neo4j_uri}")
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
        settings = get_settings()
        # NEO4J_DATABASE=neo4j works for both local and Aura
        db = getattr(settings, "neo4j_database", "neo4j")
        driver = cls.get_driver()
        session = driver.session(database=db)
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
            # Handles local Docker restarts + Aura SSL drops
            if _retry and any(
                k in str(e) for k in (
                    "SessionExpired", "SSLEOFError", "EOF",
                    "ServiceUnavailable", "ConnectionResetError",
                )
            ):
                logger.warning(f"Neo4j connection lost, reconnecting... ({e})")
                cls._reset_driver()
                return cls.run_query(query, parameters, _retry=False)
            raise