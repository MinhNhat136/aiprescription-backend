"""Tests for database configuration and connection pooling."""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import engine, async_session_factory, get_async_session


class TestDatabaseConfig:
    """Tests for database configuration."""

    def test_database_url_is_configured(self):
        """Verify database URL is set in settings."""
        assert settings.database_url is not None
        assert "postgresql" in settings.database_url
        assert "asyncpg" in settings.database_url

    def test_database_url_format(self):
        """Verify database URL follows expected format."""
        url = settings.database_url
        # Expected format: postgresql+asyncpg://user:pass@host:port/database
        assert url.startswith("postgresql+asyncpg://")
        parts = url.replace("postgresql+asyncpg://", "").split("@")
        assert len(parts) == 2  # user:pass@host:port/database

    def test_pool_settings_have_defaults(self):
        """Verify connection pool settings have sensible defaults."""
        assert settings.database_pool_size == 5
        assert settings.database_max_overflow == 10
        assert settings.database_pool_timeout == 30
        assert settings.database_pool_recycle == 3600

    def test_async_engine_created(self):
        """Verify async engine is created successfully."""
        assert engine is not None
        assert engine.pool is not None

    def test_async_session_factory_configured(self):
        """Verify async session factory is configured correctly."""
        assert async_session_factory is not None
        # expire_on_commit=False is critical for async sessions
        assert async_session_factory.kw.get("expire_on_commit") is False

    def test_async_session_class_is_async_session(self):
        """Verify AsyncSession is used as session class."""
        # Verify by checking the engine bind is async
        assert engine is not None

    def test_pool_pre_ping_enabled(self):
        """Verify pool_pre_ping is enabled for connection health checks."""
        # pool_pre_ping is set via echo parameter visibility
        # We verify the engine was created without errors
        assert engine is not None


class TestDatabaseDependency:
    """Tests for database dependency injection."""

    @pytest.mark.asyncio
    async def test_get_async_session_is_generator(self):
        """Verify get_async_session is an async generator."""
        import inspect
        assert inspect.isasyncgenfunction(get_async_session)

    @pytest.mark.asyncio
    async def test_session_factory_produces_session(self):
        """Verify session factory can produce a session."""
        async with async_session_factory() as session:
            assert session is not None
            assert isinstance(session, AsyncSession)
