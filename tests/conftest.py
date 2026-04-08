"""Pytest configuration and fixtures for FastAPI testing."""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app

# Use SQLite in-memory for tests (no external database required)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def anyio_backend():
    """Specify the async backend to use for async tests."""
    return "asyncio"


@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Create an async engine for testing.

    Uses SQLite in-memory database for fast, isolated tests.
    Each test function gets a fresh database.
    """
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def async_session(async_engine):
    """Create an async session for testing database operations.

    Provides a transactional rollback after each test for test isolation.
    """
    async with async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    ) as session:

        # Begin a nested transaction
        async with session.begin():
            yield session
            # Transaction rolled back automatically on context exit


@pytest.fixture
async def client():
    """Async client fixture for testing FastAPI endpoints."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac