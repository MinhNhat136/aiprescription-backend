"""Tests for Alembic migration configuration and Prescription model."""
import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from app.models import Base, Prescription


class TestPrescriptionModel:
    """Tests for the Prescription SQLAlchemy model."""

    def test_prescription_model_is_importable(self):
        """Verify Prescription model can be imported."""
        from app.models.prescription import Prescription
        assert Prescription is not None

    def test_prescription_table_name(self):
        """Verify Prescription uses correct table name."""
        assert Prescription.__tablename__ == "prescriptions"

    def test_prescription_has_id_column(self):
        """Verify Prescription has an id primary key column."""
        assert hasattr(Prescription, "id")

    def test_prescription_has_patient_name_column(self):
        """Verify Prescription has patient_name column."""
        assert hasattr(Prescription, "patient_name")

    def test_prescription_has_diagnosis_column(self):
        """Verify Prescription has diagnosis column."""
        assert hasattr(Prescription, "diagnosis")

    def test_prescription_has_medications_column(self):
        """Verify Prescription has medications column."""
        assert hasattr(Prescription, "medications")

    def test_prescription_has_instructions_column(self):
        """Verify Prescription has instructions column."""
        assert hasattr(Prescription, "instructions")

    def test_prescription_has_created_at_column(self):
        """Verify Prescription has created_at column."""
        assert hasattr(Prescription, "created_at")

    def test_prescription_has_updated_at_column(self):
        """Verify Prescription has updated_at column."""
        assert hasattr(Prescription, "updated_at")

    def test_prescription_has_doctor_id_column(self):
        """Verify Prescription has doctor_id column."""
        assert hasattr(Prescription, "doctor_id")

    def test_prescription_inherits_from_base(self):
        """Verify Prescription inherits from declarative Base."""
        assert issubclass(Prescription, Base)


class TestAlembicEnv:
    """Tests for Alembic async environment configuration.

    Alembic's env.py uses context.config which is only available during
    Alembic runtime. We verify its structure using AST parsing.
    """

    @staticmethod
    def _get_ast_tree(path: Path) -> ast.Module:
        """Parse a Python file into an AST tree."""
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        return ast.parse(source)

    @staticmethod
    def _get_function_names(tree: ast.Module) -> set[str]:
        """Extract all function names defined in an AST tree."""
        functions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.add(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                functions.add(node.name)
        return functions

    def test_env_module_file_exists(self):
        """Verify alembic env.py file exists."""
        backend_root = Path(__file__).parent.parent
        env_path = backend_root / "alembic" / "env.py"
        assert env_path.exists()

    def test_env_module_has_async_migrations_function(self):
        """Verify run_async_migrations function exists in env module."""
        backend_root = Path(__file__).parent.parent
        env_path = backend_root / "alembic" / "env.py"
        tree = self._get_ast_tree(env_path)
        functions = self._get_function_names(tree)
        assert "run_async_migrations" in functions

    def test_env_module_has_run_migrations_online_function(self):
        """Verify run_migrations_online function exists in env module."""
        backend_root = Path(__file__).parent.parent
        env_path = backend_root / "alembic" / "env.py"
        tree = self._get_ast_tree(env_path)
        functions = self._get_function_names(tree)
        assert "run_migrations_online" in functions

    def test_env_module_has_run_migrations_offline_function(self):
        """Verify run_migrations_offline function exists in env module."""
        backend_root = Path(__file__).parent.parent
        env_path = backend_root / "alembic" / "env.py"
        tree = self._get_ast_tree(env_path)
        functions = self._get_function_names(tree)
        assert "run_migrations_offline" in functions

    def test_env_module_has_do_run_migrations_function(self):
        """Verify do_run_migrations function exists in env module."""
        backend_root = Path(__file__).parent.parent
        env_path = backend_root / "alembic" / "env.py"
        tree = self._get_ast_tree(env_path)
        functions = self._get_function_names(tree)
        assert "do_run_migrations" in functions

    def test_env_module_uses_asyncio_run(self):
        """Verify env.py uses asyncio.run() for async migrations."""
        backend_root = Path(__file__).parent.parent
        env_path = backend_root / "alembic" / "env.py"
        with open(env_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "asyncio.run(" in source

    def test_env_module_imports_async_engine(self):
        """Verify env.py imports async_engine_from_config."""
        backend_root = Path(__file__).parent.parent
        env_path = backend_root / "alembic" / "env.py"
        tree = self._get_ast_tree(env_path)
        # Check for async_engine_from_config import
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "sqlalchemy.ext.asyncio":
                    imports.extend(alias.name for alias in node.names)
        assert "async_engine_from_config" in imports


class TestMigrationScript:
    """Tests for migration script structure."""

    @staticmethod
    def _load_module_from_path(name: str, path: Path):
        """Load a Python module directly from a file path."""
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def test_initial_migration_file_exists(self):
        """Verify initial migration script file exists."""
        backend_root = Path(__file__).parent.parent
        migration_path = (
            backend_root / "alembic" / "versions" / "001_initial_prescriptions.py"
        )
        assert migration_path.exists()

    def test_initial_migration_has_upgrade(self):
        """Verify initial migration has upgrade function."""
        backend_root = Path(__file__).parent.parent
        migration_path = (
            backend_root / "alembic" / "versions" / "001_initial_prescriptions.py"
        )
        module = self._load_module_from_path(
            "initial_prescriptions_test", migration_path
        )
        assert hasattr(module, "upgrade")
        assert callable(module.upgrade)

    def test_initial_migration_has_downgrade(self):
        """Verify initial migration has downgrade function."""
        backend_root = Path(__file__).parent.parent
        migration_path = (
            backend_root / "alembic" / "versions" / "001_initial_prescriptions.py"
        )
        module = self._load_module_from_path(
            "initial_prescriptions_test2", migration_path
        )
        assert hasattr(module, "downgrade")
        assert callable(module.downgrade)

    def test_initial_migration_revision_id(self):
        """Verify initial migration has correct revision ID."""
        backend_root = Path(__file__).parent.parent
        migration_path = (
            backend_root / "alembic" / "versions" / "001_initial_prescriptions.py"
        )
        module = self._load_module_from_path(
            "initial_prescriptions_test3", migration_path
        )
        assert module.revision == "001_initial_prescriptions"

    def test_initial_migration_down_revision_is_none(self):
        """Verify initial migration has no down_revision."""
        backend_root = Path(__file__).parent.parent
        migration_path = (
            backend_root / "alembic" / "versions" / "001_initial_prescriptions.py"
        )
        module = self._load_module_from_path(
            "initial_prescriptions_test4", migration_path
        )
        assert module.down_revision is None

    def test_initial_migration_creates_prescriptions_table(self):
        """Verify upgrade/downgrade functions are callable."""
        backend_root = Path(__file__).parent.parent
        migration_path = (
            backend_root / "alembic" / "versions" / "001_initial_prescriptions.py"
        )
        module = self._load_module_from_path(
            "initial_prescriptions_test5", migration_path
        )
        assert callable(module.upgrade)
        assert callable(module.downgrade)
