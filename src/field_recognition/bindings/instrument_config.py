"""Instrument type configuration and measurement definitions."""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from field_recognition.runtime.database import Database


class MeasurementDefinition(BaseModel):
    """Definition of a measurement type."""

    name: str = Field(min_length=1, max_length=50)
    unit: str = Field(min_length=1, max_length=20)
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    precision: float = Field(gt=0, default=0.1)
    display_format: Optional[str] = None  # e.g., ".2f"


class InstrumentType(BaseModel):
    """Instrument type with supported measurements."""

    id: str
    name: str
    measurements: list[MeasurementDefinition]
    created_at: str
    updated_at: str


class InstrumentConfig:
    """Manage instrument types and measurement definitions."""

    # Built-in measurement definitions (can be overridden in database)
    BUILTIN_MEASUREMENTS = {
        "温度": MeasurementDefinition(
            name="温度",
            unit="°C",
            range_min=None,
            range_max=None,
            precision=0.1,
            display_format=".1f",
        ),
        "转速": MeasurementDefinition(
            name="转速",
            unit="rpm",
            range_min=None,
            range_max=None,
            precision=1,
            display_format=".0f",
        ),
        "质量": MeasurementDefinition(
            name="质量",
            unit="g",
            range_min=None,
            range_max=None,
            precision=0.001,
            display_format=".3f",
        ),
    }

    def __init__(self, db: Database):
        self.db = db
        self._cache = {}
        self._init_builtin_types()

    def _init_builtin_types(self):
        """Initialize built-in instrument types if not exist."""
        with self.db.transaction("IMMEDIATE") as conn:
            # Check if types exist
            count = conn.execute("SELECT COUNT(*) FROM instrument_types").fetchone()[0]

            if count > 0:
                return

            # Create default types
            now = datetime.now(timezone.utc).isoformat()

            # Temperature/Speed instrument
            temp_speed_type = {
                "id": str(uuid.uuid4()),
                "name": "温度转速仪",
                "measurements": [
                    self.BUILTIN_MEASUREMENTS["温度"].model_dump(),
                    self.BUILTIN_MEASUREMENTS["转速"].model_dump(),
                ],
            }

            conn.execute(
                """
                INSERT INTO instrument_types VALUES(?,?,?,?,?)
            """,
                (
                    temp_speed_type["id"],
                    temp_speed_type["name"],
                    json.dumps(temp_speed_type["measurements"]),
                    now,
                    now,
                ),
            )

            # Mass measurement instrument
            mass_type = {
                "id": str(uuid.uuid4()),
                "name": "质量测量仪",
                "measurements": [self.BUILTIN_MEASUREMENTS["质量"].model_dump()],
            }

            conn.execute(
                """
                INSERT INTO instrument_types VALUES(?,?,?,?,?)
            """,
                (
                    mass_type["id"],
                    mass_type["name"],
                    json.dumps(mass_type["measurements"]),
                    now,
                    now,
                ),
            )

    def create_instrument_type(
        self, name: str, measurements: list[MeasurementDefinition]
    ) -> InstrumentType:
        """
        Create a new instrument type.

        Args:
            name: Type name
            measurements: Supported measurements

        Returns:
            Created instrument type
        """
        with self.db.transaction("IMMEDIATE") as conn:
            # Check uniqueness
            existing = conn.execute(
                "SELECT id FROM instrument_types WHERE name=?", (name,)
            ).fetchone()

            if existing:
                raise ValueError(f"仪器类型 {name} 已存在")

            now = datetime.now(timezone.utc).isoformat()
            type_id = str(uuid.uuid4())

            measurements_json = json.dumps([m.model_dump() for m in measurements])

            conn.execute(
                """
                INSERT INTO instrument_types VALUES(?,?,?,?,?)
            """,
                (type_id, name, measurements_json, now, now),
            )

            instrument_type = InstrumentType(
                id=type_id, name=name, measurements=measurements, created_at=now, updated_at=now
            )

            self._cache[type_id] = instrument_type
            return instrument_type

    def get_instrument_type(self, type_id: str) -> Optional[InstrumentType]:
        """Get instrument type by ID."""
        if type_id in self._cache:
            return self._cache[type_id]

        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM instrument_types WHERE id=?
            """,
                (type_id,),
            ).fetchone()

            if not row:
                return None

            measurements = [
                MeasurementDefinition(**m) for m in json.loads(row["measurements_json"])
            ]

            instrument_type = InstrumentType(
                id=row["id"],
                name=row["name"],
                measurements=measurements,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

            self._cache[type_id] = instrument_type
            return instrument_type

    def list_instrument_types(self) -> list[InstrumentType]:
        """List all instrument types."""
        with self.db.connection() as conn:
            rows = conn.execute("""
                SELECT * FROM instrument_types ORDER BY name
            """).fetchall()

            types = []
            for row in rows:
                measurements = [
                    MeasurementDefinition(**m) for m in json.loads(row["measurements_json"])
                ]

                types.append(
                    InstrumentType(
                        id=row["id"],
                        name=row["name"],
                        measurements=measurements,
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                )

            return types

    def update_instrument_type(
        self, type_id: str, measurements: list[MeasurementDefinition]
    ) -> InstrumentType:
        """
        Update instrument type measurements.

        Args:
            type_id: Type ID
            measurements: New measurement definitions

        Returns:
            Updated instrument type
        """
        with self.db.transaction("IMMEDIATE") as conn:
            row = conn.execute(
                """
                SELECT * FROM instrument_types WHERE id=?
            """,
                (type_id,),
            ).fetchone()

            if not row:
                raise ValueError(f"仪器类型 {type_id} 不存在")

            now = datetime.now(timezone.utc).isoformat()
            measurements_json = json.dumps([m.model_dump() for m in measurements])

            conn.execute(
                """
                UPDATE instrument_types
                SET measurements_json=?, updated_at=?
                WHERE id=?
            """,
                (measurements_json, now, type_id),
            )

            instrument_type = InstrumentType(
                id=type_id,
                name=row["name"],
                measurements=measurements,
                created_at=row["created_at"],
                updated_at=now,
            )

            self._cache[type_id] = instrument_type
            return instrument_type

    def assign_instrument_type(self, instrument_id: str, type_id: str):
        """
        Assign instrument type to an instrument.

        Args:
            instrument_id: Instrument ID
            type_id: Type ID

        Raises:
            ValueError: If type doesn't exist
        """
        # Verify type exists
        instrument_type = self.get_instrument_type(type_id)
        if not instrument_type:
            raise ValueError(f"仪器类型 {type_id} 不存在")

        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE instruments SET type_id=? WHERE id=?
            """,
                (type_id, instrument_id),
            )

    def get_measurement_definition(self, measurement_name: str) -> Optional[MeasurementDefinition]:
        """Get measurement definition by name (from built-ins or database)."""
        return self.BUILTIN_MEASUREMENTS.get(measurement_name)
