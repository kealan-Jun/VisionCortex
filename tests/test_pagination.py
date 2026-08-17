import pytest

from labvision_evidence.pagination import decode_cursor, encode_cursor


def test_opaque_cursor_round_trip_is_bound_to_namespace_and_filters():
    filters = {"archive": "Archive-001", "action_type": "liquid_transfer"}
    cursor = encode_cursor(
        namespace="key-events",
        position={"archive_id": "Archive-001", "peak_timestamp_us": 123, "event_uid": "evt"},
        filters=filters,
    )

    assert decode_cursor(cursor, namespace="key-events", filters=filters) == {
        "archive_id": "Archive-001",
        "event_uid": "evt",
        "peak_timestamp_us": 123,
    }

    with pytest.raises(ValueError, match="current filters"):
        decode_cursor(
            cursor,
            namespace="key-events",
            filters={**filters, "action_type": "object_movement"},
        )
    with pytest.raises(ValueError, match="another result set"):
        decode_cursor(cursor, namespace="physical-changes", filters=filters)


@pytest.mark.parametrize("value", ["not+a+cursor", "e30", "", "A" * 5000])
def test_opaque_cursor_rejects_malformed_payloads(value):
    with pytest.raises(ValueError):
        decode_cursor(value, namespace="key-events", filters={})
