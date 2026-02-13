from __future__ import annotations

from datetime import datetime, timezone

import pytest

from veriscope.core.calibration_recorder import CalibrationRecorder, CalibrationRecorderConfig

pytestmark = pytest.mark.unit


def test_calibration_recorder_extract_record_timestamp_is_timezone_aware_utc():
    recorder = CalibrationRecorder(CalibrationRecorderConfig(enabled=True, only_evaluated=False))
    try:
        record = recorder.extract_record(iter_num=1, audit={"evaluated": True}, combined_ok=True)
    finally:
        recorder.close()

    assert record is not None
    parsed = datetime.fromisoformat(record.timestamp_utc)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(parsed)

    round_trip = datetime.fromisoformat(parsed.isoformat())
    assert round_trip.tzinfo is not None
    assert round_trip.utcoffset() == timezone.utc.utcoffset(round_trip)
