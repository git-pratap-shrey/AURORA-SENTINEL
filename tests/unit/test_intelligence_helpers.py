import json
import os
import tempfile

from backend.api.routers.intelligence import _extract_timestamp_hint
from backend.services.chat_session_store import InMemoryChatSessionStore
from backend.services.search_service import SearchService


def test_extract_timestamp_hint_mmss():
    assert _extract_timestamp_hint("what happened at 02:15?") == 135.0


def test_extract_timestamp_hint_minutes_seconds():
    assert _extract_timestamp_hint("show me 3 minutes 10 seconds mark") == 190.0


def test_extract_timestamp_hint_beginning_end():
    assert _extract_timestamp_hint("what happened at beginning") == 0.0
    assert _extract_timestamp_hint("what happened at the end", duration_seconds=101.0) == 100.0


def test_chat_session_store_ttl_and_bounds():
    store = InMemoryChatSessionStore(ttl_seconds=1, max_turns=3)
    sid = store.get_or_create_session_id()
    store.append_turn(sid, "user", "u1")
    store.append_turn(sid, "assistant", "a1")
    store.append_turn(sid, "user", "u2")
    store.append_turn(sid, "assistant", "a2")
    history = store.get_history(sid)
    assert len(history) == 3
    assert history[0]["content"] == "a1"

    # Force-expire
    store._sessions[sid].updated_at -= 100
    assert store.get_history(sid) == []


def test_search_service_timeline_and_cross_video_fallback():
    payload = [
        {
            "id": "vid_1",
            "filename": "a.mp4",
            "processed_at": "2026-03-26T10:00:00",
            "video_summary": {"text": "Two people arguing near a gate."},
            "events": [
                {
                    "timestamp": 10.0,
                    "description": "Two people start arguing loudly.",
                    "severity": "medium",
                    "threats": ["arguing"],
                    "provider": "ollama",
                    "confidence": 0.7,
                },
                {
                    "timestamp": 20.0,
                    "description": "One person shoves another.",
                    "severity": "high",
                    "threats": ["fight"],
                    "provider": "ollama",
                    "confidence": 0.9,
                },
            ],
        },
        {
            "id": "vid_2",
            "filename": "b.mp4",
            "processed_at": "2026-03-26T10:05:00",
            "video_summary": {"text": "Normal pedestrian traffic."},
            "events": [
                {
                    "timestamp": 5.0,
                    "description": "People are walking normally.",
                    "severity": "low",
                    "threats": [],
                    "provider": "ollama",
                    "confidence": 0.8,
                }
            ],
        },
    ]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        tmp_path = f.name

    old_metadata = os.environ.get("METADATA_PATH")
    os.environ["METADATA_PATH"] = tmp_path
    try:
        service = SearchService(persistence_path="storage/vectordb")
        service._vector_enabled = False
        service.collection = None
        service._model = None

        filename, timeline = service.timeline_search(
            query="what happened at 20 seconds",
            filename="a.mp4",
            target_timestamp=20.0,
            limit=3,
        )
        assert filename == "a.mp4"
        assert timeline
        assert timeline[0]["timestamp"] in (20.0, 10.0)

        grouped = service.cross_video_search(query="shoves", limit=2)
        assert isinstance(grouped, list)
        assert grouped
        assert "filename" in grouped[0]
    finally:
        if old_metadata is None:
            os.environ.pop("METADATA_PATH", None)
        else:
            os.environ["METADATA_PATH"] = old_metadata
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
