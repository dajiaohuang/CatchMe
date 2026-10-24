from __future__ import annotations

from catchme.config import Config
from catchme.recorders import clipboard as clipboard_recorder
from catchme.recorders.clipboard import ClipboardRecorder


def test_clipboard_text_with_unpaired_surrogate_is_recorded(monkeypatch, tmp_path):
    text = "\ud800"
    monkeypatch.setattr(clipboard_recorder, "_read_clipboard_text", lambda: text)
    events = []
    recorder = ClipboardRecorder(Config(root=tmp_path))

    recorder.poll(events.append)

    assert events == [{"content": text, "type": "text/plain"}]
