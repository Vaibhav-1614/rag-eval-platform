from rag_service.store import FeedbackStore


def test_daily_cap(tmp_path):
    s = FeedbackStore(tmp_path / "f.db")
    assert [s.take_llm_call(2) for _ in range(3)] == [True, True, False]
    assert s.llm_calls_today() == 2
    assert s.add_feedback("abc12345", -1, None) == 1
