from main import TrainingController, MAX_LOG_TAIL_ENTRIES


def test_append_log_truncates_to_max_entries():
    job = {}
    total_messages = MAX_LOG_TAIL_ENTRIES + 37

    for index in range(total_messages):
        TrainingController._append_log(job, f"message {index}")

    assert "logTail" in job
    log_tail = job["logTail"]
    assert len(log_tail) == MAX_LOG_TAIL_ENTRIES

    expected_messages = [
        f"message {index}" for index in range(total_messages - MAX_LOG_TAIL_ENTRIES, total_messages)
    ]
    assert log_tail == expected_messages
