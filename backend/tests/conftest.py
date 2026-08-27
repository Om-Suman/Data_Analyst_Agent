import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.session.state import SessionManager


@pytest.fixture
def client():
    # Use isolated test session
    test_session_id = "test-session"
    session = SessionManager().get_session(test_session_id)
    session.reset()
    client = TestClient(app, headers={"X-Session-ID": test_session_id})
    yield client
    session.reset()
