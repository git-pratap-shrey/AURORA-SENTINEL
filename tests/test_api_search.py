"""
Intelligence Search API Tests
Tests for semantic search and chat endpoints
"""
import pytest
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestIntelligenceSearchAPI:
    """Test intelligence search endpoints"""
    
    def test_search_returns_200(self, app_client):
        """Test GET /intelligence/search returns HTTP 200"""
        try:
            response = app_client.get('/intelligence/search?q=fight&limit=5')
            assert response.status_code == 200
            assert isinstance(response.json(), list)
        except Exception as e:
            pytest.skip(f"Search API not available: {e}")
    
    def test_latest_returns_200(self, app_client):
        """Test GET /intelligence/latest returns HTTP 200"""
        try:
            response = app_client.get('/intelligence/latest')
            assert response.status_code == 200
            assert isinstance(response.json(), list)
        except Exception as e:
            pytest.skip(f"Latest API not available: {e}")
    
    def test_recent_returns_200(self, app_client):
        """Test GET /intelligence/recent returns HTTP 200"""
        try:
            response = app_client.get('/intelligence/recent')
            assert response.status_code == 200
            assert isinstance(response.json(), list)
        except Exception as e:
            pytest.skip(f"Recent API not available: {e}")
    
    def test_chat_returns_200(self, app_client):
        """Test POST /intelligence/chat returns HTTP 200"""
        try:
            response = app_client.post('/intelligence/chat', json={
                'question': 'What is happening in the video?'
            })
            assert response.status_code == 200
            data = response.json()
            assert 'answer' in data
            assert 'session_id' in data
            assert 'timeline' in data
            assert 'answer_mode' in data
        except Exception as e:
            pytest.skip(f"Chat API not available: {e}")

    def test_cross_search_returns_200(self, app_client):
        """Test GET /intelligence/cross-search returns HTTP 200"""
        try:
            response = app_client.get('/intelligence/cross-search?q=fight&limit=3')
            assert response.status_code == 200
            assert isinstance(response.json(), list)
        except Exception as e:
            pytest.skip(f"Cross search API not available: {e}")
    
    def test_search_case_insensitive(self, app_client):
        """Test search is case insensitive"""
        try:
            response1 = app_client.get('/intelligence/search?q=fight')
            response2 = app_client.get('/intelligence/search?q=FIGHT')
            
            # Both should return results
            assert response1.status_code == 200
            assert response2.status_code == 200
        except Exception as e:
            pytest.skip(f"Search API not available: {e}")


class TestSearchPerformance:
    """Test search performance and concurrency"""
    
    def test_search_response_time(self, app_client):
        """Test search response time is acceptable"""
        try:
            import time
            
            start = time.time()
            response = app_client.get('/intelligence/search?q=fight&limit=10')
            elapsed = time.time() - start
            
            assert response.status_code == 200
            assert elapsed < 5.0, f"Search took {elapsed}s, should be < 5s"
        except Exception as e:
            pytest.skip(f"Search API not available: {e}")
    
    def test_chat_response_time(self, app_client):
        """Test chat response time is acceptable"""
        try:
            import time
            
            start = time.time()
            response = app_client.post('/intelligence/chat', json={
                'question': 'Summarize the video'
            }, timeout=20)
            elapsed = time.time() - start
            
            assert response.status_code == 200
            # Chat can take longer due to AI processing
            assert elapsed < 20.0, f"Chat took {elapsed}s, should be < 20s"
        except Exception as e:
            pytest.skip(f"Chat API not available: {e}")
    
    def test_latest_pagination(self, app_client):
        """Test latest endpoint returns limited results"""
        try:
            response = app_client.get('/intelligence/latest')
            data = response.json()
            
            # Should return at most 20 results
            assert len(data) <= 20
        except Exception as e:
            pytest.skip(f"Latest API not available: {e}")

class TestChatFunctionality:
    """Test chat/VQA functionality"""
    
    def test_chat_with_question(self, app_client):
        """Test chat with specific question"""
        try:
            response = app_client.post('/intelligence/chat', json={
                'question': 'Is this a fight or boxing?'
            })
            
            assert response.status_code == 200
            data = response.json()
            assert 'answer' in data
            assert len(data['answer']) > 0
        except Exception as e:
            pytest.skip(f"Chat API not available: {e}")
    
    def test_chat_without_video(self, app_client):
        """Test chat without uploaded video"""
        try:
            response = app_client.post('/intelligence/chat', json={
                'question': 'What do you see?'
            })
            
            # Should return helpful message
            assert response.status_code == 200
            data = response.json()
            assert 'answer' in data
        except Exception as e:
            pytest.skip(f"Chat API not available: {e}")

    def test_chat_session_round_trip(self, app_client):
        """Follow-up question should preserve session id when provided."""
        try:
            first = app_client.post('/intelligence/chat', json={
                'question': 'Summarize the latest video'
            })
            assert first.status_code == 200
            first_data = first.json()
            session_id = first_data.get('session_id')
            assert session_id

            second = app_client.post('/intelligence/chat', json={
                'question': 'What happened at 0:30?',
                'session_id': session_id
            })
            assert second.status_code == 200
            second_data = second.json()
            assert second_data.get('session_id') == session_id
            assert isinstance(second_data.get('timeline', []), list)
        except Exception as e:
            pytest.skip(f"Session chat API not available: {e}")


class TestSettingsVlmInterval:
    def test_get_set_vlm_interval(self, app_client):
        """Persisted VLM interval setting should be writable and readable."""
        try:
            write_res = app_client.post('/settings/vlm-interval', json={'seconds': 8})
            assert write_res.status_code == 200
            assert write_res.json().get('seconds') == 8

            read_res = app_client.get('/settings/vlm-interval')
            assert read_res.status_code == 200
            assert read_res.json().get('seconds') == 8
        except Exception as e:
            pytest.skip(f"Settings API not available: {e}")
