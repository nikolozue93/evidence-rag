"""Offline tests for intermittent provider tool-call failures and answer synthesis."""
from types import SimpleNamespace as NS
import json
from src.investigation import investigate


def _reply(content=None, tool_calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))])


def _call(name='search_evidence', query='Arthur'):
    return NS(id='test-call', function=NS(name=name, arguments=json.dumps({'query': query})))


def _search(query):
    return [{'id': 'source1', 'payload': {'source_id': 'source1', 'title': 'Arthur',
            'text': 'Arthur was first published in 1844.'}}]


class RecoveringGroq:
    def __init__(self, tool_failure_mode='once', final_failure=False):
        self.chat = NS(completions=NS(create=self.create))
        self.tool_failure_mode = tool_failure_mode
        self.final_failure = final_failure
        self.tool_attempts = 0
        self.final_attempts = 0

    def create(self, **kwargs):
        if kwargs.get('tools'):
            self.tool_attempts += 1
            if self.tool_failure_mode == 'always' or (
                    self.tool_failure_mode == 'once' and self.tool_attempts == 1):
                raise Exception('BadRequestError: 400 tool_use_failed: Tool choice is required, but model did not call a tool')
            return _reply(tool_calls=[_call(query='Arthur first published')])
        self.final_attempts += 1
        if self.final_failure and self.final_attempts == 1:
            raise Exception('BadRequestError: 400 tool_use_failed: Tool choice is none, but model called a tool')
        # Fresh tool-free messages, not prior tool-call conversation.
        assert [m['role'] for m in kwargs['messages']] == ['system', 'user']
        assert 'indexed source evidence' in kwargs['messages'][1]['content'].lower()
        return _reply(content='Arthur was first published in 1844 [1].')


def test_recover_from_required_tool_error_and_tool_free_final():
    client = RecoveringGroq(tool_failure_mode='once')
    out = investigate('When was Arthur first published?', None, client,
                      search_fn=_search, search_rounds=2)
    assert out['status'] == 'cited'
    assert out['structured_tool_calls'] == 2
    assert out['fallback_searches'] == 0
    assert client.tool_attempts == 3
    assert client.final_attempts == 1


def test_labeled_fallback_when_provider_refuses_all_tools():
    client = RecoveringGroq(tool_failure_mode='always')
    out = investigate('When was Arthur first published?', None, client,
                      search_fn=_search, search_rounds=2)
    assert out['structured_tool_calls'] == 0
    assert out['fallback_searches'] == 2
    assert out['status'] == 'cited'
    assert client.tool_attempts == 6


def test_retry_tool_free_final_once():
    client = RecoveringGroq(tool_failure_mode='never', final_failure=True)
    out = investigate('When was Arthur first published?', None, client,
                      search_fn=_search, search_rounds=1)
    assert out['status'] == 'cited'
    assert client.final_attempts == 2


def test_no_evidence_skips_final_request():
    client = RecoveringGroq(tool_failure_mode='never')
    out = investigate('Unknown?', None, client,
                      search_fn=lambda query: [], search_rounds=1)
    assert out['status'] == 'no_evidence'
    assert 'Insufficient evidence' in out['answer']
    assert client.final_attempts == 0
