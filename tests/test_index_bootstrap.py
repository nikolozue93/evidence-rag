"""Unit test for index creation, including pre-existing Qdrant collections.

Does not need real cloud access. Execute with existing project dependencies installed:
    python -m pytest tests/test_index_bootstrap.py -q
"""
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.documents import CHUNKS_COLLECTION, REGISTRY_COLLECTION, create_collections


def test_create_collections_repairs_existing_indexes(monkeypatch):
    models = SimpleNamespace(PayloadSchemaType=SimpleNamespace(KEYWORD='keyword'))
    fake = types.ModuleType('qdrant_client')
    fake.models = models
    monkeypatch.setitem(sys.modules, 'qdrant_client', fake)

    db = MagicMock()
    db.collection_exists.return_value = True
    db.get_collection.return_value = SimpleNamespace(payload_schema={})

    create_collections(db)

    db.create_collection.assert_not_called()
    assert [c.kwargs['field_name'] for c in db.create_payload_index.call_args_list] == [
        'document_id', 'revision',
    ]
    for c in db.create_payload_index.call_args_list:
        assert c.kwargs['collection_name'] == CHUNKS_COLLECTION
        assert c.kwargs['field_schema'] == 'keyword'
        assert c.kwargs['wait'] is True


def test_create_collections_ignores_existing_indexes(monkeypatch):
    models = SimpleNamespace(PayloadSchemaType=SimpleNamespace(KEYWORD='keyword'))
    fake = types.ModuleType('qdrant_client')
    fake.models = models
    monkeypatch.setitem(sys.modules, 'qdrant_client', fake)
    db = MagicMock()
    db.collection_exists.return_value = True
    db.get_collection.return_value = SimpleNamespace(payload_schema={
        'document_id': {}, 'revision': {},
    })
    create_collections(db)
    db.create_payload_index.assert_not_called()
