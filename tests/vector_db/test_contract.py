# TODO: add tests for the failure contract
import pytest
from typing import Any
from langchain_core.documents import Document

from alpha_crunch.vector_db.factory import get_vector_db_provider
from alpha_crunch.vector_db.types import VectorDBProvider
from alpha_crunch.vector_db.contracts import VectorSearch

@pytest.mark.parametrize("vector_db_provider", [VectorDBProvider.CHROMA])
def test_search_returns_list_of_documents(vector_db_provider: VectorDBProvider):
    provider = get_vector_db_provider(vector_db_provider)

    result = provider.search("test", k=1, filter=None)
    assert result is not None and isinstance(result, list) and len(result) <= 1
    
    item = result[0]
    assert item is not None and isinstance(item, Document)
    assert item.page_content is not None and isinstance(item.page_content, str)
    assert item.metadata is not None and isinstance(item.metadata, dict)

@pytest.mark.parametrize("vector_db_provider", [VectorDBProvider.CHROMA])
@pytest.mark.parametrize("k", [1,2,3])
def test_search_respects_k(vector_db_provider: VectorDBProvider, k: int):
    provider = get_vector_db_provider(vector_db_provider)

    result = provider.search("test", k=k, filter=None)
    assert result is not None and isinstance(result, list) and len(result) <= k

@pytest.mark.parametrize("vector_db_provider", [VectorDBProvider.CHROMA])
@pytest.mark.parametrize("filter", [{"company": "APPLE"}, {"item_type": "item_1"}])
def test_search_respects_filter(vector_db_provider: VectorDBProvider, filter: dict[str, Any]):
    provider = get_vector_db_provider(vector_db_provider)
    result = provider.search("test", k=1, filter=filter)

    # currently only one filter is supported
    assert result[0].metadata is not None and isinstance(result[0].metadata, dict)
    key,value = list(filter.items())[0]
    
    assert key in result[0].metadata.keys()
    assert result[0].metadata[key] == value

    
