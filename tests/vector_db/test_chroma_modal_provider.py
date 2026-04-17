import httpx
import pytest
from http import HTTPStatus
from langchain_core.documents import Document
from unittest.mock import patch

from alpha_crunch.vector_db.chroma_modal_provider import ChromaModalProvider

RealClient = httpx.Client
def transport_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(HTTPStatus.OK, json={"status": "ok"})
    elif request.url.path == "/ready":
        return httpx.Response(HTTPStatus.OK, json={"ready": True})
    elif request.url.path == "/query":
        return httpx.Response(HTTPStatus.OK, json=[{"page_content": "test", "metadata": {"test": "test"}}])
    else:
        return httpx.Response(HTTPStatus.NOT_FOUND, json={"error": "Not Found"})

def transport_handler_error(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(HTTPStatus.OK, json={"status": "ok"})
    elif request.url.path == "/ready":
        return httpx.Response(HTTPStatus.OK, json={"ready": True})
    elif request.url.path == "/query":
        return httpx.Response(HTTPStatus.INTERNAL_SERVER_ERROR, json={"error": "Internal Server Error"})
    else:
        return httpx.Response(HTTPStatus.NOT_FOUND, json={"error": "Not Found"})

def test_chroma_modal_provider():
    transport = httpx.MockTransport(transport_handler)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return RealClient(*args, **kwargs)

    with patch(
        "alpha_crunch.vector_db.chroma_modal_provider.httpx.Client",
        side_effect=client_factory,
    ):
        provider = ChromaModalProvider( modal_key="test_key", modal_secret="test_secret", vector_db_url="http://test.com", timeout= 30.0)

        # check ready and health
        assert provider.ready is True
        assert provider.health is True

        # check search
        response = provider.search("test")
        assert response is not None and isinstance(response, list)
        assert len(response) == 1
        assert response[0] is not None and isinstance(response[0], Document)

        assert response[0].page_content == "test"
        assert response[0].metadata == {"test": "test"}

def test_chroma_modal_provider_error():
    transport = httpx.MockTransport(transport_handler_error)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return RealClient(*args, **kwargs)

    with patch(
        "alpha_crunch.vector_db.chroma_modal_provider.httpx.Client",
        side_effect=client_factory,
    ):
        provider = ChromaModalProvider( modal_key="test_key", modal_secret="test_secret", vector_db_url="http://test.com", timeout= 30.0)

        with pytest.raises(httpx.HTTPStatusError):
            provider.search("test")