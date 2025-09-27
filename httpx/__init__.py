from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

Header = Tuple[bytes, bytes]


@dataclass
class Response:
    status_code: int
    _headers: List[Header]
    _content: bytes

    def json(self) -> object:
        if not self._content:
            return None
        return json.loads(self._content.decode("utf-8"))

    @property
    def content(self) -> bytes:
        return self._content

    @property
    def text(self) -> str:
        return self._content.decode("utf-8")

    @property
    def headers(self) -> Mapping[str, str]:
        return {key.decode("latin-1"): value.decode("latin-1") for key, value in self._headers}


class ASGITransport:
    def __init__(self, app) -> None:  # type: ignore[no-untyped-def]
        self.app = app

    async def handle_async_request(
        self,
        method: str,
        url: str,
        *,
        body: bytes = b"",
        headers: Optional[Iterable[Header]] = None,
    ) -> Response:
        path, query = _split_url(url)
        header_list: List[Header] = []
        if headers:
            header_list.extend((key.lower(), value) for key, value in headers)
        if body:
            header_list.append((b"content-length", str(len(body)).encode("latin-1")))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method.upper(),
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("latin-1"),
            "query_string": query.encode("latin-1"),
            "headers": header_list,
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        receive_messages = [
            {"type": "http.request", "body": body, "more_body": False},
        ]
        receive_index = 0

        async def receive() -> dict:
            nonlocal receive_index
            if receive_index < len(receive_messages):
                message = receive_messages[receive_index]
                receive_index += 1
                return message
            await asyncio.sleep(0)
            return {"type": "http.disconnect"}

        sent_messages: List[dict] = []

        async def send(message: dict) -> None:
            sent_messages.append(message)

        await self.app(scope, receive, send)

        status_code = 500
        response_headers: List[Header] = []
        content = b""
        for message in sent_messages:
            message_type = message.get("type")
            if message_type == "http.response.start":
                status_code = message.get("status", 500)
                response_headers = list(message.get("headers", []))
            elif message_type == "http.response.body":
                content += message.get("body", b"")

        return Response(status_code=status_code, _headers=response_headers, _content=content)


class AsyncClient:
    def __init__(self, transport: ASGITransport, base_url: str = "http://testserver") -> None:
        self._transport = transport
        self.base_url = base_url

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        return None

    async def request(
        self,
        method: str,
        url: str,
        *,
        json: Optional[object] = None,
        files: Optional[Mapping[str, Tuple[str, bytes, Optional[str]]]] = None,
    ) -> Response:
        body: bytes = b""
        headers: List[Header] = []
        if json is not None:
            body = json_dumps(json)
            headers.append((b"content-type", b"application/json"))
        elif files is not None:
            body, boundary = encode_multipart(files)
            headers.append((b"content-type", f"multipart/form-data; boundary={boundary}".encode("latin-1")))
        return await self._transport.handle_async_request(method, url, body=body, headers=headers)

    async def get(self, url: str, **kwargs) -> Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> Response:
        return await self.request("POST", url, **kwargs)


def json_dumps(data: object) -> bytes:
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def encode_multipart(files: Mapping[str, Tuple[str, bytes, Optional[str]]]) -> Tuple[bytes, str]:
    boundary = "----pytestboundary"
    parts: List[bytes] = []
    for field, value in files.items():
        filename, content, content_type = value
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content
        content_type = content_type or "application/octet-stream"
        parts.extend(
            [
                f"--{boundary}\r\n".encode("latin-1"),
                f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode("latin-1"),
                f"Content-Type: {content_type}\r\n\r\n".encode("latin-1"),
                content_bytes,
                b"\r\n",
            ]
        )
    parts.append(f"--{boundary}--\r\n".encode("latin-1"))
    return b"".join(parts), boundary


def _split_url(url: str) -> Tuple[str, str]:
    if url.startswith("http://") or url.startswith("https://"):
        parsed = urlsplit(url)
        return parsed.path or "/", parsed.query or ""
    if "?" in url:
        path, query = url.split("?", 1)
        return path or "/", query
    return url or "/", ""
