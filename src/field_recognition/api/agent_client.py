"""Small standard-library adapter usable from any Agent tool dispatcher."""

import json
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler


class FieldToolsError(RuntimeError):
    def __init__(self, status, detail):
        self.status = status
        self.detail = detail
        super().__init__(f"Field tool HTTP {status}: {detail}")


class FieldTools:
    def __init__(
        self, base_url="http://127.0.0.1:8188", timeout=20, *, session_token=None, camera_id=None
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))
        self.headers = {"Content-Type": "application/json"}
        if session_token:
            self.headers["Authorization"] = "Bearer " + session_token
        if camera_id:
            self.headers["X-Camera-Id"] = camera_id

    def _request(self, path, arguments=None):
        data = None if arguments is None else json.dumps(arguments).encode()
        request = Request(self.base_url + path, data=data, headers=self.headers)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            payload = json.loads(exc.read())
            detail = payload.get("detail") or payload.get("error", {}).get(
                "message", "Request failed"
            )
            raise FieldToolsError(exc.code, detail) from None

    def definitions(self):
        return self._request("/api/tools")["tools"]

    def call(self, name, arguments=None):
        # Never turn a model-generated name into an arbitrary URL path.
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz_" for c in name):
            raise ValueError("Invalid tool name")
        return self._request("/api/tools/" + name, arguments or {})["result"]

    def read_saved_panel(self, binding_id, image_path, crop=None):
        """Pass the exact path returned by the existing photo tool; never recapture."""
        arguments = {"binding_id": binding_id, "image_path": image_path}
        if crop is not None:
            arguments["crop"] = crop
        return self.call("read_saved_panel", arguments)
