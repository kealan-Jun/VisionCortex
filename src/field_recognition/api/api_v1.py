"""API versioning and unified error handling."""

from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorCode(str, Enum):
    """Standard error codes."""

    # Authentication
    AUTH_REQUIRED = "auth_required"
    AUTH_INVALID = "auth_invalid"
    AUTH_EXPIRED = "auth_expired"
    PERMISSION_DENIED = "permission_denied"

    # Validation
    INVALID_INPUT = "invalid_input"
    MISSING_FIELD = "missing_field"
    INVALID_FORMAT = "invalid_format"

    # Resource
    NOT_FOUND = "not_found"
    ALREADY_EXISTS = "already_exists"
    CONFLICT = "conflict"

    # State
    INVALID_STATE = "invalid_state"
    OPERATION_NOT_ALLOWED = "operation_not_allowed"

    # Quota
    RATE_LIMIT = "rate_limit"
    QUOTA_EXCEEDED = "quota_exceeded"
    QUEUE_FULL = "queue_full"

    # System
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    EXTERNAL_SERVICE_ERROR = "external_service_error"


class APIError(BaseModel):
    """Standard API error response."""

    error: ErrorCode
    message: str
    details: Optional[dict] = None
    request_id: Optional[str] = None


class APIResponse(BaseModel):
    """Standard API success response."""

    success: bool = True
    data: Optional[dict] = None
    request_id: Optional[str] = None


def setup_error_handlers(app: FastAPI):
    """Setup standard error handlers for the app."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTPException with standard format."""

        # Map HTTP status to error code
        code_map = {
            401: ErrorCode.AUTH_REQUIRED,
            403: ErrorCode.PERMISSION_DENIED,
            404: ErrorCode.NOT_FOUND,
            409: ErrorCode.CONFLICT,
            422: ErrorCode.INVALID_INPUT,
            429: ErrorCode.RATE_LIMIT,
            500: ErrorCode.INTERNAL_ERROR,
            503: ErrorCode.SERVICE_UNAVAILABLE,
        }

        error_code = code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        request_id = request.headers.get("x-request-id")

        return JSONResponse(
            status_code=exc.status_code,
            content=APIError(
                error=error_code, message=exc.detail, request_id=request_id
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        import traceback

        # Log the error
        print(f"Unhandled exception: {exc}")
        traceback.print_exc()

        request_id = request.headers.get("x-request-id")

        return JSONResponse(
            status_code=500,
            content=APIError(
                error=ErrorCode.INTERNAL_ERROR,
                message="An internal error occurred",
                request_id=request_id,
            ).model_dump(),
        )


def add_request_id_middleware(app: FastAPI):
    """Add request ID tracking middleware."""
    import uuid

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        # Generate or extract request ID
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        # Store in request state
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add to response headers
        response.headers["x-request-id"] = request_id

        return response


def setup_api_versioning(app: FastAPI):
    """Setup API versioning structure."""
    from fastapi import APIRouter

    # Create v1 router
    v1_router = APIRouter(prefix="/api/v1", tags=["v1"])

    # Health check at root (no version)
    @app.get("/health")
    async def health_check():
        """Health check endpoint (version-agnostic)."""
        return {"status": "healthy"}

    # Version info
    @v1_router.get("/version")
    async def get_version():
        """Get API version information."""
        return {"version": "v1", "api_version": "1.0.0", "service": "FieldRecognition"}

    return v1_router


class PaginationParams(BaseModel):
    """Standard pagination parameters."""

    limit: int = 20
    offset: int = 0
    cursor: Optional[str] = None


class PaginatedResponse(BaseModel):
    """Standard paginated response."""

    items: list
    total: Optional[int] = None
    limit: int
    offset: int
    next_cursor: Optional[str] = None
    has_more: bool
