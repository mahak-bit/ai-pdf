"""Domain errors that map to clear HTTP responses."""


class AppError(Exception):
    """An error with a user-facing message, a stable machine code and an HTTP status."""

    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code


class InvalidUpload(AppError):
    status_code = 400
    code = "invalid_upload"


class UnprocessableDocument(AppError):
    status_code = 422
    code = "unprocessable_document"


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class LLMError(AppError):
    status_code = 502
    code = "llm_unavailable"
