"""Request and response models for the HTTP API."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    char_count: int
    created_at: str
    title: str | None = None
    image_only_pages: list[int] = []


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank")
        return value


class MultiAskRequest(AskRequest):
    doc_ids: list[Annotated[str, Field(pattern=r"^[0-9a-f]{12}$")]] = Field(..., min_length=1, max_length=10)


class FromBlobRequest(BaseModel):
    # Only the upload folder the frontend's token route allows, with a UUID file name.
    pathname: str = Field(..., pattern=r"^uploads/[0-9a-f-]{36}\.pdf$")
    filename: str = Field(..., min_length=1, max_length=255)


class SourceOut(BaseModel):
    source_id: str
    chunk_id: str
    page: int
    excerpt: str
    score: float
    doc_id: str
    filename: str


class AnswerOut(BaseModel):
    status: Literal["answered", "abstained", "retrieval_only"]
    answer: str
    sources: list[SourceOut]
    model: str | None
    retrieved: int


class HealthOut(BaseModel):
    status: Literal["ok"]
    llm_provider: str
    llm_model: str | None
    embedding_model: str
    embedding_model_loaded: bool
    documents: int
    max_upload_mb: int
    storage: Literal["local", "blob"]
    direct_upload: bool


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorOut(BaseModel):
    error: ErrorBody
