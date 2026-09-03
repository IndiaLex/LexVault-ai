"""Health check endpoint.

Provides a simple GET /ai/health endpoint for load balancers
and monitoring systems to verify the service is running.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/ai/health")
def health():
    """Return service health status."""
    return {"status": "ok"}
