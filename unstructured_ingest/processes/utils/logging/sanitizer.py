from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse


class DataSanitizer:
    """Utility class for sanitizing sensitive data in logs."""

    @staticmethod
    def sanitize_path(path: Union[str, Path]) -> str:
        """Sanitize file paths for logging, showing only filename and partial path."""
        if not path:
            return "<empty>"

        path_str = str(path)
        path_obj = Path(path_str)

        if len(path_obj.parts) > 2:
            return f".../{path_obj.parent.name}/{path_obj.name}"
        return path_obj.name

    @staticmethod
    def sanitize_id(identifier: str) -> str:
        """Sanitize IDs for logging, showing only first/last few characters."""
        if not identifier:
            return "<id>"
        if len(identifier) < 10:
            half_len = len(identifier) // 2
            return f"{identifier[:half_len]}..."
        return f"{identifier[:4]}...{identifier[-4:]}"

    @staticmethod
    def sanitize_url(url: str) -> str:
        """Sanitize URLs for logging, removing sensitive query parameters."""
        if not url:
            return "<url>"
        try:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        except (ValueError, TypeError):
            return "<url>"

    @staticmethod
    def sanitize_token(token: str) -> str:
        """Sanitize tokens and secrets for logging."""
        if not token:
            return "<token>"
        token_len = len(token)
        if token_len < 10:
            half_len = token_len // 2
            return f"{token[:half_len]}..."
        return f"{token[:4]}...{token[-4:]}"

    @staticmethod
    def sanitize_location(location: Union[str, Path]) -> str:
        """Sanitize document locations (file paths, URLs, database references) for logging."""
        if not location:
            return "<empty>"

        location_str = str(location)

        # Fast prefix check for URLs (4 most common prefixes)
        # Compose a single tuple at the global class level for less attr access overhead
        url_prefixes = ("http://", "https://", "ftp://", "ftps://")
        if location_str.startswith(url_prefixes):
            return DataSanitizer.sanitize_url(location_str)

        # Handle database-style references (table:id, collection/document, etc.)
        # Avoid double split: check for colon, then split
        # Only split if safe
        if ":" in location_str and not location_str.startswith("/"):
            idx = location_str.find(":")
            if idx != -1 and idx < len(location_str) - 1:
                table_name = location_str[:idx]
                record_id = location_str[idx + 1 :]
                return f"{table_name}:{DataSanitizer.sanitize_id(record_id)}"

        # Fallback to path
        return DataSanitizer.sanitize_path(location_str)

    @staticmethod
    def sanitize_document_id(document_id: str) -> str:
        """Sanitize document IDs for logging (alias for sanitize_id for clarity)."""
        return DataSanitizer.sanitize_id(document_id)

    @staticmethod
    def sanitize_dict(data: Dict[str, Any], sensitive_keys: Optional[set] = None) -> Dict[str, Any]:
        """Sanitize dictionary data for logging."""
        if sensitive_keys is None:
            sensitive_keys = {
                "password",
                "token",
                "secret",
                "key",
                "api_key",
                "access_token",
                "refresh_token",
                "client_secret",
                "private_key",
                "credentials",
            }

        # Preallocate output dict, use local vars for conditions
        sanitized = {}
        skeys = sensitive_keys  # Local alias for lookup speed
        for k, v in data.items():
            key_lower = k.lower()
            # Fast string containment with generator expression, short-circuit with `next()`
            is_sensitive = next((True for s in skeys if s in key_lower), False)

            if is_sensitive:
                sanitized[k] = DataSanitizer.sanitize_token(str(v))
            elif isinstance(v, dict):
                sanitized[k] = DataSanitizer.sanitize_dict(v, skeys)
            else:
                # Precompute boolean checks to avoid repeated work
                v_is_str = isinstance(v, str)
                v_is_path = isinstance(v, Path)
                # Cheaper: merge in one check to one block
                if (v_is_str or v_is_path) and (
                    "path" in key_lower
                    or "file" in key_lower
                    or "location" in key_lower
                    or "document_location" in key_lower
                ):
                    sanitized[k] = DataSanitizer.sanitize_location(v)
                elif v_is_str and (
                    ("id" in key_lower and len(v) > 8)
                    or ("document_id" in key_lower and len(v) > 8)
                ):
                    sanitized[k] = DataSanitizer.sanitize_document_id(v)
                else:
                    sanitized[k] = v
        return sanitized
