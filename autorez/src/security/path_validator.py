import os
from pathlib import Path

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".mxf", ".prores"}

def validate_input_path(input_path: str) -> Path:
    """Validate and normalize an input media path.

    Rules:
    - Must exist and be a regular file
    - Extension must be in ALLOWED_EXTENSIONS
    - Resolve symlinks; final real path must be within one of the allowed roots (if set)
    - Prevent path traversal by requiring absolute, normalized paths
    Allowed roots are provided via ALLOWED_MEDIA_ROOTS env, comma-separated.
    """
    if not input_path or not isinstance(input_path, str):
        raise ValueError("Path must be a non-empty string")

    # Normalize and resolve symlinks
    p = Path(input_path).expanduser().resolve(strict=False)

    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    # Extension check
    ext = p.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Disallowed file type: {ext}")

    # Allowed roots policy
    roots_env = os.getenv("ALLOWED_MEDIA_ROOTS", "").strip()
    if roots_env:
        allowed_roots = [Path(r).expanduser().resolve() for r in roots_env.split(",") if r.strip()]
        if not any(is_within_root(p, root) for root in allowed_roots):
            raise PermissionError("Path outside allowed roots")

    # Resolve real path (follow symlinks) and return
    return p.resolve(strict=True)

def is_within_root(path: Path, root: Path) -> bool:
    try:
        path = path.resolve()
        root = root.resolve()
        return str(path).startswith(str(root) + os.sep)
    except Exception:
        return False

"""
Path validation and security utilities for AutoResolve.
Prevents path traversal attacks and validates file access.
"""

from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class PathSecurityError(Exception):
    """Raised when a path security violation is detected."""
    pass

class PathValidator:
    """Validates and sanitizes file paths for security."""
    
    def __init__(self, media_roots: Optional[List[str]] = None):
        """
        Initialize path validator with allowed media roots.
        
        Args:
            media_roots: List of root directories for media files
        """
        roots_env = os.getenv("ALLOWED_MEDIA_ROOTS", "").strip()
        env_roots = [r for r in (roots_env.split(",") if roots_env else []) if r.strip()]
        roots_list = media_roots if media_roots is not None and len(media_roots) > 0 else env_roots
        if not roots_list:
            roots_list = ["/Users/hawzhin/Videos"]
        self.allowed_roots: List[Path] = [Path(r).expanduser().resolve() for r in roots_list]
        logger.info(f"PathValidator initialized with media roots: {self.allowed_roots}")
    
    def validate_path(self, user_path: str) -> Path:
        """
        Validate and sanitize a user-provided path.
        
        Args:
            user_path: Path string from user input
            
        Returns:
            Validated Path object
            
        Raises:
            PathSecurityError: If path traversal attempt detected
        """
        try:
            # Resolve to absolute path and remove any .. or . components
            requested_path = Path(user_path).resolve()
            
            # Check if path is within any allowed media root
            within_any = False
            for root in self.allowed_roots:
                try:
                    if requested_path.is_relative_to(root):
                        within_any = True
                        break
                except Exception:
                    continue
            if not within_any:
                logger.error(f"Path traversal attempt detected: {user_path}")
                allowed_str = ", ".join(str(r) for r in self.allowed_roots)
                raise PathSecurityError(
                    f"Access denied: Path must be within one of: {allowed_str}"
                )
            
            # Check if path exists
            if not requested_path.exists():
                logger.warning(f"Path does not exist: {requested_path}")
                raise FileNotFoundError(f"File not found: {user_path}")
            
            # Check if it's a file (not directory)
            if requested_path.is_dir():
                raise PathSecurityError("Directory access not allowed")
            
            # Check file extension is allowed
            allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.mp3', '.wav', '.m4a'}
            if requested_path.suffix.lower() not in allowed_extensions:
                raise PathSecurityError(
                    f"File type not allowed: {requested_path.suffix}"
                )
            
            logger.debug(f"Path validated successfully: {requested_path}")
            return requested_path
            
        except Exception as e:
            if isinstance(e, (PathSecurityError, FileNotFoundError)):
                raise
            logger.error(f"Path validation error: {e}")
            raise PathSecurityError(f"Invalid path: {user_path}")
    
    def validate_output_path(self, output_path: str) -> Path:
        """
        Validate a path for writing output files.
        
        Args:
            output_path: Desired output path
            
        Returns:
            Validated Path object
            
        Raises:
            PathSecurityError: If path is unsafe
        """
        try:
            # Resolve path
            requested_path = Path(output_path).resolve()
            
            # Ensure parent directory exists and is writable
            parent_dir = requested_path.parent
            if not parent_dir.exists():
                parent_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if within allowed root
            within_any = False
            for root in self.allowed_roots:
                try:
                    if requested_path.is_relative_to(root):
                        within_any = True
                        break
                except Exception:
                    continue
            if not within_any:
                allowed_str = ", ".join(str(r) for r in self.allowed_roots)
                raise PathSecurityError(
                    f"Output path must be within one of: {allowed_str}"
                )
            
            # Prevent overwriting system files
            forbidden_patterns = ['..', '~', '/etc/', '/usr/', '/bin/', '/sbin/']
            if any(pattern in str(requested_path) for pattern in forbidden_patterns):
                raise PathSecurityError("Forbidden path pattern detected")
            
            return requested_path
            
        except Exception as e:
            if isinstance(e, PathSecurityError):
                raise
            logger.error(f"Output path validation error: {e}")
            raise PathSecurityError(f"Invalid output path: {output_path}")
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename to remove dangerous characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        import re
        
        # Remove path separators and null bytes
        filename = filename.replace('/', '_').replace('\\', '_').replace('\0', '')
        
        # Remove other dangerous characters
        filename = re.sub(r'[<>:"|?*]', '_', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Limit length
        max_length = 255
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            filename = name[:max_length - len(ext)] + ext
        
        return filename or 'unnamed'

# Global validator instance
_validator: Optional[PathValidator] = None

def get_validator(media_root: Optional[str] = None) -> PathValidator:
    """Get or create the global path validator.
    If media_root is provided, it will be used as the single allowed root; otherwise env ALLOWED_MEDIA_ROOTS applies.
    """
    global _validator
    if _validator is None or media_root:
        roots = [media_root] if media_root else None
        _validator = PathValidator(roots)
    return _validator

def validate_input_path(path: str) -> Path:
    """Convenience function to validate input paths."""
    return get_validator().validate_path(path)

def validate_output_path(path: str) -> Path:
    """Convenience function to validate output paths."""
    return get_validator().validate_output_path(path)