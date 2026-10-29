"""Authentication and authorization system for production deployment."""

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import Header, HTTPException
from pydantic import BaseModel

from field_recognition.runtime.database import Database


Role = Literal["admin", "operator", "reviewer"]


class User(BaseModel):
    """User account model."""

    id: str
    username: str
    display_name: str
    role: Role
    created_at: str
    disabled: bool = False


class Session(BaseModel):
    """Authentication session model."""

    id: str
    user_id: str
    created_at: str
    expires_at: str
    last_active_at: str


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1)
    return "scrypt$16384$" + salt.hex() + "$" + derived.hex()


def verify_password(password: str, password_hash: str) -> bool:
    if not password_hash:
        return False
    if password_hash.startswith("scrypt$"):
        try:
            _, n, salt, expected = password_hash.split("$")
            if n != "16384":
                return False
            actual = hashlib.scrypt(
                password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1
            ).hex()
        except (ValueError, TypeError):
            return False
        return secrets.compare_digest(actual, expected)
    # Successful legacy logins upgrade the previous unsalted digest immediately.
    return secrets.compare_digest(hashlib.sha256(password.encode()).hexdigest(), password_hash)


class AuthService:
    """Authentication and authorization service."""

    def __init__(self, db: Database, *, session_duration_hours: int = 8):
        if type(session_duration_hours) is not int or not 1 <= session_duration_hours <= 72:
            raise ValueError("session_duration_hours must be an integer between 1 and 72")
        self.db = db
        self.session_duration_hours = session_duration_hours

    def create_initial_admin(self, username: str, display_name: str, password: str) -> User:
        """
        Create the first admin user. Only works when no users exist.

        Args:
            username: Unique username
            display_name: Display name
            password: Admin password

        Returns:
            Created user

        Raises:
            HTTPException: If users already exist or creation fails
        """
        with self.db.transaction("IMMEDIATE") as conn:
            existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if existing > 0:
                raise HTTPException(403, "管理员账号已存在，请使用现有管理员创建新用户")

            user_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            conn.execute(
                "INSERT INTO users VALUES(?,?,?,?,?,?,?)",
                (user_id, username, display_name, "admin", hash_password(password), now, 0),
            )

            return User(
                id=user_id,
                username=username,
                display_name=display_name,
                role="admin",
                created_at=now,
                disabled=False,
            )

    def create_user(
        self, username: str, display_name: str, role: Role, password: str | None, created_by: str
    ) -> User:
        """
        Create a new user (admin only).

        Args:
            username: Unique username
            display_name: Display name
            role: User role
            password: Optional password (None for device accounts)
            created_by: User ID of creator (must be admin)

        Returns:
            Created user

        Raises:
            HTTPException: If creator is not admin or username exists
        """
        with self.db.transaction("IMMEDIATE") as conn:
            # Verify creator is admin
            creator = conn.execute(
                "SELECT role FROM users WHERE id=? AND disabled=0", (created_by,)
            ).fetchone()
            if not creator or creator["role"] != "admin":
                raise HTTPException(403, "只有管理员可以创建用户")

            # Check username uniqueness
            existing = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
            if existing:
                raise HTTPException(409, "用户名已存在")

            user_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            pwd_hash = hash_password(password) if password else None

            conn.execute(
                "INSERT INTO users VALUES(?,?,?,?,?,?,?)",
                (user_id, username, display_name, role, pwd_hash, now, 0),
            )

            return User(
                id=user_id,
                username=username,
                display_name=display_name,
                role=role,
                created_at=now,
                disabled=False,
            )

    def login(self, username: str, password: str) -> tuple[Session, User]:
        """
        Authenticate user and create session.

        Args:
            username: Username
            password: Password

        Returns:
            Tuple of (session, user)

        Raises:
            HTTPException: If authentication fails
        """
        with self.db.transaction("IMMEDIATE") as conn:
            row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()

            if not row:
                raise HTTPException(401, "用户名或密码错误")

            user = dict(row)

            if user["disabled"]:
                raise HTTPException(403, "账号已禁用")

            if not user["password_hash"]:
                raise HTTPException(401, "此账号不支持密码登录")

            if not verify_password(password, user["password_hash"]):
                raise HTTPException(401, "用户名或密码错误")

            if not user["password_hash"].startswith("scrypt$"):
                conn.execute(
                    "UPDATE users SET password_hash=? WHERE id=?",
                    (hash_password(password), user["id"]),
                )

            # Create session
            session_id = secrets.token_urlsafe(32)
            now = datetime.now(timezone.utc)
            expires = now + timedelta(hours=self.session_duration_hours)

            conn.execute(
                "INSERT INTO sessions VALUES(?,?,?,?,?,?)",
                (
                    hashlib.sha256(session_id.encode()).hexdigest(),
                    user["id"],
                    now.isoformat(),
                    expires.isoformat(),
                    now.isoformat(),
                    None,
                ),
            )

            session = Session(
                id=session_id,
                user_id=user["id"],
                created_at=now.isoformat(),
                expires_at=expires.isoformat(),
                last_active_at=now.isoformat(),
            )

            return session, User(**{k: user[k] for k in User.model_fields})

    def verify_session(self, session_id: str | None) -> User:
        """
        Verify session and return user.

        Args:
            session_id: Session ID from header

        Returns:
            Authenticated user

        Raises:
            HTTPException: If session is invalid or expired
        """
        if not session_id:
            raise HTTPException(401, "缺少认证令牌")

        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT u.*,s.last_active_at AS session_last_active_at FROM users u
                JOIN sessions s ON s.user_id = u.id
                WHERE s.id = ? AND s.expires_at > ?
            """,
                (
                    hashlib.sha256(session_id.encode()).hexdigest(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            ).fetchone()

            if not row:
                raise HTTPException(401, "会话已过期或无效")

            user = dict(row)

            if user["disabled"]:
                raise HTTPException(403, "账号已禁用")

            # Polling/stream reads must not upgrade a deferred read transaction
            # into a write (SQLITE_BUSY_SNAPSHOT with concurrent camera workers).
            now = datetime.now(timezone.utc)
            if (now - datetime.fromisoformat(user["session_last_active_at"])).total_seconds() >= 60:
                conn.execute(
                    "UPDATE sessions SET last_active_at=? WHERE id=?",
                    (now.isoformat(), hashlib.sha256(session_id.encode()).hexdigest()),
                )

            return User(**{k: user[k] for k in User.model_fields})

    def logout(self, session_id: str):
        """Delete session (logout)."""
        with self.db.transaction() as conn:
            conn.execute(
                "DELETE FROM sessions WHERE id=?",
                (hashlib.sha256(session_id.encode()).hexdigest(),),
            )

    def cleanup_expired_sessions(self):
        """Remove expired sessions (call periodically)."""
        with self.db.transaction() as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))

    def register_device(
        self, device_id: str, device_type: str, credential: str, registered_by: str
    ) -> dict:
        """
        Register a camera device with authentication credential.

        Args:
            device_id: Unique device identifier
            device_type: Device type (e.g., 'neck_camera')
            credential: Device secret/token
            registered_by: User ID of registrar (must be admin)

        Returns:
            Device registration info

        Raises:
            HTTPException: If registrar is not admin
        """
        with self.db.transaction("IMMEDIATE") as conn:
            # Verify registrar is admin
            registrar = conn.execute(
                "SELECT role FROM users WHERE id=? AND disabled=0", (registered_by,)
            ).fetchone()
            if not registrar or registrar["role"] != "admin":
                raise HTTPException(403, "只有管理员可以注册设备")

            credential_hash = hash_password(credential)
            now = datetime.now(timezone.utc).isoformat()

            conn.execute(
                "INSERT INTO device_credentials VALUES(?,?,?,?,?) ON CONFLICT(device_id) DO UPDATE SET credential_hash=excluded.credential_hash,disabled=0",
                (device_id, device_type, credential_hash, now, 0),
            )

            return {"device_id": device_id, "device_type": device_type, "registered_at": now}

    def verify_device(self, device_id: str, credential: str) -> dict:
        """
        Verify device credential.

        Args:
            device_id: Device identifier
            credential: Device secret/token

        Returns:
            Device info

        Raises:
            HTTPException: If device authentication fails
        """
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM device_credentials WHERE device_id=?", (device_id,)
            ).fetchone()

            if not row:
                raise HTTPException(401, "设备未注册")

            device = dict(row)

            if device["disabled"]:
                raise HTTPException(403, "设备已禁用")

            if not verify_password(credential, device["credential_hash"]):
                raise HTTPException(401, "设备凭证错误")

            return {"device_id": device["device_id"], "device_type": device["device_type"]}

    def require_role(self, user: User, required_role: Role | list[Role]):
        """
        Check if user has required role.

        Args:
            user: User to check
            required_role: Required role or list of acceptable roles

        Raises:
            HTTPException: If user doesn't have required role
        """
        allowed_roles = [required_role] if isinstance(required_role, str) else required_role

        if user.role not in allowed_roles:
            raise HTTPException(403, f"需要 {'/'.join(allowed_roles)} 权限")


def get_current_user_optional(
    x_session_id: str | None = Header(None), auth_service: AuthService | None = None
) -> User | None:
    """Dependency to get current user (optional, returns None if not authenticated)."""
    if not x_session_id or not auth_service:
        return None
    try:
        return auth_service.verify_session(x_session_id)
    except HTTPException:
        return None


def get_current_user_required(
    x_session_id: str | None = Header(None), auth_service: AuthService | None = None
) -> User:
    """Dependency to require authenticated user."""
    if not auth_service:
        raise HTTPException(503, "认证服务未启用")
    return auth_service.verify_session(x_session_id)


def require_admin(user: User = None):
    """Dependency to require admin role."""
    if not user or user.role != "admin":
        raise HTTPException(403, "需要管理员权限")
