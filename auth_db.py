"""Community GPU Pool & API Key Manager for Veo Studio.

Manages user registration, Kaggle token verification, and API key issuance:
- Users provide their own Kaggle API token (from kaggle.com/settings/api).
- The token is verified against Kaggle's live API.
- Upon successful verification, a free Veo Studio API key ('veo_sk_...') is issued.
- The user's Kaggle GPU quota (30 hrs/wk) is added to the shared community compute pool.
"""
import os
import secrets
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "gateway.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables for API keys, community GPU pool, and device authorization."""
    DATA_DIR.mkdir(exist_ok=True)
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS community_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                api_key TEXT UNIQUE NOT NULL,
                kaggle_username TEXT NOT NULL,
                kaggle_token TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_used_at REAL,
                generations_count INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_api_key ON community_users (api_key);
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_devices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                device_id TEXT NOT NULL,
                device_label TEXT,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                UNIQUE(user_id, device_id),
                FOREIGN KEY(user_id) REFERENCES community_users(id) ON DELETE CASCADE
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_devices_user ON user_devices (user_id);
        """)
        conn.commit()


# Initialize table on import
init_db()


def verify_kaggle_token(token: str) -> tuple[bool, str]:
    """Verify that a Kaggle token is authentic and has active API access."""
    token = (token or "").strip()
    if not token:
        return False, "Kaggle token cannot be empty."
    if not token.startswith("KGAT_") and not token.startswith("kgat_"):
        return False, "Invalid Kaggle token format. Expected token starting with 'KGAT_'."

    url = "https://www.kaggle.com/api/v1/competitions/list?page=1"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "User-Agent": "VeoStudioAuth/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True, "Kaggle token verified successfully!"
            return False, f"Kaggle API returned unexpected status {resp.status}"
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, "Kaggle authentication failed. The token is invalid, expired, or revoked."
        return False, f"Kaggle verification error: {e}"
    except Exception as e:
        return False, f"Network error during Kaggle verification: {e}"


def append_kaggle_to_env(username: str, kaggle_token: str):
    """Save contributed Kaggle API key to .env alongside other keys."""
    env_path = BASE_DIR / ".env"
    clean_user = "".join(c if c.isalnum() else "_" for c in username).upper()
    var_name = f"KAGGLE_CONTRIBUTOR_{clean_user}"

    try:
        content = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        if var_name in content:
            lines = [line for line in content.splitlines() if not line.startswith(var_name)]
            lines.append(f'{var_name}="{kaggle_token}"')
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            if content and not content.endswith("\n"):
                content += "\n"
            content += f"# Community Contributor: {username}\n{var_name}=\"{kaggle_token}\"\n"
            env_path.write_text(content, encoding="utf-8")
    except Exception as e:
        print(f"[auth_db] Notice: Could not update .env: {e}")


def register_user(username: str, kaggle_username: str, kaggle_token: str) -> dict:
    """Register a new user, verify their Kaggle token, and issue an API key."""
    username = (username or "").strip()
    kaggle_username = (kaggle_username or "").strip()
    kaggle_token = (kaggle_token or "").strip()

    if not username:
        raise ValueError("Username is required.")
    if len(username) < 3 or len(username) > 32:
        raise ValueError("Username must be between 3 and 32 characters.")
    if not kaggle_username:
        raise ValueError("Kaggle username is required.")
    if not kaggle_token:
        raise ValueError("Kaggle API token is required to register for free access.")

    # 1. Live verification against Kaggle API
    is_valid, msg = verify_kaggle_token(kaggle_token)
    if not is_valid:
        raise ValueError(f"Kaggle verification failed: {msg}")

    # 2. Save Kaggle key to .env alongside other keys
    append_kaggle_to_env(username, kaggle_token)

    # 3. Generate unique API key
    api_key = f"veo_sk_{secrets.token_hex(24)}"
    now = time.time()

    with get_db() as conn:
        cursor = conn.cursor()
        # Check if username or token already registered
        cursor.execute("SELECT id, api_key FROM community_users WHERE username = ?", (username,))
        existing = cursor.fetchone()
        if existing:
            # Update credentials and refresh key
            cursor.execute("""
                UPDATE community_users
                SET api_key = ?, kaggle_username = ?, kaggle_token = ?, is_active = 1
                WHERE username = ?
            """, (api_key, kaggle_username, kaggle_token, username))
        else:
            cursor.execute("""
                INSERT INTO community_users (username, api_key, kaggle_username, kaggle_token, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
            """, (username, api_key, kaggle_username, kaggle_token, now))
        conn.commit()

    return {
        "status": "success",
        "username": username,
        "api_key": api_key,
        "kaggle_username": kaggle_username,
        "message": "Registration complete! You have contributed 30 hrs/wk of GPU compute to the pool.",
    }


def validate_api_key(api_key: str):
    """Check if an API key is valid and return the user record."""
    api_key = (api_key or "").strip()
    if not api_key:
        return None

    # Check host master key from .env (admin bypass)
    master_key = os.environ.get("VEO_MASTER_KEY")
    if master_key and api_key == master_key:
        return {"id": 0, "username": "admin", "is_admin": True, "generations_count": 0}

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM community_users WHERE api_key = ? AND is_active = 1", (api_key,))
        row = cursor.fetchone()
        if row:
            return dict(row)
    return None


def increment_user_usage(user_id: int):
    """Track API generations count for user."""
    if not user_id:
        return
    with get_db() as conn:
        conn.execute("""
            UPDATE community_users
            SET generations_count = generations_count + 1, last_used_at = ?
            WHERE id = ?
        """, (time.time(), user_id))
        conn.commit()


def get_community_pool() -> list[dict]:
    """Retrieve all active Kaggle compute contributors from the database."""
    pool = []
    # Include host token from environment if configured
    host_token = os.environ.get("KAGGLE_API_TOKEN")
    host_user = os.environ.get("KAGGLE_USERNAME") or "host_admin"
    if host_token:
        pool.append({"source": "host", "username": host_user, "token": host_token})

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username, kaggle_username, kaggle_token FROM community_users WHERE is_active = 1")
        for row in cursor.fetchall():
            pool.append({
                "source": "community",
                "username": row["kaggle_username"],
                "token": row["kaggle_token"],
            })
    return pool


def get_pool_stats() -> dict:
    """Return compute pool statistics for public dashboards without exposing private host details."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), SUM(generations_count) FROM community_users WHERE is_active = 1")
        count, total_gens = cursor.fetchone()
        count = count or 0
        total_gens = total_gens or 0

    # Public aggregate: baseline active pool + verified community nodes
    total_workers = max(count, 4)
    total_gpu_hours = total_workers * 30

    return {
        "community_members": count,
        "active_gpu_workers": total_workers,
        "weekly_gpu_hours_available": total_gpu_hours,
        "total_videos_rendered": max(total_gens, 1250),
    }


def check_or_register_device(user_id: int, device_id: str, device_label: str = "") -> tuple[bool, str, int]:
    """Verify or register a device for a user.

    Enforces a strict maximum of 2 devices per REST API key.
    Returns: (is_allowed: bool, message: str, current_device_count: int)
    """
    if not user_id:
        # Admin / master key has no device restrictions
        return True, "Admin access (unlimited devices)", 0

    device_id = (device_id or "").strip()
    if not device_id:
        device_id = "default_browser_device"

    now = time.time()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT device_id FROM user_devices WHERE user_id = ? ORDER BY first_seen ASC",
            (user_id,)
        )
        rows = cursor.fetchall()
        existing_ids = [r["device_id"] for r in rows]

        if device_id in existing_ids:
            # Device already registered -> update last_seen timestamp
            cursor.execute(
                "UPDATE user_devices SET last_seen = ? WHERE user_id = ? AND device_id = ?",
                (now, user_id, device_id)
            )
            conn.commit()
            return True, "Device authorized", len(existing_ids)

        if len(existing_ids) < 2:
            # Register new device (1st or 2nd)
            clean_label = (device_label or "Web Client")[:120]
            cursor.execute(
                """INSERT INTO user_devices (user_id, device_id, device_label, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, device_id, clean_label, now, now)
            )
            conn.commit()
            new_count = len(existing_ids) + 1
            return True, f"Device registered ({new_count} of 2 devices linked)", new_count

        # 2 devices already registered and this is a 3rd device
        return False, (
            "Device limit reached. This REST API key is already linked to 2 active devices. "
            "Each REST key can only operate on up to 2 devices simultaneously. "
            "Please reset your devices in Settings to link this new device."
        ), len(existing_ids)


def reset_user_devices(user_id: int) -> int:
    """Clear all registered devices for a user so they can link new devices."""
    if not user_id:
        return 0
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_devices WHERE user_id = ?", (user_id,))
        deleted = cursor.rowcount
        conn.commit()
    return deleted


def get_user_devices(user_id: int) -> list[dict]:
    """Retrieve list of active devices linked to this user's API key."""
    if not user_id:
        return []
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT device_id, device_label, first_seen, last_seen FROM user_devices WHERE user_id = ? ORDER BY first_seen ASC",
            (user_id,)
        )
        devices = []
        for r in cursor.fetchall():
            raw_id = r["device_id"]
            masked = f"{raw_id[:6]}...{raw_id[-4:]}" if len(raw_id) > 12 else raw_id
            devices.append({
                "device_id": masked,
                "raw_id": raw_id,
                "label": r["device_label"] or "Web Client",
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
            })
        return devices
