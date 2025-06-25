"""
Identity and Access Management
==============================

Keycloak-integrated identity management with RBAC, SSO, and multi-tenant support.

Context7 References:
- /keycloak/keycloak: Identity provider and SSO
- /waza-ari/fastapi-keycloak-middleware: FastAPI integration
- /tiangolo/fastapi: API authentication and authorization
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import jwt
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class AuthenticationMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    SSO = "sso"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    OAUTH2 = "oauth2"

class UserStatus(Enum):
    """User status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"

class PermissionType(Enum):
    """Permission types"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"

@dataclass
class Permission:
    """Permission definition"""
    resource: str
    action: str
    conditions: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        return f"{self.resource}:{self.action}"

@dataclass
class Role:
    """Role definition with permissions"""
    role_id: str
    name: str
    description: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    tenant_id: Optional[str] = None  # For tenant-specific roles
    is_system_role: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_permission(self, resource: str, action: str, conditions: Optional[Dict[str, Any]] = None):
        """Add permission to role."""
        permission = Permission(resource, action, conditions)
        self.permissions.add(permission)
    
    def has_permission(self, resource: str, action: str) -> bool:
        """Check if role has specific permission."""
        return any(p.resource == resource and p.action == action for p in self.permissions)

@dataclass
class User:
    """User with multi-tenant and role information"""
    user_id: str
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    status: UserStatus = UserStatus.ACTIVE
    tenant_id: Optional[str] = None
    roles: Set[str] = field(default_factory=set)  # Role IDs
    attributes: Dict[str, Any] = field(default_factory=dict)
    last_login: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username

@dataclass
class AuthenticationResult:
    """Authentication result"""
    success: bool
    user: Optional[User] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    error_message: Optional[str] = None
    method: Optional[AuthenticationMethod] = None

@dataclass
class AuthorizationResult:
    """Authorization result"""
    allowed: bool
    user: Optional[User] = None
    matched_permissions: List[Permission] = field(default_factory=list)
    reason: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class IdentityManager:
    """
    Identity and access management with Keycloak integration.
    
    Features:
    - User authentication and authorization
    - Role-based access control (RBAC)
    - Multi-tenant user management
    - SSO integration via Keycloak
    - Token management and validation
    - Permission-based authorization
    
    Context7 Pattern:
    - Uses Keycloak for identity provider services
    - Integrates with FastAPI middleware for authentication
    - Implements RBAC with flexible permission system
    """
    
    def __init__(self,
                 keycloak_url: str = "http://localhost:8080",
                 keycloak_realm: str = "master",
                 keycloak_client_id: str = "m3tm-client",
                 keycloak_client_secret: Optional[str] = None,
                 admin_username: Optional[str] = None,
                 admin_password: Optional[str] = None):
        """
        Initialize identity manager with Keycloak integration.
        
        Args:
            keycloak_url: Keycloak server URL
            keycloak_realm: Default Keycloak realm
            keycloak_client_id: Client ID for M³TM
            keycloak_client_secret: Client secret for confidential clients
            admin_username: Admin username for Keycloak admin operations
            admin_password: Admin password for Keycloak admin operations
        """
        self.keycloak_url = keycloak_url
        self.keycloak_realm = keycloak_realm
        self.keycloak_client_id = keycloak_client_id
        self.keycloak_client_secret = keycloak_client_secret
        self.admin_username = admin_username
        self.admin_password = admin_password
        
        # Internal storage (cache)
        self._users: Dict[str, User] = {}
        self._roles: Dict[str, Role] = {}
        self._user_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Token validation cache
        self._token_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize system roles
        asyncio.create_task(self._initialize_system_roles())
        
        logger.info(f"IdentityManager initialized with Keycloak: {keycloak_url}")
    
    async def _initialize_system_roles(self):
        """Initialize default system roles."""
        try:
            system_roles = [
                Role(
                    role_id="system_admin",
                    name="System Administrator",
                    description="Full system access",
                    is_system_role=True
                ),
                Role(
                    role_id="tenant_admin",
                    name="Tenant Administrator", 
                    description="Tenant administration access",
                    is_system_role=True
                ),
                Role(
                    role_id="data_scientist",
                    name="Data Scientist",
                    description="Model training and experimentation access",
                    is_system_role=True
                ),
                Role(
                    role_id="api_user",
                    name="API User",
                    description="Basic API access",
                    is_system_role=True
                )
            ]
            
            # Add permissions to system roles
            # System Admin - full access
            system_admin = system_roles[0]
            system_admin.add_permission("*", "*")
            
            # Tenant Admin - tenant management
            tenant_admin = system_roles[1]
            tenant_admin.add_permission("tenant", "read")
            tenant_admin.add_permission("tenant", "write")
            tenant_admin.add_permission("user", "read")
            tenant_admin.add_permission("user", "write")
            tenant_admin.add_permission("model", "read")
            tenant_admin.add_permission("model", "write")
            
            # Data Scientist - model and data access
            data_scientist = system_roles[2]
            data_scientist.add_permission("model", "read")
            data_scientist.add_permission("model", "write")
            data_scientist.add_permission("model", "execute")
            data_scientist.add_permission("data", "read")
            data_scientist.add_permission("experiment", "read")
            data_scientist.add_permission("experiment", "write")
            
            # API User - basic read access
            api_user = system_roles[3]
            api_user.add_permission("model", "read")
            api_user.add_permission("model", "execute")
            api_user.add_permission("data", "read")
            
            # Store roles
            for role in system_roles:
                self._roles[role.role_id] = role
            
            logger.info(f"Initialized {len(system_roles)} system roles")
            
        except Exception as e:
            logger.error(f"Failed to initialize system roles: {e}")
    
    async def authenticate_user(self, username: str, password: str,
                              tenant_id: Optional[str] = None) -> AuthenticationResult:
        """
        Authenticate user with username/password.
        
        Context7 Pattern: Uses Keycloak token endpoint
        """
        try:
            # Determine realm
            realm = f"m3tm-{tenant_id}" if tenant_id else self.keycloak_realm
            
            # TODO: Integrate with actual Keycloak token endpoint
            # For now, mock authentication
            if username == "admin" and password == "admin":
                user = User(
                    user_id="admin",
                    username="admin",
                    email="admin@m3tm.example.com",
                    first_name="System",
                    last_name="Administrator",
                    tenant_id=tenant_id,
                    roles={"system_admin"}
                )
                
                # Mock JWT token
                token_payload = {
                    "sub": user.user_id,
                    "preferred_username": user.username,
                    "email": user.email,
                    "realm_access": {"roles": list(user.roles)},
                    "tenant_id": tenant_id,
                    "iat": datetime.now(timezone.utc).timestamp(),
                    "exp": (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()
                }
                
                access_token = jwt.encode(token_payload, "secret", algorithm="HS256")
                
                # Store user and update last login
                user.last_login = datetime.now(timezone.utc)
                self._users[user.user_id] = user
                
                # Store session
                self._user_sessions[access_token] = {
                    "user_id": user.user_id,
                    "tenant_id": tenant_id,
                    "created_at": datetime.now(timezone.utc),
                    "last_activity": datetime.now(timezone.utc)
                }
                
                return AuthenticationResult(
                    success=True,
                    user=user,
                    access_token=access_token,
                    expires_in=3600,
                    method=AuthenticationMethod.PASSWORD
                )
            
            return AuthenticationResult(
                success=False,
                error_message="Invalid credentials",
                method=AuthenticationMethod.PASSWORD
            )
            
        except Exception as e:
            logger.error(f"Authentication failed for user {username}: {e}")
            return AuthenticationResult(
                success=False,
                error_message=f"Authentication error: {e}",
                method=AuthenticationMethod.PASSWORD
            )
    
    async def validate_token(self, token: str) -> AuthenticationResult:
        """
        Validate JWT token and extract user information.
        
        Context7 Pattern: Validates Keycloak-issued JWT tokens
        """
        try:
            # Check token cache
            if token in self._token_cache:
                cached_data = self._token_cache[token]
                if datetime.now(timezone.utc) < cached_data["expires_at"]:
                    user_id = cached_data["user_id"]
                    user = self._users.get(user_id)
                    if user:
                        return AuthenticationResult(
                            success=True,
                            user=user,
                            access_token=token
                        )
            
            # Decode and validate JWT
            # TODO: Use Keycloak public key for validation
            try:
                payload = jwt.decode(token, "secret", algorithms=["HS256"])
            except jwt.ExpiredSignatureError:
                return AuthenticationResult(
                    success=False,
                    error_message="Token expired"
                )
            except jwt.InvalidTokenError:
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid token"
                )
            
            # Extract user information
            user_id = payload.get("sub")
            username = payload.get("preferred_username")
            email = payload.get("email")
            tenant_id = payload.get("tenant_id")
            roles = set(payload.get("realm_access", {}).get("roles", []))
            
            # Get or create user
            user = self._users.get(user_id)
            if not user:
                user = User(
                    user_id=user_id,
                    username=username,
                    email=email,
                    tenant_id=tenant_id,
                    roles=roles
                )
                self._users[user_id] = user
            
            # Cache token
            self._token_cache[token] = {
                "user_id": user_id,
                "expires_at": datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
            }
            
            # Update session activity
            if token in self._user_sessions:
                self._user_sessions[token]["last_activity"] = datetime.now(timezone.utc)
            
            return AuthenticationResult(
                success=True,
                user=user,
                access_token=token
            )
            
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return AuthenticationResult(
                success=False,
                error_message=f"Token validation error: {e}"
            )
    
    async def authorize_user(self, user: User, resource: str, action: str,
                           context: Optional[Dict[str, Any]] = None) -> AuthorizationResult:
        """
        Authorize user action on resource.
        
        Checks user roles and permissions against requested resource/action.
        """
        try:
            matched_permissions = []
            
            # Check user roles
            for role_id in user.roles:
                role = self._roles.get(role_id)
                if not role:
                    continue
                
                # Check role permissions
                for permission in role.permissions:
                    # Wildcard permissions
                    if permission.resource == "*" and permission.action == "*":
                        matched_permissions.append(permission)
                        continue
                    
                    # Exact match
                    if permission.resource == resource and permission.action == action:
                        # Check conditions if specified
                        if permission.conditions:
                            if not self._check_permission_conditions(permission.conditions, context):
                                continue
                        matched_permissions.append(permission)
                        continue
                    
                    # Resource wildcard
                    if permission.resource == resource and permission.action == "*":
                        matched_permissions.append(permission)
                        continue
            
            # Authorization decision
            allowed = len(matched_permissions) > 0
            
            result = AuthorizationResult(
                allowed=allowed,
                user=user,
                matched_permissions=matched_permissions,
                reason="Permission granted" if allowed else "No matching permissions found",
                context=context
            )
            
            # Log authorization decision
            logger.debug(f"Authorization: {user.username} {action} {resource} = {allowed}")
            
            return result
            
        except Exception as e:
            logger.error(f"Authorization failed for user {user.username}: {e}")
            return AuthorizationResult(
                allowed=False,
                user=user,
                reason=f"Authorization error: {e}",
                context=context
            )
    
    def _check_permission_conditions(self, conditions: Dict[str, Any], 
                                   context: Optional[Dict[str, Any]]) -> bool:
        """Check if permission conditions are met."""
        if not context:
            return False
        
        for key, expected_value in conditions.items():
            if key not in context:
                return False
            if context[key] != expected_value:
                return False
        
        return True
    
    async def create_user(self, username: str, email: str, password: str,
                        first_name: Optional[str] = None,
                        last_name: Optional[str] = None,
                        tenant_id: Optional[str] = None,
                        roles: Optional[Set[str]] = None) -> User:
        """
        Create new user in identity system.
        
        Context7 Pattern: Uses Keycloak Admin API for user creation
        """
        try:
            # Generate user ID
            user_id = f"user_{len(self._users) + 1}"
            
            # Create user object
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                first_name=first_name,
                last_name=last_name,
                tenant_id=tenant_id,
                roles=roles or {"api_user"}
            )
            
            # TODO: Create user in Keycloak
            realm = f"m3tm-{tenant_id}" if tenant_id else self.keycloak_realm
            
            # Store user locally
            self._users[user_id] = user
            
            logger.info(f"User created: {username} ({user_id})")
            return user
            
        except Exception as e:
            logger.error(f"Failed to create user {username}: {e}")
            raise
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        for user in self._users.values():
            if user.username == username:
                return user
        return None
    
    async def update_user_roles(self, user_id: str, roles: Set[str]) -> bool:
        """Update user roles."""
        try:
            user = self._users.get(user_id)
            if not user:
                return False
            
            user.roles = roles
            user.updated_at = datetime.now(timezone.utc)
            
            logger.info(f"User roles updated: {user.username} -> {roles}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user roles for {user_id}: {e}")
            return False
    
    async def create_role(self, role_id: str, name: str, description: Optional[str] = None,
                        permissions: Optional[List[tuple]] = None,
                        tenant_id: Optional[str] = None) -> Role:
        """Create custom role with permissions."""
        try:
            role = Role(
                role_id=role_id,
                name=name,
                description=description,
                tenant_id=tenant_id,
                is_system_role=False
            )
            
            # Add permissions
            if permissions:
                for resource, action in permissions:
                    role.add_permission(resource, action)
            
            self._roles[role_id] = role
            
            logger.info(f"Role created: {role_id} ({name})")
            return role
            
        except Exception as e:
            logger.error(f"Failed to create role {role_id}: {e}")
            raise
    
    async def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID."""
        return self._roles.get(role_id)
    
    async def list_roles(self, tenant_id: Optional[str] = None) -> List[Role]:
        """List roles, optionally filtered by tenant."""
        roles = list(self._roles.values())
        if tenant_id:
            roles = [r for r in roles if r.tenant_id == tenant_id or r.is_system_role]
        return roles
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke access token."""
        try:
            # Remove from cache
            self._token_cache.pop(token, None)
            
            # Remove session
            self._user_sessions.pop(token, None)
            
            # TODO: Revoke token in Keycloak
            
            logger.info("Token revoked")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False
    
    async def get_user_permissions(self, user: User) -> Set[str]:
        """Get all permissions for a user."""
        permissions = set()
        
        for role_id in user.roles:
            role = self._roles.get(role_id)
            if role:
                for permission in role.permissions:
                    permissions.add(str(permission))
        
        return permissions
    
    async def get_identity_metrics(self) -> Dict[str, Any]:
        """Get identity management metrics."""
        try:
            users = list(self._users.values())
            active_sessions = len(self._user_sessions)
            
            # User status distribution
            status_counts = {}
            for user in users:
                status = user.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Role distribution
            role_counts = {}
            for user in users:
                for role_id in user.roles:
                    role_counts[role_id] = role_counts.get(role_id, 0) + 1
            
            return {
                "total_users": len(users),
                "total_roles": len(self._roles),
                "active_sessions": active_sessions,
                "user_status_distribution": status_counts,
                "role_distribution": role_counts,
                "cached_tokens": len(self._token_cache),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get identity metrics: {e}")
            return {"error": str(e)}

# FastAPI dependencies for authentication and authorization
async def get_current_user(token: str, identity_manager: IdentityManager) -> User:
    """
    FastAPI dependency to get current authenticated user.
    
    Context7 Pattern: FastAPI dependency injection with Keycloak middleware
    """
    result = await identity_manager.validate_token(token)
    if not result.success or not result.user:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    return result.user

def require_permission(resource: str, action: str):
    """
    FastAPI dependency factory for permission-based authorization.
    
    Context7 Pattern: Permission-based access control decorator
    """
    async def permission_checker(user: User = None, identity_manager: IdentityManager = None) -> bool:
        if not user or not identity_manager:
            from fastapi import HTTPException
            raise HTTPException(status_code=401, detail="Authentication required")
        
        result = await identity_manager.authorize_user(user, resource, action)
        if not result.allowed:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail=f"Permission denied: {result.reason}")
        
        return True
    
    return permission_checker

# Export main classes
__all__ = [
    "IdentityManager", "User", "Role", "Permission", "AuthenticationResult", "AuthorizationResult",
    "AuthenticationMethod", "UserStatus", "PermissionType", "get_current_user", "require_permission"
]
