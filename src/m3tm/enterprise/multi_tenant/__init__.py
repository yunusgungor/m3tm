"""
Multi-Tenant Architecture Manager
=================================

Enterprise multi-tenancy with data isolation, resource quotas, and tenant management.

Context7 References:
- /keycloak/keycloak: Multi-realm identity management
- /tiangolo/fastapi: Tenant-aware API endpoints
- /sqlalchemy/sqlalchemy: Multi-tenant database patterns
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import uuid

logger = logging.getLogger(__name__)

class TenantStatus(Enum):
    """Tenant status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    DEACTIVATED = "deactivated"

class IsolationLevel(Enum):
    """Data isolation levels"""
    SHARED_DB = "shared_database"          # Schema-level isolation
    DEDICATED_DB = "dedicated_database"    # Database-level isolation
    SHARED_SCHEMA = "shared_schema"        # Row-level isolation
    HYBRID = "hybrid"                      # Mixed approach

class ResourceType(Enum):
    """Resource types for quota management"""
    STORAGE = "storage"
    COMPUTE = "compute"
    API_CALLS = "api_calls"
    USERS = "users"
    MODELS = "models"
    EMBEDDINGS = "embeddings"

@dataclass
class ResourceQuota:
    """Resource quota definition"""
    resource_type: ResourceType
    limit: float
    used: float = 0.0
    unit: str = "count"
    period: Optional[str] = None  # e.g., "monthly", "daily"
    
    @property
    def usage_percentage(self) -> float:
        return (self.used / self.limit * 100) if self.limit > 0 else 0.0
    
    @property
    def is_exceeded(self) -> bool:
        return self.used >= self.limit

@dataclass
class Tenant:
    """Tenant configuration"""
    tenant_id: str
    name: str
    status: TenantStatus = TenantStatus.ACTIVE
    isolation_level: IsolationLevel = IsolationLevel.SHARED_SCHEMA
    quotas: Dict[ResourceType, ResourceQuota] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    keycloak_realm: Optional[str] = None
    database_config: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_quota(self, resource_type: ResourceType) -> Optional[ResourceQuota]:
        """Get quota for specific resource type."""
        return self.quotas.get(resource_type)
    
    def check_quota(self, resource_type: ResourceType, amount: float = 1.0) -> bool:
        """Check if resource usage would exceed quota."""
        quota = self.get_quota(resource_type)
        if not quota:
            return True  # No quota means unlimited
        return (quota.used + amount) <= quota.limit

@dataclass
class TenantContext:
    """Current tenant context for request processing"""
    tenant_id: str
    user_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: Set[str] = field(default_factory=set)
    request_id: Optional[str] = None
    
class TenantManager:
    """
    Multi-tenant architecture manager.
    
    Features:
    - Tenant provisioning and management
    - Data isolation strategies
    - Resource quota management
    - Keycloak realm integration
    - Cross-tenant security
    
    Context7 Pattern:
    - Integrates with Keycloak for multi-realm identity management
    - Uses FastAPI dependency injection for tenant context
    - Implements database isolation patterns with SQLAlchemy
    """
    
    def __init__(self,
                 default_isolation_level: IsolationLevel = IsolationLevel.SHARED_SCHEMA,
                 keycloak_admin_endpoint: Optional[str] = None,
                 keycloak_admin_token: Optional[str] = None):
        """
        Initialize tenant manager.
        
        Args:
            default_isolation_level: Default data isolation strategy
            keycloak_admin_endpoint: Keycloak admin API endpoint
            keycloak_admin_token: Admin token for Keycloak operations
        """
        self.default_isolation_level = default_isolation_level
        self.keycloak_admin_endpoint = keycloak_admin_endpoint
        self.keycloak_admin_token = keycloak_admin_token
        
        # Tenant storage
        self._tenants: Dict[str, Tenant] = {}
        self._tenant_contexts: Dict[str, TenantContext] = {}  # Request ID -> Context
        
        # Default quotas for new tenants
        self.default_quotas = {
            ResourceType.STORAGE: ResourceQuota(ResourceType.STORAGE, 10.0, unit="GB"),
            ResourceType.API_CALLS: ResourceQuota(ResourceType.API_CALLS, 10000.0, unit="calls", period="monthly"),
            ResourceType.USERS: ResourceQuota(ResourceType.USERS, 100.0, unit="users"),
            ResourceType.MODELS: ResourceQuota(ResourceType.MODELS, 10.0, unit="models"),
            ResourceType.EMBEDDINGS: ResourceQuota(ResourceType.EMBEDDINGS, 1000000.0, unit="embeddings", period="monthly")
        }
        
        logger.info(f"TenantManager initialized with isolation level: {default_isolation_level.value}")
    
    async def create_tenant(self, tenant_name: str, 
                          isolation_level: Optional[IsolationLevel] = None,
                          custom_quotas: Optional[Dict[ResourceType, ResourceQuota]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Tenant:
        """
        Create a new tenant with provisioning.
        
        Context7 Pattern: Creates Keycloak realm and database resources
        """
        try:
            # Generate tenant ID
            tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"
            
            # Set up isolation level
            if not isolation_level:
                isolation_level = self.default_isolation_level
            
            # Set up quotas
            quotas = self.default_quotas.copy()
            if custom_quotas:
                quotas.update(custom_quotas)
            
            # Create Keycloak realm if configured
            keycloak_realm = None
            if self.keycloak_admin_endpoint:
                keycloak_realm = await self._create_keycloak_realm(tenant_id, tenant_name)
            
            # Set up database configuration based on isolation level
            database_config = await self._setup_database_isolation(tenant_id, isolation_level)
            
            # Create tenant object
            tenant = Tenant(
                tenant_id=tenant_id,
                name=tenant_name,
                status=TenantStatus.ACTIVE,
                isolation_level=isolation_level,
                quotas=quotas,
                metadata=metadata or {},
                keycloak_realm=keycloak_realm,
                database_config=database_config
            )
            
            # Store tenant
            self._tenants[tenant_id] = tenant
            
            # Emit tenant creation event
            await self._emit_tenant_event("tenant_created", {
                "tenant_id": tenant_id,
                "name": tenant_name,
                "isolation_level": isolation_level.value
            })
            
            logger.info(f"Tenant created: {tenant_id} ({tenant_name})")
            return tenant
            
        except Exception as e:
            logger.error(f"Failed to create tenant {tenant_name}: {e}")
            raise
    
    async def _create_keycloak_realm(self, tenant_id: str, tenant_name: str) -> str:
        """
        Create Keycloak realm for tenant.
        
        Context7 Pattern: Uses Keycloak Admin REST API
        """
        try:
            realm_name = f"m3tm-{tenant_id}"
            
            # TODO: Integrate with actual Keycloak Admin API
            realm_config = {
                "realm": realm_name,
                "displayName": f"M³TM - {tenant_name}",
                "enabled": True,
                "registrationAllowed": False,
                "loginWithEmailAllowed": True,
                "duplicateEmailsAllowed": False,
                "resetPasswordAllowed": True,
                "editUsernameAllowed": False,
                "bruteForceProtected": True,
                "permanentLockout": False,
                "maxFailureWaitSeconds": 900,
                "minimumQuickLoginWaitSeconds": 60,
                "waitIncrementSeconds": 60,
                "quickLoginCheckMilliSeconds": 1000,
                "maxDeltaTimeSeconds": 43200,
                "failureFactor": 30
            }
            
            # Mock realm creation
            logger.info(f"Keycloak realm created: {realm_name}")
            return realm_name
            
        except Exception as e:
            logger.error(f"Failed to create Keycloak realm for tenant {tenant_id}: {e}")
            raise
    
    async def _setup_database_isolation(self, tenant_id: str, 
                                      isolation_level: IsolationLevel) -> Dict[str, Any]:
        """Set up database isolation based on strategy."""
        try:
            config = {"isolation_level": isolation_level.value}
            
            if isolation_level == IsolationLevel.DEDICATED_DB:
                # Create dedicated database
                config.update({
                    "database_name": f"m3tm_{tenant_id}",
                    "connection_string": f"postgresql://user:pass@localhost/m3tm_{tenant_id}",
                    "schema": "public"
                })
                
            elif isolation_level == IsolationLevel.SHARED_DB:
                # Use tenant-specific schema in shared database
                config.update({
                    "database_name": "m3tm_shared",
                    "connection_string": "postgresql://user:pass@localhost/m3tm_shared",
                    "schema": f"tenant_{tenant_id}"
                })
                
            elif isolation_level == IsolationLevel.SHARED_SCHEMA:
                # Row-level isolation with tenant_id column
                config.update({
                    "database_name": "m3tm_shared",
                    "connection_string": "postgresql://user:pass@localhost/m3tm_shared",
                    "schema": "public",
                    "row_level_security": True,
                    "tenant_column": "tenant_id"
                })
            
            logger.info(f"Database isolation configured for tenant {tenant_id}: {isolation_level.value}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to setup database isolation for tenant {tenant_id}: {e}")
            return {"error": str(e)}
    
    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self._tenants.get(tenant_id)
    
    async def list_tenants(self, status: Optional[TenantStatus] = None) -> List[Tenant]:
        """List tenants, optionally filtered by status."""
        tenants = list(self._tenants.values())
        if status:
            tenants = [t for t in tenants if t.status == status]
        return tenants
    
    async def update_tenant_status(self, tenant_id: str, status: TenantStatus) -> bool:
        """Update tenant status."""
        try:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                return False
            
            old_status = tenant.status
            tenant.status = status
            tenant.updated_at = datetime.now(timezone.utc)
            
            await self._emit_tenant_event("tenant_status_changed", {
                "tenant_id": tenant_id,
                "old_status": old_status.value,
                "new_status": status.value
            })
            
            logger.info(f"Tenant {tenant_id} status changed: {old_status.value} -> {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update tenant {tenant_id} status: {e}")
            return False
    
    async def update_quota(self, tenant_id: str, resource_type: ResourceType,
                         new_limit: float) -> bool:
        """Update resource quota for tenant."""
        try:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                return False
            
            if resource_type in tenant.quotas:
                old_limit = tenant.quotas[resource_type].limit
                tenant.quotas[resource_type].limit = new_limit
            else:
                old_limit = 0
                tenant.quotas[resource_type] = ResourceQuota(resource_type, new_limit)
            
            tenant.updated_at = datetime.now(timezone.utc)
            
            await self._emit_tenant_event("quota_updated", {
                "tenant_id": tenant_id,
                "resource_type": resource_type.value,
                "old_limit": old_limit,
                "new_limit": new_limit
            })
            
            logger.info(f"Quota updated for tenant {tenant_id}: {resource_type.value} = {new_limit}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update quota for tenant {tenant_id}: {e}")
            return False
    
    async def consume_quota(self, tenant_id: str, resource_type: ResourceType,
                          amount: float = 1.0) -> bool:
        """
        Consume resource quota and check limits.
        
        Returns True if consumption is allowed, False if quota exceeded.
        """
        try:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                logger.warning(f"Tenant not found: {tenant_id}")
                return False
            
            # Check if tenant is active
            if tenant.status != TenantStatus.ACTIVE:
                logger.warning(f"Tenant {tenant_id} is not active: {tenant.status.value}")
                return False
            
            # Get quota
            quota = tenant.get_quota(resource_type)
            if not quota:
                # No quota means unlimited
                return True
            
            # Check if consumption would exceed limit
            if quota.used + amount > quota.limit:
                logger.warning(f"Quota exceeded for tenant {tenant_id}: {resource_type.value}")
                await self._emit_tenant_event("quota_exceeded", {
                    "tenant_id": tenant_id,
                    "resource_type": resource_type.value,
                    "used": quota.used,
                    "limit": quota.limit,
                    "requested": amount
                })
                return False
            
            # Consume quota
            quota.used += amount
            tenant.updated_at = datetime.now(timezone.utc)
            
            # Log high usage
            if quota.usage_percentage > 80:
                await self._emit_tenant_event("quota_warning", {
                    "tenant_id": tenant_id,
                    "resource_type": resource_type.value,
                    "usage_percentage": quota.usage_percentage,
                    "used": quota.used,
                    "limit": quota.limit
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to consume quota for tenant {tenant_id}: {e}")
            return False
    
    def create_tenant_context(self, tenant_id: str, user_id: Optional[str] = None,
                            roles: Optional[List[str]] = None,
                            permissions: Optional[Set[str]] = None,
                            request_id: Optional[str] = None) -> TenantContext:
        """
        Create tenant context for request processing.
        
        Context7 Pattern: Used with FastAPI dependency injection
        """
        if not request_id:
            request_id = str(uuid.uuid4())
        
        context = TenantContext(
            tenant_id=tenant_id,
            user_id=user_id,
            roles=roles or [],
            permissions=permissions or set(),
            request_id=request_id
        )
        
        self._tenant_contexts[request_id] = context
        return context
    
    def get_tenant_context(self, request_id: str) -> Optional[TenantContext]:
        """Get tenant context by request ID."""
        return self._tenant_contexts.get(request_id)
    
    def clear_tenant_context(self, request_id: str):
        """Clear tenant context after request completion."""
        self._tenant_contexts.pop(request_id, None)
    
    async def get_tenant_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant usage metrics and statistics."""
        try:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                return {"error": "Tenant not found"}
            
            # Calculate quota usage
            quota_metrics = {}
            for resource_type, quota in tenant.quotas.items():
                quota_metrics[resource_type.value] = {
                    "used": quota.used,
                    "limit": quota.limit,
                    "usage_percentage": quota.usage_percentage,
                    "unit": quota.unit,
                    "period": quota.period,
                    "is_exceeded": quota.is_exceeded
                }
            
            return {
                "tenant_id": tenant_id,
                "name": tenant.name,
                "status": tenant.status.value,
                "isolation_level": tenant.isolation_level.value,
                "quotas": quota_metrics,
                "created_at": tenant.created_at.isoformat(),
                "updated_at": tenant.updated_at.isoformat(),
                "active_days": (datetime.now(timezone.utc) - tenant.created_at).days
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics for tenant {tenant_id}: {e}")
            return {"error": str(e)}
    
    async def get_multi_tenant_overview(self) -> Dict[str, Any]:
        """Get overview of all tenants and their status."""
        try:
            tenants = list(self._tenants.values())
            
            # Status distribution
            status_counts = {}
            for tenant in tenants:
                status = tenant.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Isolation level distribution
            isolation_counts = {}
            for tenant in tenants:
                isolation = tenant.isolation_level.value
                isolation_counts[isolation] = isolation_counts.get(isolation, 0) + 1
            
            # Resource usage summary
            total_usage = {}
            for tenant in tenants:
                for resource_type, quota in tenant.quotas.items():
                    resource_key = resource_type.value
                    if resource_key not in total_usage:
                        total_usage[resource_key] = {"used": 0, "limit": 0, "tenants": 0}
                    total_usage[resource_key]["used"] += quota.used
                    total_usage[resource_key]["limit"] += quota.limit
                    total_usage[resource_key]["tenants"] += 1
            
            return {
                "total_tenants": len(tenants),
                "status_distribution": status_counts,
                "isolation_distribution": isolation_counts,
                "resource_usage_summary": total_usage,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get multi-tenant overview: {e}")
            return {"error": str(e)}
    
    async def _emit_tenant_event(self, event_type: str, payload: Dict[str, Any]):
        """Emit tenant management event for monitoring."""
        try:
            event = {
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
                "source": "m3tm-tenant-manager"
            }
            
            # TODO: Send to event stream (Kafka, etc.)
            logger.debug(f"Tenant event: {json.dumps(event)}")
            
        except Exception as e:
            logger.error(f"Failed to emit tenant event: {e}")

# FastAPI dependency for tenant context injection
def get_tenant_context(request_id: str, tenant_manager: TenantManager) -> Optional[TenantContext]:
    """
    FastAPI dependency to get current tenant context.
    
    Context7 Pattern: Dependency injection for multi-tenant APIs
    """
    return tenant_manager.get_tenant_context(request_id)

# Export main classes
__all__ = [
    "TenantManager", "Tenant", "TenantContext", "ResourceQuota",
    "TenantStatus", "IsolationLevel", "ResourceType", "get_tenant_context"
]
