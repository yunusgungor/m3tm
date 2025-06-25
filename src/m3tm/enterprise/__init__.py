"""
M³TM Enterprise Features & Governance
=====================================

Enterprise-grade governance, compliance, and multi-tenancy features for M³TM.

Implements:
- Data governance and lineage tracking
- Multi-tenant architecture with role-based access control
- Policy enforcement and compliance auditing
- Identity and access management integration
- Audit logging and reporting
- Data quality and security monitoring

Architecture:
- governance/: Data governance and lineage management
- compliance/: Compliance monitoring and reporting
- multi_tenant/: Multi-tenant architecture and isolation
- identity/: Identity and access management integration
- audit/: Audit logging and trail management
- dashboard/: Enterprise metrics and monitoring

Context7 References:
- DataHub: /datahub/datahub - Metadata management and lineage
- Keycloak: /keycloak/keycloak - Identity and access management
- OPA: /open-policy-agent/opa - Policy enforcement engine
- FastAPI: /tiangolo/fastapi - API framework for governance endpoints
"""

from .governance import GovernanceEngine
from .compliance import ComplianceManager
from .multi_tenant import TenantManager
from .identity import IdentityManager
from .audit import AuditManager
from .dashboard import EnterpriseDashboard

__version__ = "2.3.0"
__enterprise_version__ = "1.0.0"

__all__ = [
    "GovernanceEngine",
    "ComplianceManager", 
    "TenantManager",
    "IdentityManager",
    "AuditManager",
    "EnterpriseDashboard"
]

# Enterprise feature flags
ENTERPRISE_FEATURES = {
    "governance": True,
    "compliance": True,
    "multi_tenant": True,
    "identity_management": True,
    "audit_logging": True,
    "dashboard": True,
    "policy_enforcement": True,
    "data_lineage": True
}
