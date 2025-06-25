"""
Compliance Management System
============================

Enterprise compliance monitoring, reporting, and audit trail management.

Context7 References:
- /open-policy-agent/opa: Policy-based compliance checks
- /elastic/elasticsearch: Compliance event storage and search
- /grafana/grafana: Compliance dashboards and alerting
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

logger = logging.getLogger(__name__)

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    CUSTOM = "custom"

class ComplianceStatus(Enum):
    """Compliance check status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING = "pending"
    EXEMPTED = "exempted"
    UNKNOWN = "unknown"

class RiskLevel(Enum):
    """Risk level assessment"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ComplianceRule:
    """Compliance rule definition"""
    rule_id: str
    name: str
    description: str
    framework: ComplianceFramework
    policy_query: str  # OPA Rego query
    risk_level: RiskLevel = RiskLevel.MEDIUM
    auto_remediation: bool = False
    remediation_steps: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass 
class ComplianceCheck:
    """Compliance check result"""
    check_id: str
    rule_id: str
    resource_id: str
    resource_type: str
    status: ComplianceStatus
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    remediation_required: bool = False
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    evidence: Optional[Dict[str, Any]] = None

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    rule_id: str
    resource_id: str
    description: str
    risk_level: RiskLevel
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    assignee: Optional[str] = None

class ComplianceManager:
    """
    Enterprise compliance management system.
    
    Features:
    - Multi-framework compliance monitoring (GDPR, HIPAA, SOX, etc.)
    - Automated compliance checks using OPA policies
    - Violation tracking and remediation workflows
    - Compliance reporting and dashboards
    - Audit trail management
    
    Context7 Pattern:
    - Uses OPA for policy-based compliance evaluation
    - Integrates with Elasticsearch for compliance event storage
    - Provides REST APIs for compliance management
    """
    
    def __init__(self,
                 opa_endpoint: str = "http://localhost:8181",
                 elasticsearch_endpoint: str = "http://localhost:9200",
                 compliance_frameworks: Optional[List[ComplianceFramework]] = None):
        """
        Initialize compliance manager.
        
        Args:
            opa_endpoint: Open Policy Agent endpoint for policy evaluation
            elasticsearch_endpoint: Elasticsearch endpoint for event storage
            compliance_frameworks: List of enabled compliance frameworks
        """
        self.opa_endpoint = opa_endpoint
        self.elasticsearch_endpoint = elasticsearch_endpoint
        self.enabled_frameworks = compliance_frameworks or [
            ComplianceFramework.GDPR,
            ComplianceFramework.ISO27001
        ]
        
        # Internal storage
        self._rules: Dict[str, ComplianceRule] = {}
        self._checks: Dict[str, ComplianceCheck] = {}
        self._violations: Dict[str, ComplianceViolation] = {}
        
        # Load default compliance rules
        asyncio.create_task(self._load_default_rules())
        
        logger.info(f"ComplianceManager initialized with frameworks: {[f.value for f in self.enabled_frameworks]}")
    
    async def _load_default_rules(self):
        """Load default compliance rules for enabled frameworks."""
        try:
            # GDPR Rules
            if ComplianceFramework.GDPR in self.enabled_frameworks:
                gdpr_rules = [
                    ComplianceRule(
                        rule_id="gdpr_data_encryption",
                        name="Data Encryption at Rest",
                        description="Personal data must be encrypted at rest",
                        framework=ComplianceFramework.GDPR,
                        policy_query="data.gdpr.encryption_required",
                        risk_level=RiskLevel.HIGH,
                        remediation_steps=[
                            "Enable encryption for data storage",
                            "Verify encryption keys are properly managed",
                            "Document encryption implementation"
                        ],
                        tags=["gdpr", "encryption", "personal-data"]
                    ),
                    ComplianceRule(
                        rule_id="gdpr_data_retention",
                        name="Data Retention Policy",
                        description="Personal data retention must comply with GDPR limits",
                        framework=ComplianceFramework.GDPR,
                        policy_query="data.gdpr.retention_policy",
                        risk_level=RiskLevel.MEDIUM,
                        remediation_steps=[
                            "Review data retention periods",
                            "Implement automated data deletion",
                            "Update privacy policies"
                        ],
                        tags=["gdpr", "retention", "privacy"]
                    ),
                    ComplianceRule(
                        rule_id="gdpr_consent_tracking",
                        name="Consent Tracking",
                        description="User consent must be tracked and auditable",
                        framework=ComplianceFramework.GDPR,
                        policy_query="data.gdpr.consent_tracking",
                        risk_level=RiskLevel.HIGH,
                        remediation_steps=[
                            "Implement consent tracking system",
                            "Ensure consent withdrawal mechanisms",
                            "Maintain consent audit trail"
                        ],
                        tags=["gdpr", "consent", "audit"]
                    )
                ]
                
                for rule in gdpr_rules:
                    self._rules[rule.rule_id] = rule
            
            # ISO27001 Rules  
            if ComplianceFramework.ISO27001 in self.enabled_frameworks:
                iso_rules = [
                    ComplianceRule(
                        rule_id="iso27001_access_control",
                        name="Access Control Management",
                        description="Access to systems must be controlled and logged",
                        framework=ComplianceFramework.ISO27001,
                        policy_query="data.iso27001.access_control",
                        risk_level=RiskLevel.HIGH,
                        remediation_steps=[
                            "Implement role-based access control",
                            "Enable access logging",
                            "Regular access reviews"
                        ],
                        tags=["iso27001", "access-control", "security"]
                    ),
                    ComplianceRule(
                        rule_id="iso27001_vulnerability_mgmt",
                        name="Vulnerability Management",
                        description="Systems must be regularly scanned for vulnerabilities",
                        framework=ComplianceFramework.ISO27001,
                        policy_query="data.iso27001.vulnerability_scanning",
                        risk_level=RiskLevel.MEDIUM,
                        remediation_steps=[
                            "Schedule regular vulnerability scans",
                            "Implement patch management process",
                            "Document vulnerability remediation"
                        ],
                        tags=["iso27001", "vulnerability", "scanning"]
                    )
                ]
                
                for rule in iso_rules:
                    self._rules[rule.rule_id] = rule
            
            logger.info(f"Loaded {len(self._rules)} default compliance rules")
            
        except Exception as e:
            logger.error(f"Failed to load default compliance rules: {e}")
    
    async def add_compliance_rule(self, rule: ComplianceRule) -> bool:
        """Add a custom compliance rule."""
        try:
            self._rules[rule.rule_id] = rule
            
            await self._emit_compliance_event("rule_added", {
                "rule_id": rule.rule_id,
                "framework": rule.framework.value,
                "risk_level": rule.risk_level.value
            })
            
            logger.info(f"Compliance rule added: {rule.rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add compliance rule {rule.rule_id}: {e}")
            return False
    
    async def run_compliance_check(self, resource_id: str, resource_type: str,
                                 resource_data: Dict[str, Any],
                                 rule_ids: Optional[List[str]] = None) -> List[ComplianceCheck]:
        """
        Run compliance checks against a resource.
        
        Context7 Pattern: Uses OPA for policy evaluation
        """
        try:
            results = []
            rules_to_check = rule_ids or list(self._rules.keys())
            
            for rule_id in rules_to_check:
                if rule_id not in self._rules:
                    continue
                    
                rule = self._rules[rule_id]
                check_id = self._generate_check_id(rule_id, resource_id)
                
                # Prepare OPA input
                opa_input = {
                    "input": {
                        "resource": {
                            "id": resource_id,
                            "type": resource_type,
                            "data": resource_data
                        },
                        "rule": {
                            "id": rule_id,
                            "framework": rule.framework.value
                        }
                    }
                }
                
                # Evaluate policy (mock implementation)
                policy_result = await self._evaluate_policy(rule.policy_query, opa_input)
                
                # Create compliance check result
                status = ComplianceStatus.COMPLIANT if policy_result.get("compliant", False) else ComplianceStatus.NON_COMPLIANT
                
                check = ComplianceCheck(
                    check_id=check_id,
                    rule_id=rule_id,
                    resource_id=resource_id,
                    resource_type=resource_type,
                    status=status,
                    details=policy_result.get("details", {}),
                    risk_score=policy_result.get("risk_score", 0.0),
                    remediation_required=(status == ComplianceStatus.NON_COMPLIANT),
                    evidence=policy_result.get("evidence")
                )
                
                self._checks[check_id] = check
                results.append(check)
                
                # Create violation if non-compliant
                if status == ComplianceStatus.NON_COMPLIANT:
                    await self._create_violation(rule, resource_id, policy_result.get("details", {}))
            
            logger.info(f"Compliance checks completed for {resource_id}: {len(results)} checks")
            return results
            
        except Exception as e:
            logger.error(f"Compliance check failed for {resource_id}: {e}")
            return []
    
    async def _create_violation(self, rule: ComplianceRule, resource_id: str, details: Dict[str, Any]):
        """Create a compliance violation record."""
        try:
            violation_id = hashlib.md5(f"{rule.rule_id}:{resource_id}:{datetime.now().isoformat()}".encode()).hexdigest()
            
            violation = ComplianceViolation(
                violation_id=violation_id,
                rule_id=rule.rule_id,
                resource_id=resource_id,
                description=f"Violation of {rule.name}: {details.get('message', 'No details')}",
                risk_level=rule.risk_level
            )
            
            self._violations[violation_id] = violation
            
            await self._emit_compliance_event("violation_detected", {
                "violation_id": violation_id,
                "rule_id": rule.rule_id,
                "resource_id": resource_id,
                "risk_level": rule.risk_level.value
            })
            
            logger.warning(f"Compliance violation detected: {violation_id}")
            
        except Exception as e:
            logger.error(f"Failed to create violation: {e}")
    
    async def _evaluate_policy(self, policy_query: str, opa_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate compliance policy using OPA.
        
        Context7 Pattern: Uses OPA REST API
        """
        try:
            # TODO: Integrate with actual OPA API
            # For now, return mock policy evaluation based on common patterns
            
            resource_data = opa_input["input"]["resource"]["data"]
            
            # Mock GDPR encryption check
            if "gdpr.encryption_required" in policy_query:
                encrypted = resource_data.get("encrypted", False)
                has_personal_data = resource_data.get("contains_personal_data", False)
                
                return {
                    "compliant": not has_personal_data or encrypted,
                    "details": {
                        "message": "Personal data encryption check",
                        "encrypted": encrypted,
                        "contains_personal_data": has_personal_data
                    },
                    "risk_score": 8.0 if has_personal_data and not encrypted else 2.0
                }
            
            # Mock access control check
            if "access_control" in policy_query:
                has_rbac = resource_data.get("rbac_enabled", False)
                has_logging = resource_data.get("access_logging", False)
                
                return {
                    "compliant": has_rbac and has_logging,
                    "details": {
                        "message": "Access control compliance check",
                        "rbac_enabled": has_rbac,
                        "access_logging": has_logging
                    },
                    "risk_score": 6.0 if not (has_rbac and has_logging) else 1.0
                }
            
            # Default: assume compliant
            return {
                "compliant": True,
                "details": {"message": "Policy evaluation completed"},
                "risk_score": 1.0
            }
            
        except Exception as e:
            logger.error(f"Policy evaluation failed: {e}")
            return {
                "compliant": False,
                "details": {"message": f"Policy evaluation error: {e}"},
                "risk_score": 10.0
            }
    
    async def get_compliance_status(self, resource_id: Optional[str] = None,
                                  framework: Optional[ComplianceFramework] = None) -> Dict[str, Any]:
        """Get compliance status overview."""
        try:
            checks = list(self._checks.values())
            
            # Filter by resource if specified
            if resource_id:
                checks = [c for c in checks if c.resource_id == resource_id]
            
            # Filter by framework if specified
            if framework:
                rule_ids = [r.rule_id for r in self._rules.values() if r.framework == framework]
                checks = [c for c in checks if c.rule_id in rule_ids]
            
            total_checks = len(checks)
            compliant_checks = len([c for c in checks if c.status == ComplianceStatus.COMPLIANT])
            non_compliant_checks = len([c for c in checks if c.status == ComplianceStatus.NON_COMPLIANT])
            
            # Calculate compliance score
            compliance_score = (compliant_checks / total_checks * 100) if total_checks > 0 else 100
            
            # Risk assessment
            violations = list(self._violations.values())
            critical_violations = len([v for v in violations if v.risk_level == RiskLevel.CRITICAL and not v.resolved_at])
            high_violations = len([v for v in violations if v.risk_level == RiskLevel.HIGH and not v.resolved_at])
            
            return {
                "compliance_score": round(compliance_score, 2),
                "total_checks": total_checks,
                "compliant_checks": compliant_checks,
                "non_compliant_checks": non_compliant_checks,
                "open_violations": len([v for v in violations if not v.resolved_at]),
                "critical_violations": critical_violations,
                "high_violations": high_violations,
                "frameworks": [f.value for f in self.enabled_frameworks],
                "last_assessment": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get compliance status: {e}")
            return {"error": str(e)}
    
    async def resolve_violation(self, violation_id: str, resolution_notes: str,
                              resolver: str) -> bool:
        """Mark a compliance violation as resolved."""
        try:
            if violation_id not in self._violations:
                return False
            
            violation = self._violations[violation_id]
            violation.resolved_at = datetime.now(timezone.utc)
            violation.resolution_notes = resolution_notes
            violation.assignee = resolver
            
            await self._emit_compliance_event("violation_resolved", {
                "violation_id": violation_id,
                "resolver": resolver,
                "resolution_time": violation.resolved_at.isoformat()
            })
            
            logger.info(f"Compliance violation resolved: {violation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve violation {violation_id}: {e}")
            return False
    
    async def generate_compliance_report(self, framework: Optional[ComplianceFramework] = None,
                                       start_date: Optional[datetime] = None,
                                       end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        try:
            # Date range
            if not start_date:
                start_date = datetime.now(timezone.utc) - timedelta(days=30)
            if not end_date:
                end_date = datetime.now(timezone.utc)
            
            # Filter data by date range and framework
            checks = [c for c in self._checks.values() 
                     if start_date <= c.checked_at <= end_date]
            violations = [v for v in self._violations.values()
                         if start_date <= v.detected_at <= end_date]
            
            if framework:
                framework_rules = [r.rule_id for r in self._rules.values() if r.framework == framework]
                checks = [c for c in checks if c.rule_id in framework_rules]
                violations = [v for v in violations if v.rule_id in framework_rules]
            
            # Generate report
            report = {
                "report_metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "period": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    },
                    "framework": framework.value if framework else "all",
                    "total_rules": len(self._rules),
                    "total_checks": len(checks),
                    "total_violations": len(violations)
                },
                "compliance_summary": await self.get_compliance_status(framework=framework),
                "violations_by_risk": {
                    "critical": len([v for v in violations if v.risk_level == RiskLevel.CRITICAL]),
                    "high": len([v for v in violations if v.risk_level == RiskLevel.HIGH]),
                    "medium": len([v for v in violations if v.risk_level == RiskLevel.MEDIUM]),
                    "low": len([v for v in violations if v.risk_level == RiskLevel.LOW])
                },
                "top_violations": [
                    {
                        "rule_id": v.rule_id,
                        "resource_id": v.resource_id,
                        "risk_level": v.risk_level.value,
                        "detected_at": v.detected_at.isoformat(),
                        "resolved": v.resolved_at is not None
                    }
                    for v in sorted(violations, key=lambda x: x.detected_at, reverse=True)[:10]
                ],
                "remediation_recommendations": self._get_remediation_recommendations(violations)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            return {"error": str(e)}
    
    def _get_remediation_recommendations(self, violations: List[ComplianceViolation]) -> List[Dict[str, Any]]:
        """Generate remediation recommendations based on violations."""
        recommendations = []
        
        # Group violations by rule
        rule_violations = {}
        for violation in violations:
            if not violation.resolved_at:  # Only open violations
                rule_violations.setdefault(violation.rule_id, []).append(violation)
        
        # Generate recommendations
        for rule_id, rule_violations_list in rule_violations.items():
            if rule_id in self._rules:
                rule = self._rules[rule_id]
                recommendations.append({
                    "rule_id": rule_id,
                    "rule_name": rule.name,
                    "affected_resources": len(rule_violations_list),
                    "risk_level": rule.risk_level.value,
                    "remediation_steps": rule.remediation_steps,
                    "priority": len(rule_violations_list) * (4 - list(RiskLevel).index(rule.risk_level))
                })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _generate_check_id(self, rule_id: str, resource_id: str) -> str:
        """Generate unique check ID."""
        return hashlib.md5(f"{rule_id}:{resource_id}:{datetime.now().isoformat()}".encode()).hexdigest()
    
    async def _emit_compliance_event(self, event_type: str, payload: Dict[str, Any]):
        """Emit compliance event for monitoring and alerting."""
        try:
            event = {
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
                "source": "m3tm-compliance"
            }
            
            # TODO: Send to Elasticsearch or other event store
            logger.debug(f"Compliance event: {json.dumps(event)}")
            
        except Exception as e:
            logger.error(f"Failed to emit compliance event: {e}")

# Export main classes
__all__ = [
    "ComplianceManager", "ComplianceRule", "ComplianceCheck", "ComplianceViolation",
    "ComplianceFramework", "ComplianceStatus", "RiskLevel"
]
