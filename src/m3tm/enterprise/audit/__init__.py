"""
Audit Trail Management
======================

Comprehensive audit logging, trail management, and compliance reporting.

Context7 References:
- /elastic/elasticsearch: Audit log storage and search
- /grafana/grafana: Audit dashboards and alerting
- /apache/kafka: Real-time audit event streaming
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import hashlib
import uuid

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Audit event types"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_CHANGE = "system_change"
    COMPLIANCE_CHECK = "compliance_check"
    POLICY_VIOLATION = "policy_violation"
    ADMIN_ACTION = "admin_action"
    USER_MANAGEMENT = "user_management"
    CONFIGURATION_CHANGE = "configuration_change"

class AuditSeverity(Enum):
    """Audit event severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditStatus(Enum):
    """Audit event processing status"""
    PENDING = "pending"
    PROCESSED = "processed"
    ARCHIVED = "archived"
    FAILED = "failed"

@dataclass
class AuditEvent:
    """Audit event record"""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    status: AuditStatus = AuditStatus.PENDING
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure timestamp has timezone info."""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        data["status"] = self.status.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

@dataclass
class AuditQuery:
    """Audit query parameters"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    severities: Optional[List[AuditSeverity]] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    search_text: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = 100
    offset: int = 0

@dataclass
class AuditSummary:
    """Audit trail summary"""
    total_events: int
    event_type_counts: Dict[str, int]
    severity_counts: Dict[str, int]
    top_users: List[Dict[str, Any]]
    top_resources: List[Dict[str, Any]]
    time_range: Dict[str, str]
    compliance_score: float

class AuditManager:
    """
    Comprehensive audit trail management system.
    
    Features:
    - Structured audit event logging
    - Real-time event streaming
    - Compliance reporting
    - Security monitoring
    - Data retention management
    - Search and analysis capabilities
    
    Context7 Pattern:
    - Uses Elasticsearch for scalable audit log storage
    - Integrates with Kafka for real-time event streaming
    - Provides REST APIs for audit management
    """
    
    def __init__(self,
                 elasticsearch_endpoint: str = "http://localhost:9200",
                 kafka_endpoint: Optional[str] = None,
                 retention_days: int = 2555,  # 7 years default
                 auto_archive: bool = True):
        """
        Initialize audit manager.
        
        Args:
            elasticsearch_endpoint: Elasticsearch endpoint for audit storage
            kafka_endpoint: Kafka endpoint for real-time streaming
            retention_days: Audit data retention period in days
            auto_archive: Enable automatic archiving of old audit data
        """
        self.elasticsearch_endpoint = elasticsearch_endpoint
        self.kafka_endpoint = kafka_endpoint
        self.retention_days = retention_days
        self.auto_archive = auto_archive
        
        # Internal storage (fallback when external systems unavailable)
        self._audit_events: List[AuditEvent] = []
        self._event_index: Dict[str, AuditEvent] = {}
        
        # Audit statistics
        self._stats = {
            "total_events": 0,
            "events_by_type": {},
            "events_by_severity": {},
            "last_event_time": None
        }
        
        # Start background tasks
        if self.auto_archive:
            asyncio.create_task(self._periodic_cleanup())
        
        logger.info(f"AuditManager initialized with retention: {retention_days} days")
    
    async def log_event(self, event_type: AuditEventType, description: str,
                       severity: AuditSeverity = AuditSeverity.INFO,
                       user_id: Optional[str] = None,
                       tenant_id: Optional[str] = None,
                       resource_type: Optional[str] = None,
                       resource_id: Optional[str] = None,
                       action: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None,
                       source_ip: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       session_id: Optional[str] = None,
                       request_id: Optional[str] = None,
                       tags: Optional[List[str]] = None) -> AuditEvent:
        """
        Log an audit event.
        
        Context7 Pattern: Creates structured audit events with full context
        """
        try:
            # Generate event ID
            event_id = str(uuid.uuid4())
            
            # Create audit event
            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                severity=severity,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                tenant_id=tenant_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                description=description,
                details=details or {},
                source_ip=source_ip,
                user_agent=user_agent,
                session_id=session_id,
                request_id=request_id,
                tags=tags or []
            )
            
            # Store event
            await self._store_event(event)
            
            # Stream event for real-time processing
            await self._stream_event(event)
            
            # Update statistics
            self._update_stats(event)
            
            logger.debug(f"Audit event logged: {event_id} ({event_type.value})")
            return event
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            raise
    
    async def _store_event(self, event: AuditEvent):
        """Store audit event in persistent storage."""
        try:
            # Add to internal storage
            self._audit_events.append(event)
            self._event_index[event.event_id] = event
            
            # TODO: Store in Elasticsearch
            # index_name = f"audit-{event.timestamp.strftime('%Y-%m')}"
            # await self._elasticsearch_client.index(
            #     index=index_name,
            #     body=event.to_dict()
            # )
            
            event.status = AuditStatus.PROCESSED
            
        except Exception as e:
            logger.error(f"Failed to store audit event {event.event_id}: {e}")
            event.status = AuditStatus.FAILED
            raise
    
    async def _stream_event(self, event: AuditEvent):
        """Stream audit event for real-time processing."""
        try:
            if not self.kafka_endpoint:
                return
            
            # TODO: Send to Kafka topic
            event_data = event.to_dict()
            
            # Mock Kafka streaming
            logger.debug(f"Audit event streamed: {json.dumps(event_data)}")
            
        except Exception as e:
            logger.error(f"Failed to stream audit event {event.event_id}: {e}")
    
    def _update_stats(self, event: AuditEvent):
        """Update audit statistics."""
        try:
            self._stats["total_events"] += 1
            
            # Event type counts
            event_type = event.event_type.value
            self._stats["events_by_type"][event_type] = \
                self._stats["events_by_type"].get(event_type, 0) + 1
            
            # Severity counts
            severity = event.severity.value
            self._stats["events_by_severity"][severity] = \
                self._stats["events_by_severity"].get(severity, 0) + 1
            
            # Last event time
            self._stats["last_event_time"] = event.timestamp.isoformat()
            
        except Exception as e:
            logger.error(f"Failed to update audit stats: {e}")
    
    async def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """
        Query audit events with filtering and pagination.
        
        Context7 Pattern: Uses Elasticsearch query DSL for complex searches
        """
        try:
            events = self._audit_events.copy()
            
            # Apply filters
            if query.start_date:
                events = [e for e in events if e.timestamp >= query.start_date]
            
            if query.end_date:
                events = [e for e in events if e.timestamp <= query.end_date]
            
            if query.event_types:
                events = [e for e in events if e.event_type in query.event_types]
            
            if query.severities:
                events = [e for e in events if e.severity in query.severities]
            
            if query.user_id:
                events = [e for e in events if e.user_id == query.user_id]
            
            if query.tenant_id:
                events = [e for e in events if e.tenant_id == query.tenant_id]
            
            if query.resource_type:
                events = [e for e in events if e.resource_type == query.resource_type]
            
            if query.resource_id:
                events = [e for e in events if e.resource_id == query.resource_id]
            
            if query.search_text:
                search_lower = query.search_text.lower()
                events = [e for e in events 
                         if search_lower in e.description.lower() or
                         search_lower in str(e.details).lower()]
            
            if query.tags:
                events = [e for e in events 
                         if any(tag in e.tags for tag in query.tags)]
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply pagination
            start_idx = query.offset
            end_idx = start_idx + query.limit
            events = events[start_idx:end_idx]
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")
            return []
    
    async def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get specific audit event by ID."""
        return self._event_index.get(event_id)
    
    async def generate_audit_report(self, query: AuditQuery) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        try:
            events = await self.query_events(query)
            
            if not events:
                return {
                    "summary": {"total_events": 0},
                    "events": [],
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }
            
            # Calculate summary statistics
            event_type_counts = {}
            severity_counts = {}
            user_activity = {}
            resource_activity = {}
            hourly_distribution = {}
            
            for event in events:
                # Event type distribution
                event_type = event.event_type.value
                event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
                
                # Severity distribution
                severity = event.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                # User activity
                if event.user_id:
                    user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
                
                # Resource activity
                if event.resource_type and event.resource_id:
                    resource_key = f"{event.resource_type}:{event.resource_id}"
                    resource_activity[resource_key] = resource_activity.get(resource_key, 0) + 1
                
                # Hourly distribution
                hour = event.timestamp.hour
                hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
            
            # Top users and resources
            top_users = sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
            top_resources = sorted(resource_activity.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(events)
            
            # Security incidents (high/critical events)
            security_incidents = [
                e for e in events 
                if e.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]
            ]
            
            summary = AuditSummary(
                total_events=len(events),
                event_type_counts=event_type_counts,
                severity_counts=severity_counts,
                top_users=[{"user_id": uid, "event_count": count} for uid, count in top_users],
                top_resources=[{"resource": res, "event_count": count} for res, count in top_resources],
                time_range={
                    "start": events[-1].timestamp.isoformat() if events else None,
                    "end": events[0].timestamp.isoformat() if events else None
                },
                compliance_score=compliance_score
            )
            
            return {
                "summary": asdict(summary),
                "hourly_distribution": hourly_distribution,
                "security_incidents": [e.to_dict() for e in security_incidents[:20]],
                "sample_events": [e.to_dict() for e in events[:10]],
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "query_parameters": {
                    "start_date": query.start_date.isoformat() if query.start_date else None,
                    "end_date": query.end_date.isoformat() if query.end_date else None,
                    "total_matched": len(events)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate audit report: {e}")
            return {"error": str(e)}
    
    def _calculate_compliance_score(self, events: List[AuditEvent]) -> float:
        """Calculate compliance score based on audit events."""
        if not events:
            return 100.0
        
        # Simple compliance scoring
        total_events = len(events)
        violations = len([e for e in events if e.event_type == AuditEventType.POLICY_VIOLATION])
        critical_events = len([e for e in events if e.severity == AuditSeverity.CRITICAL])
        
        # Penalty for violations and critical events
        penalty = (violations * 5 + critical_events * 10) / total_events * 100
        score = max(0, 100 - penalty)
        
        return round(score, 2)
    
    async def archive_old_events(self, cutoff_date: Optional[datetime] = None) -> int:
        """Archive old audit events based on retention policy."""
        try:
            if not cutoff_date:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            
            # Find events to archive
            events_to_archive = [
                e for e in self._audit_events 
                if e.timestamp < cutoff_date and e.status != AuditStatus.ARCHIVED
            ]
            
            if not events_to_archive:
                return 0
            
            # Mark as archived
            for event in events_to_archive:
                event.status = AuditStatus.ARCHIVED
            
            # TODO: Move to cold storage or delete from active storage
            
            logger.info(f"Archived {len(events_to_archive)} audit events")
            return len(events_to_archive)
            
        except Exception as e:
            logger.error(f"Failed to archive audit events: {e}")
            return 0
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task for audit data."""
        while True:
            try:
                await asyncio.sleep(24 * 3600)  # Run daily
                archived_count = await self.archive_old_events()
                if archived_count > 0:
                    logger.info(f"Daily cleanup: archived {archived_count} events")
                    
            except Exception as e:
                logger.error(f"Periodic cleanup failed: {e}")
    
    async def get_audit_metrics(self) -> Dict[str, Any]:
        """Get audit system metrics and health status."""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Recent activity (last 24 hours)
            recent_cutoff = current_time - timedelta(hours=24)
            recent_events = [e for e in self._audit_events if e.timestamp >= recent_cutoff]
            
            # Calculate event rates
            events_per_hour = len(recent_events) / 24 if recent_events else 0
            
            # Health indicators
            health_score = 100.0
            health_issues = []
            
            # Check for recent high-severity events
            critical_recent = [e for e in recent_events if e.severity == AuditSeverity.CRITICAL]
            if len(critical_recent) > 10:
                health_score -= 20
                health_issues.append(f"High number of critical events: {len(critical_recent)}")
            
            # Check for failed events
            failed_events = [e for e in self._audit_events if e.status == AuditStatus.FAILED]
            if len(failed_events) > 0:
                health_score -= 10
                health_issues.append(f"Failed audit events: {len(failed_events)}")
            
            return {
                "total_events": len(self._audit_events),
                "recent_events_24h": len(recent_events),
                "events_per_hour": round(events_per_hour, 2),
                "event_type_distribution": self._stats["events_by_type"],
                "severity_distribution": self._stats["events_by_severity"],
                "storage_health": {
                    "elasticsearch_available": False,  # TODO: Check actual status
                    "kafka_available": False  # TODO: Check actual status
                },
                "health_score": health_score,
                "health_issues": health_issues,
                "last_event_time": self._stats["last_event_time"],
                "retention_days": self.retention_days,
                "auto_archive_enabled": self.auto_archive,
                "generated_at": current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get audit metrics: {e}")
            return {"error": str(e)}
    
    # Convenience methods for common audit events
    
    async def log_authentication(self, user_id: str, success: bool, 
                               source_ip: Optional[str] = None,
                               details: Optional[Dict[str, Any]] = None):
        """Log authentication event."""
        severity = AuditSeverity.INFO if success else AuditSeverity.MEDIUM
        description = f"User authentication {'successful' if success else 'failed'}"
        
        return await self.log_event(
            event_type=AuditEventType.AUTHENTICATION,
            description=description,
            severity=severity,
            user_id=user_id,
            source_ip=source_ip,
            details=details or {"success": success}
        )
    
    async def log_data_access(self, user_id: str, resource_type: str, resource_id: str,
                            action: str, tenant_id: Optional[str] = None,
                            details: Optional[Dict[str, Any]] = None):
        """Log data access event."""
        description = f"Data access: {action} on {resource_type}:{resource_id}"
        
        return await self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            description=description,
            severity=AuditSeverity.INFO,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details
        )
    
    async def log_policy_violation(self, user_id: str, violation_type: str,
                                 resource_type: Optional[str] = None,
                                 resource_id: Optional[str] = None,
                                 details: Optional[Dict[str, Any]] = None):
        """Log policy violation event."""
        description = f"Policy violation: {violation_type}"
        
        return await self.log_event(
            event_type=AuditEventType.POLICY_VIOLATION,
            description=description,
            severity=AuditSeverity.HIGH,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            tags=["policy", "violation", "security"]
        )
    
    async def log_admin_action(self, user_id: str, action: str,
                             target_resource: Optional[str] = None,
                             details: Optional[Dict[str, Any]] = None):
        """Log administrative action."""
        description = f"Admin action: {action}"
        if target_resource:
            description += f" on {target_resource}"
        
        return await self.log_event(
            event_type=AuditEventType.ADMIN_ACTION,
            description=description,
            severity=AuditSeverity.MEDIUM,
            user_id=user_id,
            action=action,
            details=details,
            tags=["admin", "management"]
        )

# Export main classes
__all__ = [
    "AuditManager", "AuditEvent", "AuditQuery", "AuditSummary",
    "AuditEventType", "AuditSeverity", "AuditStatus"
]
