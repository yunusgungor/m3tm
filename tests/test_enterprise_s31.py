"""
Test Suite for S31 - Enterprise Features & Governance
=====================================================

Comprehensive tests for enterprise governance, compliance, multi-tenancy, 
identity management, audit trails, and dashboard functionality.

Context7 Integration Tests:
- DataHub metadata management
- Keycloak authentication and authorization
- OPA policy enforcement
- Multi-tenant isolation
- Audit trail compliance
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# Import enterprise modules
from src.m3tm.enterprise.governance import (
    GovernanceEngine, DataAsset, LineageRelation, 
    DataClassification, LineageDirection
)
from src.m3tm.enterprise.compliance import (
    ComplianceManager, ComplianceRule, ComplianceFramework, 
    RiskLevel, ComplianceStatus
)
from src.m3tm.enterprise.multi_tenant import (
    TenantManager, Tenant, TenantStatus, IsolationLevel, 
    ResourceType, ResourceQuota
)
from src.m3tm.enterprise.identity import (
    IdentityManager, User, Role, Permission, 
    AuthenticationMethod, UserStatus
)
from src.m3tm.enterprise.audit import (
    AuditManager, AuditEvent, AuditEventType, 
    AuditSeverity, AuditQuery
)
from src.m3tm.enterprise.dashboard import (
    EnterpriseDashboard, Dashboard, DashboardWidget, 
    Metric, MetricType, DashboardType
)

class TestGovernanceEngine:
    """Test data governance and lineage management."""
    
    @pytest.fixture
    async def governance_engine(self):
        """Create governance engine for testing."""
        return GovernanceEngine(
            datahub_endpoint="http://localhost:8080",
            opa_endpoint="http://localhost:8181"
        )
    
    @pytest.mark.asyncio
    async def test_asset_registration(self, governance_engine):
        """Test data asset registration."""
        asset = DataAsset(
            asset_id="test_dataset_1",
            name="Test Dataset",
            description="Test dataset for governance",
            classification=DataClassification.CONFIDENTIAL,
            owner="data_team",
            tags=["test", "ml", "confidential"]
        )
        
        result = await governance_engine.register_asset(asset)
        assert result is True
        
        # Verify asset is stored
        retrieved_asset = await governance_engine.get_asset_metadata("test_dataset_1")
        assert retrieved_asset is not None
        assert retrieved_asset.name == "Test Dataset"
        assert retrieved_asset.classification == DataClassification.CONFIDENTIAL
        assert "test" in retrieved_asset.tags
    
    @pytest.mark.asyncio
    async def test_lineage_tracking(self, governance_engine):
        """Test data lineage tracking."""
        # Register source and target assets
        source_asset = DataAsset(
            asset_id="source_data",
            name="Source Data",
            classification=DataClassification.INTERNAL
        )
        target_asset = DataAsset(
            asset_id="processed_data", 
            name="Processed Data",
            classification=DataClassification.INTERNAL
        )
        
        await governance_engine.register_asset(source_asset)
        await governance_engine.register_asset(target_asset)
        
        # Create lineage relation
        relation = LineageRelation(
            source_asset="source_data",
            target_asset="processed_data",
            relation_type="DERIVES_FROM",
            transformation_info={"type": "data_processing", "script": "process.py"},
            confidence_score=0.95
        )
        
        result = await governance_engine.track_lineage(relation)
        assert result is True
        
        # Get lineage graph
        lineage = await governance_engine.get_lineage("processed_data", LineageDirection.UPSTREAM)
        assert lineage["asset_id"] == "processed_data"
        assert len(lineage["edges"]) > 0
        assert lineage["edges"][0]["source"] == "source_data"
        assert lineage["edges"][0]["target"] == "processed_data"
    
    @pytest.mark.asyncio
    async def test_policy_evaluation(self, governance_engine):
        """Test governance policy evaluation."""
        # Register restricted asset
        asset = DataAsset(
            asset_id="restricted_data",
            name="Restricted Dataset",
            classification=DataClassification.RESTRICTED,
            properties={"contains_pii": True}
        )
        await governance_engine.register_asset(asset)
        
        # Test policy evaluation
        result = await governance_engine.evaluate_policy("restricted_data", "read")
        assert result["allow"] is False
        assert "restricted" in result["reason"].lower()
        assert "required_permissions" in result
    
    @pytest.mark.asyncio
    async def test_asset_search(self, governance_engine):
        """Test asset search functionality."""
        # Register multiple assets
        assets = [
            DataAsset(
                asset_id="customer_data",
                name="Customer Database", 
                classification=DataClassification.CONFIDENTIAL,
                tags=["customer", "database"]
            ),
            DataAsset(
                asset_id="transaction_log",
                name="Transaction Logs",
                classification=DataClassification.INTERNAL,
                tags=["transaction", "log"]
            )
        ]
        
        for asset in assets:
            await governance_engine.register_asset(asset)
        
        # Search by query
        results = await governance_engine.search_assets("customer")
        assert len(results) == 1
        assert results[0].asset_id == "customer_data"
        
        # Search by classification
        results = await governance_engine.search_assets("", 
                                                       classification=DataClassification.CONFIDENTIAL)
        assert len(results) >= 1
        assert any(r.asset_id == "customer_data" for r in results)
        
        # Search by tags
        results = await governance_engine.search_assets("", tags=["database"])
        assert len(results) >= 1
        assert any(r.asset_id == "customer_data" for r in results)

class TestComplianceManager:
    """Test compliance management and monitoring."""
    
    @pytest.fixture
    async def compliance_manager(self):
        """Create compliance manager for testing."""
        return ComplianceManager(
            opa_endpoint="http://localhost:8181",
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.ISO27001]
        )
    
    @pytest.mark.asyncio
    async def test_compliance_rule_creation(self, compliance_manager):
        """Test custom compliance rule creation."""
        rule = ComplianceRule(
            rule_id="test_encryption_rule",
            name="Test Encryption Rule",
            description="Test rule requiring encryption",
            framework=ComplianceFramework.GDPR,
            policy_query="data.test.encryption_required",
            risk_level=RiskLevel.HIGH,
            remediation_steps=["Enable encryption", "Verify implementation"]
        )
        
        result = await compliance_manager.add_compliance_rule(rule)
        assert result is True
        
        # Verify rule is stored
        stored_rules = list(compliance_manager._rules.values())
        test_rule = next((r for r in stored_rules if r.rule_id == "test_encryption_rule"), None)
        assert test_rule is not None
        assert test_rule.framework == ComplianceFramework.GDPR
        assert test_rule.risk_level == RiskLevel.HIGH
    
    @pytest.mark.asyncio
    async def test_compliance_check_execution(self, compliance_manager):
        """Test compliance check execution."""
        # Test data with encryption disabled
        resource_data = {
            "encrypted": False,
            "contains_personal_data": True,
            "data_type": "customer_records"
        }
        
        # Run compliance checks
        results = await compliance_manager.run_compliance_check(
            resource_id="test_resource_1",
            resource_type="database",
            resource_data=resource_data
        )
        
        assert len(results) > 0
        
        # Check for GDPR encryption violation
        gdpr_checks = [r for r in results if "encryption" in compliance_manager._rules[r.rule_id].name.lower()]
        if gdpr_checks:
            assert gdpr_checks[0].status == ComplianceStatus.NON_COMPLIANT
            assert gdpr_checks[0].remediation_required is True
    
    @pytest.mark.asyncio
    async def test_compliance_status_reporting(self, compliance_manager):
        """Test compliance status reporting."""
        # Run some compliance checks first
        resource_data = {
            "encrypted": True,
            "rbac_enabled": True,
            "access_logging": True
        }
        
        await compliance_manager.run_compliance_check(
            resource_id="compliant_resource",
            resource_type="system",
            resource_data=resource_data
        )
        
        # Get compliance status
        status = await compliance_manager.get_compliance_status()
        
        assert "compliance_score" in status
        assert "total_checks" in status
        assert "compliant_checks" in status
        assert "non_compliant_checks" in status
        assert status["compliance_score"] >= 0
        assert status["compliance_score"] <= 100
    
    @pytest.mark.asyncio
    async def test_violation_resolution(self, compliance_manager):
        """Test compliance violation resolution."""
        # Create a violation by running a failing check
        resource_data = {
            "encrypted": False,
            "contains_personal_data": True
        }
        
        results = await compliance_manager.run_compliance_check(
            resource_id="violation_resource",
            resource_type="database", 
            resource_data=resource_data
        )
        
        # Check if violations were created
        violations = list(compliance_manager._violations.values())
        if violations:
            violation = violations[0]
            
            # Resolve the violation
            resolved = await compliance_manager.resolve_violation(
                violation.violation_id,
                "Encryption has been enabled",
                "security_admin"
            )
            
            assert resolved is True
            assert violation.resolved_at is not None
            assert violation.resolution_notes == "Encryption has been enabled"
            assert violation.assignee == "security_admin"

class TestTenantManager:
    """Test multi-tenant architecture and management."""
    
    @pytest.fixture
    async def tenant_manager(self):
        """Create tenant manager for testing."""
        return TenantManager(
            default_isolation_level=IsolationLevel.SHARED_SCHEMA
        )
    
    @pytest.mark.asyncio
    async def test_tenant_creation(self, tenant_manager):
        """Test tenant creation and provisioning."""
        tenant = await tenant_manager.create_tenant(
            tenant_name="Test Company",
            isolation_level=IsolationLevel.SHARED_SCHEMA,
            metadata={"industry": "technology", "region": "us-east"}
        )
        
        assert tenant.name == "Test Company"
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.isolation_level == IsolationLevel.SHARED_SCHEMA
        assert tenant.metadata["industry"] == "technology"
        
        # Check default quotas
        assert ResourceType.STORAGE in tenant.quotas
        assert ResourceType.API_CALLS in tenant.quotas
        assert tenant.quotas[ResourceType.STORAGE].limit == 10.0
    
    @pytest.mark.asyncio
    async def test_quota_management(self, tenant_manager):
        """Test resource quota management."""
        # Create tenant
        tenant = await tenant_manager.create_tenant("Quota Test Corp")
        
        # Update quota
        result = await tenant_manager.update_quota(
            tenant.tenant_id,
            ResourceType.STORAGE,
            50.0  # 50GB
        )
        assert result is True
        
        # Verify quota update
        updated_tenant = await tenant_manager.get_tenant(tenant.tenant_id)
        assert updated_tenant.quotas[ResourceType.STORAGE].limit == 50.0
        
        # Test quota consumption
        consumed = await tenant_manager.consume_quota(
            tenant.tenant_id,
            ResourceType.STORAGE,
            25.0  # 25GB
        )
        assert consumed is True
        assert updated_tenant.quotas[ResourceType.STORAGE].used == 25.0
        
        # Test quota exceeding
        exceeded = await tenant_manager.consume_quota(
            tenant.tenant_id,
            ResourceType.STORAGE,
            30.0  # Would exceed 50GB limit
        )
        assert exceeded is False
    
    @pytest.mark.asyncio
    async def test_tenant_status_management(self, tenant_manager):
        """Test tenant status updates."""
        tenant = await tenant_manager.create_tenant("Status Test Corp")
        
        # Update to suspended
        result = await tenant_manager.update_tenant_status(
            tenant.tenant_id,
            TenantStatus.SUSPENDED
        )
        assert result is True
        
        # Verify status change
        updated_tenant = await tenant_manager.get_tenant(tenant.tenant_id)
        assert updated_tenant.status == TenantStatus.SUSPENDED
        
        # Test quota consumption with suspended tenant
        consumed = await tenant_manager.consume_quota(
            tenant.tenant_id,
            ResourceType.API_CALLS,
            100
        )
        assert consumed is False  # Should fail for suspended tenant
    
    @pytest.mark.asyncio
    async def test_tenant_context_management(self, tenant_manager):
        """Test tenant context for request processing."""
        tenant = await tenant_manager.create_tenant("Context Test Corp")
        
        # Create tenant context
        context = tenant_manager.create_tenant_context(
            tenant_id=tenant.tenant_id,
            user_id="user123",
            roles=["data_scientist"],
            permissions={"model.read", "data.read"}
        )
        
        assert context.tenant_id == tenant.tenant_id
        assert context.user_id == "user123"
        assert "data_scientist" in context.roles
        assert "model.read" in context.permissions
        
        # Retrieve context
        retrieved_context = tenant_manager.get_tenant_context(context.request_id)
        assert retrieved_context is not None
        assert retrieved_context.tenant_id == tenant.tenant_id
        
        # Clear context
        tenant_manager.clear_tenant_context(context.request_id)
        cleared_context = tenant_manager.get_tenant_context(context.request_id)
        assert cleared_context is None

class TestIdentityManager:
    """Test identity and access management."""
    
    @pytest.fixture
    async def identity_manager(self):
        """Create identity manager for testing."""
        return IdentityManager(
            keycloak_url="http://localhost:8080",
            keycloak_realm="test-realm",
            keycloak_client_id="m3tm-test"
        )
    
    @pytest.mark.asyncio
    async def test_user_authentication(self, identity_manager):
        """Test user authentication."""
        # Test successful authentication
        result = await identity_manager.authenticate_user("admin", "admin")
        
        assert result.success is True
        assert result.user is not None
        assert result.user.username == "admin"
        assert result.access_token is not None
        assert result.method == AuthenticationMethod.PASSWORD
        
        # Test failed authentication
        failed_result = await identity_manager.authenticate_user("admin", "wrong_password")
        assert failed_result.success is False
        assert "Invalid credentials" in failed_result.error_message
    
    @pytest.mark.asyncio
    async def test_token_validation(self, identity_manager):
        """Test JWT token validation."""
        # Authenticate to get token
        auth_result = await identity_manager.authenticate_user("admin", "admin")
        assert auth_result.success is True
        
        token = auth_result.access_token
        
        # Validate token
        validation_result = await identity_manager.validate_token(token)
        assert validation_result.success is True
        assert validation_result.user is not None
        assert validation_result.user.username == "admin"
    
    @pytest.mark.asyncio
    async def test_user_authorization(self, identity_manager):
        """Test user authorization."""
        # Create user with specific roles
        user = await identity_manager.create_user(
            username="test_user",
            email="test@example.com",
            password="password123",
            roles={"data_scientist"}
        )
        
        # Test authorization for allowed action
        auth_result = await identity_manager.authorize_user(
            user, "model", "read"
        )
        assert auth_result.allowed is True
        assert len(auth_result.matched_permissions) > 0
        
        # Test authorization for denied action
        denied_result = await identity_manager.authorize_user(
            user, "system", "admin"
        )
        assert denied_result.allowed is False
    
    @pytest.mark.asyncio
    async def test_role_management(self, identity_manager):
        """Test role creation and management."""
        # Create custom role
        role = await identity_manager.create_role(
            role_id="test_role",
            name="Test Role",
            description="Test role for unit testing",
            permissions=[("data", "read"), ("model", "execute")]
        )
        
        assert role.role_id == "test_role"
        assert role.name == "Test Role"
        assert role.has_permission("data", "read") is True
        assert role.has_permission("model", "execute") is True
        assert role.has_permission("system", "admin") is False
        
        # Create user with custom role
        user = await identity_manager.create_user(
            username="role_test_user",
            email="roletest@example.com", 
            password="password123",
            roles={"test_role"}
        )
        
        # Test permissions from custom role
        permissions = await identity_manager.get_user_permissions(user)
        assert "data:read" in permissions or any("data" in p and "read" in p for p in permissions)

class TestAuditManager:
    """Test audit trail management."""
    
    @pytest.fixture
    async def audit_manager(self):
        """Create audit manager for testing."""
        return AuditManager(
            elasticsearch_endpoint="http://localhost:9200",
            retention_days=30
        )
    
    @pytest.mark.asyncio
    async def test_audit_event_logging(self, audit_manager):
        """Test audit event logging."""
        event = await audit_manager.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            description="User accessed customer data",
            severity=AuditSeverity.INFO,
            user_id="user123",
            tenant_id="tenant_1",
            resource_type="database",
            resource_id="customer_db",
            action="read",
            details={"query": "SELECT * FROM customers LIMIT 10"},
            source_ip="192.168.1.100",
            tags=["data_access", "customer_data"]
        )
        
        assert event.event_id is not None
        assert event.event_type == AuditEventType.DATA_ACCESS
        assert event.user_id == "user123"
        assert event.resource_type == "database"
        assert "customer_data" in event.tags
        
        # Verify event is stored
        retrieved_event = await audit_manager.get_event(event.event_id)
        assert retrieved_event is not None
        assert retrieved_event.description == "User accessed customer data"
    
    @pytest.mark.asyncio
    async def test_audit_event_querying(self, audit_manager):
        """Test audit event querying."""
        # Log multiple events
        events = []
        for i in range(5):
            event = await audit_manager.log_event(
                event_type=AuditEventType.AUTHENTICATION,
                description=f"User login attempt {i}",
                severity=AuditSeverity.INFO,
                user_id=f"user{i}",
                tenant_id="tenant_1"
            )
            events.append(event)
        
        # Query all events
        query = AuditQuery(limit=10)
        results = await audit_manager.query_events(query)
        assert len(results) >= 5
        
        # Query by event type
        auth_query = AuditQuery(
            event_types=[AuditEventType.AUTHENTICATION],
            limit=10
        )
        auth_results = await audit_manager.query_events(auth_query)
        assert len(auth_results) >= 5
        assert all(r.event_type == AuditEventType.AUTHENTICATION for r in auth_results)
        
        # Query by user
        user_query = AuditQuery(
            user_id="user1",
            limit=10
        )
        user_results = await audit_manager.query_events(user_query)
        assert len(user_results) >= 1
        assert all(r.user_id == "user1" for r in user_results)
    
    @pytest.mark.asyncio
    async def test_audit_report_generation(self, audit_manager):
        """Test audit report generation."""
        # Log various events
        await audit_manager.log_authentication("user1", True)
        await audit_manager.log_authentication("user2", False)
        await audit_manager.log_data_access("user1", "database", "customers", "read")
        await audit_manager.log_policy_violation("user3", "unauthorized_access")
        
        # Generate report
        query = AuditQuery(
            start_date=datetime.now(timezone.utc) - timedelta(hours=1),
            end_date=datetime.now(timezone.utc),
            limit=100
        )
        
        report = await audit_manager.generate_audit_report(query)
        
        assert "summary" in report
        assert report["summary"]["total_events"] >= 4
        assert "hourly_distribution" in report
        assert "security_incidents" in report
        assert "sample_events" in report
        
        # Check compliance score calculation
        assert "compliance_score" in report["summary"]
        assert 0 <= report["summary"]["compliance_score"] <= 100

class TestEnterpriseDashboard:
    """Test enterprise dashboard and monitoring."""
    
    @pytest.fixture
    async def dashboard(self):
        """Create enterprise dashboard for testing."""
        return EnterpriseDashboard(
            prometheus_endpoint="http://localhost:9090",
            grafana_endpoint="http://localhost:3000"
        )
    
    @pytest.mark.asyncio
    async def test_metric_recording(self, dashboard):
        """Test metric recording and retrieval."""
        # Record metrics
        await dashboard.record_metric("test_counter", 10, MetricType.COUNTER)
        await dashboard.record_metric("test_gauge", 75.5, MetricType.GAUGE, unit="%")
        await dashboard.record_metric(
            "test_labeled_metric", 
            42, 
            MetricType.GAUGE,
            labels={"tenant": "test_tenant", "region": "us-east"}
        )
        
        # Retrieve metrics
        counter_values = await dashboard.get_metric_values("test_counter")
        assert len(counter_values) >= 1
        assert counter_values[0].value == 10
        assert counter_values[0].type == MetricType.COUNTER
        
        gauge_values = await dashboard.get_metric_values("test_gauge")
        assert len(gauge_values) >= 1
        assert gauge_values[0].value == 75.5
        assert gauge_values[0].unit == "%"
        
        # Retrieve with label filter
        labeled_values = await dashboard.get_metric_values(
            "test_labeled_metric",
            labels={"tenant": "test_tenant"}
        )
        assert len(labeled_values) >= 1
        assert labeled_values[0].labels["tenant"] == "test_tenant"
    
    @pytest.mark.asyncio
    async def test_dashboard_creation(self, dashboard):
        """Test dashboard creation and management."""
        # Create custom dashboard
        custom_dashboard = Dashboard(
            dashboard_id="test_dashboard",
            title="Test Dashboard",
            description="Dashboard for testing",
            dashboard_type=DashboardType.PERFORMANCE,
            widgets=[
                DashboardWidget(
                    widget_id="cpu_usage",
                    title="CPU Usage",
                    type="gauge",
                    metric_queries=["m3tm_cpu_usage_percent"],
                    position={"x": 0, "y": 0, "width": 6, "height": 4}
                )
            ]
        )
        
        result = await dashboard.create_dashboard(custom_dashboard)
        assert result is True
        
        # Retrieve dashboard
        retrieved = await dashboard.get_dashboard("test_dashboard")
        assert retrieved is not None
        assert retrieved.title == "Test Dashboard"
        assert len(retrieved.widgets) == 1
        assert retrieved.widgets[0].widget_id == "cpu_usage"
    
    @pytest.mark.asyncio 
    async def test_dashboard_data_retrieval(self, dashboard):
        """Test dashboard data retrieval with metrics."""
        # Record some metrics first
        await dashboard.record_metric("m3tm_cpu_usage_percent", 45.2, MetricType.GAUGE)
        await dashboard.record_metric("m3tm_memory_usage_percent", 62.8, MetricType.GAUGE)
        
        # Get default overview dashboard data
        dashboard_data = await dashboard.get_dashboard_data("enterprise_overview")
        
        assert "dashboard" in dashboard_data
        assert "data" in dashboard_data
        assert "last_updated" in dashboard_data
        assert dashboard_data["dashboard"]["dashboard_id"] == "enterprise_overview"
    
    @pytest.mark.asyncio
    async def test_health_status_monitoring(self, dashboard):
        """Test enterprise health status monitoring."""
        # Record some health metrics
        await dashboard.record_metric("m3tm_cpu_usage_percent", 25.0, MetricType.GAUGE)
        await dashboard.record_metric("m3tm_memory_usage_percent", 60.0, MetricType.GAUGE)
        await dashboard.record_metric("m3tm_disk_usage_percent", 45.0, MetricType.GAUGE)
        await dashboard.record_metric("m3tm_compliance_score", 92.5, MetricType.GAUGE)
        
        # Get health status
        health_status = await dashboard.get_enterprise_health_status()
        
        assert "overall_status" in health_status
        assert "overall_score" in health_status
        assert "health_indicators" in health_status
        
        # Check individual health indicators
        assert "system" in health_status["health_indicators"]
        assert "compliance" in health_status["health_indicators"]
        assert "security" in health_status["health_indicators"]
        
        # System should be healthy with these values
        assert health_status["health_indicators"]["system"]["status"] == "healthy"
        assert health_status["health_indicators"]["compliance"]["status"] == "healthy"

class TestEnterpriseIntegration:
    """Integration tests for enterprise components."""
    
    @pytest.fixture
    async def enterprise_system(self):
        """Create integrated enterprise system for testing."""
        return {
            "governance": GovernanceEngine(),
            "compliance": ComplianceManager(),
            "tenant_manager": TenantManager(),
            "identity": IdentityManager(),
            "audit": AuditManager(),
            "dashboard": EnterpriseDashboard()
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, enterprise_system):
        """Test end-to-end enterprise workflow."""
        governance = enterprise_system["governance"]
        compliance = enterprise_system["compliance"]
        tenant_manager = enterprise_system["tenant_manager"]
        identity = enterprise_system["identity"]
        audit = enterprise_system["audit"]
        dashboard = enterprise_system["dashboard"]
        
        # 1. Create tenant
        tenant = await tenant_manager.create_tenant("Integration Test Corp")
        
        # 2. Create user for tenant
        user = await identity.create_user(
            username="integration_user",
            email="integration@test.com",
            password="password123",
            tenant_id=tenant.tenant_id,
            roles={"data_scientist"}
        )
        
        # 3. Register data asset
        asset = DataAsset(
            asset_id="integration_dataset",
            name="Integration Test Dataset",
            classification=DataClassification.INTERNAL,
            owner=user.user_id,
            properties={"tenant_id": tenant.tenant_id}
        )
        await governance.register_asset(asset)
        
        # 4. Log audit events
        await audit.log_authentication(user.user_id, True)
        await audit.log_data_access(
            user.user_id, 
            "dataset", 
            asset.asset_id, 
            "read",
            tenant_id=tenant.tenant_id
        )
        
        # 5. Run compliance checks
        resource_data = {
            "encrypted": True,
            "contains_personal_data": False,
            "tenant_id": tenant.tenant_id
        }
        compliance_results = await compliance.run_compliance_check(
            resource_id=asset.asset_id,
            resource_type="dataset",
            resource_data=resource_data
        )
        
        # 6. Record metrics
        await dashboard.record_metric(
            "integration_test_completed", 
            1, 
            MetricType.COUNTER,
            labels={"tenant": tenant.tenant_id}
        )
        
        # 7. Verify integration
        assert tenant.status == TenantStatus.ACTIVE
        assert user.tenant_id == tenant.tenant_id
        
        governance_asset = await governance.get_asset_metadata(asset.asset_id)
        assert governance_asset is not None
        
        audit_query = AuditQuery(user_id=user.user_id, limit=10)
        audit_events = await audit.query_events(audit_query)
        assert len(audit_events) >= 2  # Auth + data access events
        
        assert len(compliance_results) > 0
        
        metric_values = await dashboard.get_metric_values("integration_test_completed")
        assert len(metric_values) >= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
