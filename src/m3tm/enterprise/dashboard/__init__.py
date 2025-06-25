"""
Enterprise Dashboard & Monitoring
=================================

Real-time enterprise metrics, monitoring dashboards, and alerting system.

Context7 References:
- /grafana/grafana: Dashboard and visualization platform
- /prometheus/prometheus: Metrics collection and monitoring
- /elastic/kibana: Log analytics and visualization
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import statistics

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class DashboardType(Enum):
    """Dashboard types"""
    OVERVIEW = "overview"
    GOVERNANCE = "governance"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USAGE = "usage"

@dataclass
class Metric:
    """Metric definition"""
    name: str
    type: MetricType
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: Optional[str] = None
    unit: Optional[str] = None
    
    def to_prometheus_format(self) -> str:
        """Convert to Prometheus metric format."""
        labels_str = ""
        if self.labels:
            label_pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
            labels_str = "{" + ",".join(label_pairs) + "}"
        
        return f"{self.name}{labels_str} {self.value} {int(self.timestamp.timestamp() * 1000)}"

@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str  # Alert condition expression
    threshold: Union[int, float]
    metric_name: str
    labels: Dict[str, str] = field(default_factory=dict)
    active: bool = True
    last_triggered: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    title: str
    type: str  # chart, gauge, table, stat, etc.
    metric_queries: List[str]
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)  # x, y, width, height

@dataclass
class Dashboard:
    """Dashboard configuration"""
    dashboard_id: str
    title: str
    description: str
    dashboard_type: DashboardType
    widgets: List[DashboardWidget] = field(default_factory=list)
    refresh_interval: int = 30  # seconds
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class EnterpriseDashboard:
    """
    Enterprise dashboard and monitoring system.
    
    Features:
    - Real-time metrics collection and visualization
    - Configurable dashboards and widgets
    - Alert management and notifications
    - Integration with Grafana and Prometheus
    - Multi-tenant metric isolation
    - Performance monitoring
    
    Context7 Pattern:
    - Integrates with Prometheus for metrics storage
    - Uses Grafana for dashboard visualization
    - Provides REST APIs for metrics and dashboard management
    """
    
    def __init__(self,
                 prometheus_endpoint: str = "http://localhost:9090",
                 grafana_endpoint: str = "http://localhost:3000",
                 grafana_api_key: Optional[str] = None):
        """
        Initialize enterprise dashboard system.
        
        Args:
            prometheus_endpoint: Prometheus server endpoint
            grafana_endpoint: Grafana server endpoint
            grafana_api_key: API key for Grafana operations
        """
        self.prometheus_endpoint = prometheus_endpoint
        self.grafana_endpoint = grafana_endpoint
        self.grafana_api_key = grafana_api_key
        
        # Internal storage
        self._metrics: Dict[str, List[Metric]] = {}
        self._alerts: Dict[str, Alert] = {}
        self._dashboards: Dict[str, Dashboard] = {}
        
        # Alert handlers
        self._alert_handlers: List[Callable] = []
        
        # Metric aggregation cache
        self._metric_cache: Dict[str, Any] = {}
        self._cache_ttl: timedelta = timedelta(minutes=5)
        
        # Initialize default dashboards and alerts
        asyncio.create_task(self._initialize_defaults())
        
        # Start background tasks
        asyncio.create_task(self._metrics_collector())
        asyncio.create_task(self._alert_evaluator())
        
        logger.info("EnterpriseDashboard initialized")
    
    async def _initialize_defaults(self):
        """Initialize default dashboards and alerts."""
        try:
            # Create default dashboards
            await self._create_default_dashboards()
            
            # Create default alerts
            await self._create_default_alerts()
            
            logger.info("Default dashboards and alerts initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize defaults: {e}")
    
    async def _create_default_dashboards(self):
        """Create default enterprise dashboards."""
        try:
            # Enterprise Overview Dashboard
            overview_dashboard = Dashboard(
                dashboard_id="enterprise_overview",
                title="Enterprise Overview",
                description="High-level enterprise metrics and health status",
                dashboard_type=DashboardType.OVERVIEW,
                widgets=[
                    DashboardWidget(
                        widget_id="total_users",
                        title="Total Users",
                        type="stat",
                        metric_queries=["m3tm_users_total"],
                        position={"x": 0, "y": 0, "width": 6, "height": 3}
                    ),
                    DashboardWidget(
                        widget_id="active_tenants",
                        title="Active Tenants",
                        type="stat",
                        metric_queries=["m3tm_tenants_active"],
                        position={"x": 6, "y": 0, "width": 6, "height": 3}
                    ),
                    DashboardWidget(
                        widget_id="api_requests",
                        title="API Requests/Hour",
                        type="graph",
                        metric_queries=["rate(m3tm_api_requests_total[1h])"],
                        position={"x": 0, "y": 3, "width": 12, "height": 6}
                    ),
                    DashboardWidget(
                        widget_id="compliance_score",
                        title="Compliance Score",
                        type="gauge",
                        metric_queries=["m3tm_compliance_score"],
                        config={"min": 0, "max": 100, "unit": "%"},
                        position={"x": 12, "y": 0, "width": 6, "height": 6}
                    )
                ]
            )
            
            # Governance Dashboard
            governance_dashboard = Dashboard(
                dashboard_id="governance",
                title="Data Governance",
                description="Data governance and lineage metrics",
                dashboard_type=DashboardType.GOVERNANCE,
                widgets=[
                    DashboardWidget(
                        widget_id="data_assets",
                        title="Total Data Assets",
                        type="stat",
                        metric_queries=["m3tm_data_assets_total"],
                        position={"x": 0, "y": 0, "width": 4, "height": 3}
                    ),
                    DashboardWidget(
                        widget_id="lineage_relations",
                        title="Lineage Relations",
                        type="stat",
                        metric_queries=["m3tm_lineage_relations_total"],
                        position={"x": 4, "y": 0, "width": 4, "height": 3}
                    ),
                    DashboardWidget(
                        widget_id="classification_distribution",
                        title="Data Classification",
                        type="pie",
                        metric_queries=["m3tm_data_classification"],
                        position={"x": 8, "y": 0, "width": 4, "height": 6}
                    )
                ]
            )
            
            # Security Dashboard
            security_dashboard = Dashboard(
                dashboard_id="security",
                title="Security Monitoring",
                description="Security events and threat monitoring",
                dashboard_type=DashboardType.SECURITY,
                widgets=[
                    DashboardWidget(
                        widget_id="failed_logins",
                        title="Failed Login Attempts",
                        type="graph",
                        metric_queries=["m3tm_authentication_failures_total"],
                        position={"x": 0, "y": 0, "width": 6, "height": 4}
                    ),
                    DashboardWidget(
                        widget_id="policy_violations",
                        title="Policy Violations",
                        type="table",
                        metric_queries=["m3tm_policy_violations"],
                        position={"x": 6, "y": 0, "width": 6, "height": 4}
                    )
                ]
            )
            
            # Store dashboards
            self._dashboards[overview_dashboard.dashboard_id] = overview_dashboard
            self._dashboards[governance_dashboard.dashboard_id] = governance_dashboard
            self._dashboards[security_dashboard.dashboard_id] = security_dashboard
            
        except Exception as e:
            logger.error(f"Failed to create default dashboards: {e}")
    
    async def _create_default_alerts(self):
        """Create default enterprise alerts."""
        try:
            default_alerts = [
                Alert(
                    alert_id="high_api_error_rate",
                    name="High API Error Rate",
                    description="API error rate exceeds threshold",
                    severity=AlertSeverity.WARNING,
                    condition="rate(m3tm_api_errors_total[5m]) > 0.1",
                    threshold=0.1,
                    metric_name="m3tm_api_errors_total"
                ),
                Alert(
                    alert_id="compliance_score_low",
                    name="Low Compliance Score",
                    description="Compliance score has dropped below acceptable threshold",
                    severity=AlertSeverity.CRITICAL,
                    condition="m3tm_compliance_score < 80",
                    threshold=80,
                    metric_name="m3tm_compliance_score"
                ),
                Alert(
                    alert_id="failed_authentication_spike",
                    name="Authentication Failure Spike",
                    description="Unusual spike in authentication failures",
                    severity=AlertSeverity.WARNING,
                    condition="increase(m3tm_authentication_failures_total[1h]) > 100",
                    threshold=100,
                    metric_name="m3tm_authentication_failures_total"
                ),
                Alert(
                    alert_id="disk_usage_high",
                    name="High Disk Usage",
                    description="Disk usage exceeds 85%",
                    severity=AlertSeverity.WARNING,
                    condition="m3tm_disk_usage_percent > 85",
                    threshold=85,
                    metric_name="m3tm_disk_usage_percent"
                )
            ]
            
            for alert in default_alerts:
                self._alerts[alert.alert_id] = alert
                
        except Exception as e:
            logger.error(f"Failed to create default alerts: {e}")
    
    async def record_metric(self, name: str, value: Union[int, float],
                          metric_type: MetricType = MetricType.GAUGE,
                          labels: Optional[Dict[str, str]] = None,
                          description: Optional[str] = None,
                          unit: Optional[str] = None):
        """Record a metric value."""
        try:
            metric = Metric(
                name=name,
                type=metric_type,
                value=value,
                labels=labels or {},
                description=description,
                unit=unit
            )
            
            # Store metric
            if name not in self._metrics:
                self._metrics[name] = []
            
            self._metrics[name].append(metric)
            
            # Keep only recent metrics (last 24 hours)
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            self._metrics[name] = [
                m for m in self._metrics[name] 
                if m.timestamp >= cutoff_time
            ]
            
            # TODO: Send to Prometheus
            # await self._send_to_prometheus(metric)
            
            logger.debug(f"Metric recorded: {name} = {value}")
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    async def get_metric_values(self, metric_name: str,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              labels: Optional[Dict[str, str]] = None) -> List[Metric]:
        """Get metric values within time range."""
        try:
            if metric_name not in self._metrics:
                return []
            
            metrics = self._metrics[metric_name]
            
            # Apply time filters
            if start_time:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            
            if end_time:
                metrics = [m for m in metrics if m.timestamp <= end_time]
            
            # Apply label filters
            if labels:
                metrics = [
                    m for m in metrics 
                    if all(m.labels.get(k) == v for k, v in labels.items())
                ]
            
            return sorted(metrics, key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.error(f"Failed to get metric values for {metric_name}: {e}")
            return []
    
    async def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard configuration."""
        return self._dashboards.get(dashboard_id)
    
    async def list_dashboards(self, dashboard_type: Optional[DashboardType] = None) -> List[Dashboard]:
        """List available dashboards."""
        dashboards = list(self._dashboards.values())
        if dashboard_type:
            dashboards = [d for d in dashboards if d.dashboard_type == dashboard_type]
        return dashboards
    
    async def create_dashboard(self, dashboard: Dashboard) -> bool:
        """Create or update dashboard."""
        try:
            dashboard.updated_at = datetime.now(timezone.utc)
            self._dashboards[dashboard.dashboard_id] = dashboard
            
            # TODO: Create/update in Grafana
            # await self._sync_with_grafana(dashboard)
            
            logger.info(f"Dashboard created/updated: {dashboard.dashboard_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dashboard {dashboard.dashboard_id}: {e}")
            return False
    
    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard data with current metric values."""
        try:
            dashboard = self._dashboards.get(dashboard_id)
            if not dashboard:
                return {"error": "Dashboard not found"}
            
            widget_data = {}
            
            for widget in dashboard.widgets:
                widget_values = {}
                
                for query in widget.metric_queries:
                    # Simple metric query resolution (would use PromQL in real implementation)
                    metric_name = query.split("(")[0] if "(" in query else query
                    
                    # Get recent values
                    recent_time = datetime.now(timezone.utc) - timedelta(minutes=30)
                    values = await self.get_metric_values(metric_name, start_time=recent_time)
                    
                    if values:
                        if widget.type == "stat":
                            # Latest value
                            widget_values[query] = values[-1].value
                        elif widget.type == "graph":
                            # Time series data
                            widget_values[query] = [
                                {"timestamp": v.timestamp.isoformat(), "value": v.value}
                                for v in values
                            ]
                        else:
                            # Default to latest value
                            widget_values[query] = values[-1].value if values else 0
                    else:
                        widget_values[query] = 0
                
                widget_data[widget.widget_id] = widget_values
            
            return {
                "dashboard": asdict(dashboard),
                "data": widget_data,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data for {dashboard_id}: {e}")
            return {"error": str(e)}
    
    async def create_alert(self, alert: Alert) -> bool:
        """Create or update alert."""
        try:
            self._alerts[alert.alert_id] = alert
            logger.info(f"Alert created/updated: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create alert {alert.alert_id}: {e}")
            return False
    
    async def get_alerts(self, active_only: bool = True) -> List[Alert]:
        """Get alerts."""
        alerts = list(self._alerts.values())
        if active_only:
            alerts = [a for a in alerts if a.active]
        return alerts
    
    def add_alert_handler(self, handler: Callable[[Alert, float], None]):
        """Add alert handler function."""
        self._alert_handlers.append(handler)
    
    async def _metrics_collector(self):
        """Background task to collect system metrics."""
        while True:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds
                
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect enterprise metrics
                await self._collect_enterprise_metrics()
                
            except Exception as e:
                logger.error(f"Metrics collection failed: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.record_metric("m3tm_cpu_usage_percent", cpu_percent, MetricType.GAUGE, unit="%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            await self.record_metric("m3tm_memory_usage_percent", memory.percent, MetricType.GAUGE, unit="%")
            await self.record_metric("m3tm_memory_used_bytes", memory.used, MetricType.GAUGE, unit="bytes")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self.record_metric("m3tm_disk_usage_percent", disk_percent, MetricType.GAUGE, unit="%")
            await self.record_metric("m3tm_disk_used_bytes", disk.used, MetricType.GAUGE, unit="bytes")
            
        except ImportError:
            # psutil not available, use mock values
            await self.record_metric("m3tm_cpu_usage_percent", 25.0, MetricType.GAUGE, unit="%")
            await self.record_metric("m3tm_memory_usage_percent", 60.0, MetricType.GAUGE, unit="%")
            await self.record_metric("m3tm_disk_usage_percent", 45.0, MetricType.GAUGE, unit="%")
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    async def _collect_enterprise_metrics(self):
        """Collect enterprise-specific metrics."""
        try:
            # These would be collected from the actual enterprise components
            # For now, using mock values
            
            await self.record_metric("m3tm_users_total", 150, MetricType.GAUGE)
            await self.record_metric("m3tm_tenants_active", 12, MetricType.GAUGE)
            await self.record_metric("m3tm_data_assets_total", 1247, MetricType.GAUGE)
            await self.record_metric("m3tm_lineage_relations_total", 3456, MetricType.GAUGE)
            await self.record_metric("m3tm_compliance_score", 87.5, MetricType.GAUGE, unit="%")
            
            # API metrics (would be collected from actual API gateway)
            await self.record_metric("m3tm_api_requests_total", 50000, MetricType.COUNTER)
            await self.record_metric("m3tm_api_errors_total", 125, MetricType.COUNTER) 
            await self.record_metric("m3tm_authentication_failures_total", 23, MetricType.COUNTER)
            
        except Exception as e:
            logger.error(f"Enterprise metrics collection failed: {e}")
    
    async def _alert_evaluator(self):
        """Background task to evaluate alerts."""
        while True:
            try:
                await asyncio.sleep(60)  # Evaluate every minute
                
                for alert in self._alerts.values():
                    if not alert.active:
                        continue
                    
                    # Get current metric value
                    current_values = await self.get_metric_values(
                        alert.metric_name,
                        start_time=datetime.now(timezone.utc) - timedelta(minutes=5)
                    )
                    
                    if not current_values:
                        continue
                    
                    # Simple threshold evaluation
                    current_value = current_values[-1].value
                    
                    triggered = False
                    
                    # Evaluate condition
                    if ">" in alert.condition:
                        triggered = current_value > alert.threshold
                    elif "<" in alert.condition:
                        triggered = current_value < alert.threshold
                    
                    if triggered:
                        alert.last_triggered = datetime.now(timezone.utc)
                        
                        # Call alert handlers
                        for handler in self._alert_handlers:
                            try:
                                await asyncio.create_task(handler(alert, current_value))
                            except Exception as e:
                                logger.error(f"Alert handler failed: {e}")
                        
                        logger.warning(f"Alert triggered: {alert.name} (value: {current_value})")
                
            except Exception as e:
                logger.error(f"Alert evaluation failed: {e}")
    
    async def get_enterprise_health_status(self) -> Dict[str, Any]:
        """Get overall enterprise health status."""
        try:
            # Get recent metrics
            recent_time = datetime.now(timezone.utc) - timedelta(minutes=10)
            
            health_indicators = {}
            
            # System health
            cpu_values = await self.get_metric_values("m3tm_cpu_usage_percent", start_time=recent_time)
            memory_values = await self.get_metric_values("m3tm_memory_usage_percent", start_time=recent_time)
            disk_values = await self.get_metric_values("m3tm_disk_usage_percent", start_time=recent_time)
            
            system_health = "healthy"
            if cpu_values and cpu_values[-1].value > 80:
                system_health = "warning"
            if memory_values and memory_values[-1].value > 85:
                system_health = "critical"
            if disk_values and disk_values[-1].value > 90:
                system_health = "critical"
            
            health_indicators["system"] = {
                "status": system_health,
                "cpu_usage": cpu_values[-1].value if cpu_values else 0,
                "memory_usage": memory_values[-1].value if memory_values else 0,
                "disk_usage": disk_values[-1].value if disk_values else 0
            }
            
            # Compliance health
            compliance_values = await self.get_metric_values("m3tm_compliance_score", start_time=recent_time)
            compliance_health = "healthy"
            if compliance_values:
                score = compliance_values[-1].value
                if score < 70:
                    compliance_health = "critical"
                elif score < 85:
                    compliance_health = "warning"
            
            health_indicators["compliance"] = {
                "status": compliance_health,
                "score": compliance_values[-1].value if compliance_values else 0
            }
            
            # Security health
            auth_failure_values = await self.get_metric_values("m3tm_authentication_failures_total", start_time=recent_time)
            security_health = "healthy"
            if auth_failure_values and len(auth_failure_values) > 10:
                security_health = "warning"
            
            health_indicators["security"] = {
                "status": security_health,
                "recent_auth_failures": len(auth_failure_values) if auth_failure_values else 0
            }
            
            # Overall health score
            health_scores = {"healthy": 100, "warning": 75, "critical": 25}
            overall_score = statistics.mean([
                health_scores[health_indicators["system"]["status"]],
                health_scores[health_indicators["compliance"]["status"]],
                health_scores[health_indicators["security"]["status"]]
            ])
            
            overall_status = "healthy"
            if overall_score < 50:
                overall_status = "critical"
            elif overall_score < 80:
                overall_status = "warning"
            
            return {
                "overall_status": overall_status,
                "overall_score": round(overall_score, 1),
                "health_indicators": health_indicators,
                "active_alerts": len([a for a in self._alerts.values() if a.last_triggered and 
                                    (datetime.now(timezone.utc) - a.last_triggered).seconds < 3600]),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get enterprise health status: {e}")
            return {"error": str(e)}

# Default alert handler for logging
async def default_alert_handler(alert: Alert, current_value: float):
    """Default alert handler that logs alerts."""
    logger.warning(f"ALERT: {alert.name} - {alert.description} (current value: {current_value}, threshold: {alert.threshold})")

# Export main classes
__all__ = [
    "EnterpriseDashboard", "Dashboard", "DashboardWidget", "Metric", "Alert",
    "MetricType", "AlertSeverity", "DashboardType", "default_alert_handler"
]
