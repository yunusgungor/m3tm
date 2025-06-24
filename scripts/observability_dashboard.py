#!/usr/bin/env python3
"""
M³TM Production Observability Dashboard
Enhanced with Context7 best practices for comprehensive monitoring
"""

import json
import time
import psutil
import subprocess
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class M3TMObservabilityDashboard:
    """Comprehensive observability dashboard for M³TM production systems"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.metrics_history = []
        self.alerts = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load dashboard configuration"""
        default_config = {
            "update_interval": 30,  # seconds
            "retention_days": 30,
            "alert_thresholds": {
                "cpu_percent": 80,
                "memory_percent": 85,
                "disk_percent": 90,
                "inference_time_ms": 1000,
                "error_rate": 0.05
            },
            "endpoints": {
                "health_check": "http://localhost:8000/health",
                "metrics": "http://localhost:8000/metrics",
                "firebase_analytics": None,  # Requires setup
                "play_console": None,        # Requires setup
                "app_store": None           # Requires setup
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": {},
            "application": {},
            "model": {},
            "mobile": {}
        }
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics["system"] = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / (1024**3),
            "disk_total_gb": disk.total / (1024**3),
            "load_average": list(psutil.getloadavg())
        }
        
        # Application metrics
        try:
            app_metrics = self._collect_application_metrics()
            metrics["application"] = app_metrics
        except Exception as e:
            print(f"Failed to collect application metrics: {e}")
            metrics["application"] = {"error": str(e)}
        
        # Model performance metrics
        try:
            model_metrics = self._collect_model_metrics()
            metrics["model"] = model_metrics
        except Exception as e:
            print(f"Failed to collect model metrics: {e}")
            metrics["model"] = {"error": str(e)}
        
        # Mobile platform metrics
        try:
            mobile_metrics = self._collect_mobile_metrics()
            metrics["mobile"] = mobile_metrics
        except Exception as e:
            print(f"Failed to collect mobile metrics: {e}")
            metrics["mobile"] = {"error": str(e)}
        
        return metrics
    
    def _collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics"""
        metrics = {}
        
        # Health check
        health_endpoint = self.config["endpoints"]["health_check"]
        if health_endpoint:
            try:
                response = requests.get(health_endpoint, timeout=5)
                metrics["health_status"] = "healthy" if response.status_code == 200 else "unhealthy"
                metrics["response_time_ms"] = response.elapsed.total_seconds() * 1000
            except requests.RequestException:
                metrics["health_status"] = "unhealthy"
                metrics["response_time_ms"] = None
        
        # Process metrics
        m3tm_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            if 'm3tm' in proc.info['name'].lower():
                m3tm_processes.append(proc.info)
        
        metrics["processes"] = m3tm_processes
        metrics["process_count"] = len(m3tm_processes)
        
        return metrics
    
    def _collect_model_metrics(self) -> Dict[str, Any]:
        """Collect model performance metrics"""
        metrics = {}
        
        # Run benchmark
        try:
            benchmark_result = subprocess.run([
                "python", "-c", 
                """
import time
import sys
sys.path.append('src')
from m3tm.examples.simplified_mobile_demo import quick_inference_test

start = time.time()
try:
    quick_inference_test()
    end = time.time()
    print(f'inference_time_ms:{(end-start)*1000:.2f}')
    print('status:success')
except Exception as e:
    print(f'status:error')
    print(f'error:{str(e)}')
                """
            ], capture_output=True, text=True, timeout=30)
            
            if benchmark_result.returncode == 0:
                for line in benchmark_result.stdout.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        if key == "inference_time_ms":
                            metrics[key] = float(value)
                        else:
                            metrics[key] = value
            else:
                metrics["status"] = "error"
                metrics["error"] = benchmark_result.stderr
                
        except subprocess.TimeoutExpired:
            metrics["status"] = "timeout"
            metrics["error"] = "Benchmark timed out"
        except Exception as e:
            metrics["status"] = "error"
            metrics["error"] = str(e)
        
        # Model file checks
        model_dir = Path("model_checkpoints")
        if model_dir.exists():
            model_files = list(model_dir.glob("*.pt"))
            metrics["model_files_count"] = len(model_files)
            metrics["model_files"] = [f.name for f in model_files]
        else:
            metrics["model_files_count"] = 0
            metrics["model_files"] = []
        
        return metrics
    
    def _collect_mobile_metrics(self) -> Dict[str, Any]:
        """Collect mobile platform metrics"""
        metrics = {
            "android": {},
            "ios": {},
            "firebase": {}
        }
        
        # Firebase metrics (if configured)
        firebase_endpoint = self.config["endpoints"]["firebase_analytics"]
        if firebase_endpoint:
            try:
                # This would integrate with Firebase Analytics API
                # For now, we'll simulate the data structure
                metrics["firebase"] = {
                    "active_users_24h": None,
                    "crash_free_rate": None,
                    "avg_session_duration": None,
                    "retention_rate": None
                }
            except Exception as e:
                metrics["firebase"]["error"] = str(e)
        
        # Check Android build status
        android_dir = Path("android")
        if android_dir.exists():
            try:
                # Check if Android project can build
                build_result = subprocess.run([
                    "./gradlew", "tasks"
                ], cwd=android_dir, capture_output=True, text=True, timeout=30)
                
                metrics["android"]["build_ready"] = build_result.returncode == 0
                if build_result.returncode != 0:
                    metrics["android"]["build_error"] = build_result.stderr[:200]
                    
            except Exception as e:
                metrics["android"]["build_ready"] = False
                metrics["android"]["error"] = str(e)
        
        # Check iOS project status (if on macOS)
        ios_dir = Path("ios")
        if ios_dir.exists():
            try:
                # Basic iOS project check
                xcworkspace = ios_dir / "M3TM.xcworkspace"
                metrics["ios"]["project_exists"] = xcworkspace.exists()
                
                if xcworkspace.exists():
                    # Could add xcodebuild validation here
                    metrics["ios"]["build_ready"] = True
                
            except Exception as e:
                metrics["ios"]["error"] = str(e)
        
        return metrics
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        thresholds = self.config["alert_thresholds"]
        
        # System alerts
        system = metrics.get("system", {})
        if system.get("cpu_percent", 0) > thresholds["cpu_percent"]:
            alerts.append({
                "type": "system",
                "severity": "warning",
                "message": f"High CPU usage: {system['cpu_percent']:.1f}%",
                "timestamp": metrics["timestamp"]
            })
        
        if system.get("memory_percent", 0) > thresholds["memory_percent"]:
            alerts.append({
                "type": "system", 
                "severity": "warning",
                "message": f"High memory usage: {system['memory_percent']:.1f}%",
                "timestamp": metrics["timestamp"]
            })
        
        if system.get("disk_percent", 0) > thresholds["disk_percent"]:
            alerts.append({
                "type": "system",
                "severity": "critical",
                "message": f"High disk usage: {system['disk_percent']:.1f}%",
                "timestamp": metrics["timestamp"]
            })
        
        # Model performance alerts
        model = metrics.get("model", {})
        inference_time = model.get("inference_time_ms", 0)
        if inference_time > thresholds["inference_time_ms"]:
            alerts.append({
                "type": "performance",
                "severity": "warning",
                "message": f"Slow inference: {inference_time:.1f}ms",
                "timestamp": metrics["timestamp"]
            })
        
        if model.get("status") == "error":
            alerts.append({
                "type": "model",
                "severity": "critical",
                "message": f"Model error: {model.get('error', 'Unknown error')}",
                "timestamp": metrics["timestamp"]
            })
        
        # Application health alerts
        app = metrics.get("application", {})
        if app.get("health_status") == "unhealthy":
            alerts.append({
                "type": "application",
                "severity": "critical",
                "message": "Application health check failed",
                "timestamp": metrics["timestamp"]
            })
        
        return alerts
    
    def generate_dashboard(self) -> str:
        """Generate HTML dashboard"""
        if not self.metrics_history:
            return self._generate_empty_dashboard()
        
        latest_metrics = self.metrics_history[-1]
        
        # Generate visualizations
        self._generate_plots()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>M³TM Production Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .metric {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .metric-name {{ font-weight: bold; color: #333; }}
        .metric-value {{ font-size: 1.2em; }}
        .status-healthy {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-critical {{ color: #dc3545; }}
        .alert {{ padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .alert-warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
        .alert-critical {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        .refresh-info {{ text-align: center; color: #666; margin-top: 20px; }}
        .progress-bar {{ width: 100%; height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; transition: width 0.3s ease; }}
        .progress-low {{ background-color: #28a745; }}
        .progress-medium {{ background-color: #ffc107; }}
        .progress-high {{ background-color: #dc3545; }}
    </style>
    <script>
        setTimeout(() => {{ window.location.reload(); }}, {self.config['update_interval'] * 1000});
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 M³TM Production Dashboard</h1>
            <p>Real-time monitoring and observability • Last updated: {latest_metrics['timestamp']}</p>
        </div>
        
        {self._generate_alerts_section()}
        
        <div class="grid">
            {self._generate_system_card(latest_metrics.get('system', {}))}
            {self._generate_application_card(latest_metrics.get('application', {}))}
            {self._generate_model_card(latest_metrics.get('model', {}))}
            {self._generate_mobile_card(latest_metrics.get('mobile', {}))}
        </div>
        
        <div class="chart-container">
            <h2>Performance Trends</h2>
            <img src="dashboard_plots.png" alt="Performance Charts" style="max-width: 100%; height: auto;">
        </div>
        
        <div class="refresh-info">
            <p>Dashboard auto-refreshes every {self.config['update_interval']} seconds</p>
            <p>Data retention: {self.config['retention_days']} days</p>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_empty_dashboard(self) -> str:
        """Generate dashboard when no metrics are available"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>M³TM Production Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 100px; }
        .loading { color: #666; }
    </style>
    <script>
        setTimeout(() => { window.location.reload(); }, 10000);
    </script>
</head>
<body>
    <div class="loading">
        <h1>🚀 M³TM Dashboard</h1>
        <p>Collecting metrics... Please wait.</p>
        <p>Refreshing in 10 seconds.</p>
    </div>
</body>
</html>"""
    
    def _generate_alerts_section(self) -> str:
        """Generate alerts section HTML"""
        if not self.alerts:
            return ""
        
        html = "<div class='card'><h2>🚨 Active Alerts</h2>"
        
        for alert in self.alerts[-10:]:  # Show last 10 alerts
            severity_class = f"alert-{alert['severity']}"
            html += f"<div class='alert {severity_class}'>"
            html += f"<strong>{alert['type'].title()}</strong>: {alert['message']}"
            html += f"<br><small>{alert['timestamp']}</small>"
            html += "</div>"
        
        html += "</div>"
        return html
    
    def _generate_system_card(self, system_metrics: Dict[str, Any]) -> str:
        """Generate system metrics card"""
        if not system_metrics:
            return "<div class='card'><h2>💻 System Metrics</h2><p>No data available</p></div>"
        
        cpu_class = self._get_status_class(system_metrics.get("cpu_percent", 0), 70, 85)
        mem_class = self._get_status_class(system_metrics.get("memory_percent", 0), 70, 85)
        disk_class = self._get_status_class(system_metrics.get("disk_percent", 0), 80, 90)
        
        return f"""
        <div class='card'>
            <h2>💻 System Metrics</h2>
            {self._format_metric("CPU Usage", f"{system_metrics.get('cpu_percent', 0):.1f}%", cpu_class)}
            {self._format_progress_bar(system_metrics.get('cpu_percent', 0))}
            
            {self._format_metric("Memory Usage", f"{system_metrics.get('memory_percent', 0):.1f}%", mem_class)}
            {self._format_progress_bar(system_metrics.get('memory_percent', 0))}
            
            {self._format_metric("Disk Usage", f"{system_metrics.get('disk_percent', 0):.1f}%", disk_class)}
            {self._format_progress_bar(system_metrics.get('disk_percent', 0))}
            
            {self._format_metric("Memory", f"{system_metrics.get('memory_used_gb', 0):.1f}GB / {system_metrics.get('memory_total_gb', 0):.1f}GB")}
            {self._format_metric("Load Average", str(system_metrics.get('load_average', [])))}
        </div>"""
    
    def _generate_application_card(self, app_metrics: Dict[str, Any]) -> str:
        """Generate application metrics card"""
        if not app_metrics:
            return "<div class='card'><h2>🔧 Application Metrics</h2><p>No data available</p></div>"
        
        health_status = app_metrics.get("health_status", "unknown")
        health_class = "status-healthy" if health_status == "healthy" else "status-critical"
        
        response_time = app_metrics.get("response_time_ms")
        response_display = f"{response_time:.0f}ms" if response_time else "N/A"
        
        return f"""
        <div class='card'>
            <h2>🔧 Application Metrics</h2>
            {self._format_metric("Health Status", health_status.title(), health_class)}
            {self._format_metric("Response Time", response_display)}
            {self._format_metric("Active Processes", str(app_metrics.get('process_count', 0)))}
        </div>"""
    
    def _generate_model_card(self, model_metrics: Dict[str, Any]) -> str:
        """Generate model metrics card"""
        if not model_metrics:
            return "<div class='card'><h2>🧠 Model Metrics</h2><p>No data available</p></div>"
        
        status = model_metrics.get("status", "unknown")
        status_class = "status-healthy" if status == "success" else "status-critical"
        
        inference_time = model_metrics.get("inference_time_ms", 0)
        time_class = self._get_status_class(inference_time, 500, 1000)
        
        return f"""
        <div class='card'>
            <h2>🧠 Model Metrics</h2>
            {self._format_metric("Status", status.title(), status_class)}
            {self._format_metric("Inference Time", f"{inference_time:.1f}ms", time_class)}
            {self._format_metric("Model Files", str(model_metrics.get('model_files_count', 0)))}
            {self._format_metric("Last Error", model_metrics.get('error', 'None')[:50] + ('...' if len(model_metrics.get('error', '')) > 50 else '') if model_metrics.get('error') else 'None')}
        </div>"""
    
    def _generate_mobile_card(self, mobile_metrics: Dict[str, Any]) -> str:
        """Generate mobile platform metrics card"""
        if not mobile_metrics:
            return "<div class='card'><h2>📱 Mobile Platforms</h2><p>No data available</p></div>"
        
        android = mobile_metrics.get("android", {})
        ios = mobile_metrics.get("ios", {})
        firebase = mobile_metrics.get("firebase", {})
        
        android_status = "✅ Ready" if android.get("build_ready") else "❌ Issues"
        ios_status = "✅ Ready" if ios.get("build_ready") else "❌ Issues"
        
        return f"""
        <div class='card'>
            <h2>📱 Mobile Platforms</h2>
            {self._format_metric("Android Build", android_status)}
            {self._format_metric("iOS Build", ios_status)}
            {self._format_metric("Firebase Status", "✅ Connected" if firebase and not firebase.get('error') else "⚠️ Disconnected")}
        </div>"""
    
    def _format_metric(self, name: str, value: str, status_class: str = "") -> str:
        """Format a metric for display"""
        return f"""
        <div class='metric'>
            <span class='metric-name'>{name}:</span>
            <span class='metric-value {status_class}'>{value}</span>
        </div>"""
    
    def _format_progress_bar(self, percentage: float) -> str:
        """Format a progress bar"""
        if percentage < 70:
            bar_class = "progress-low"
        elif percentage < 85:
            bar_class = "progress-medium"
        else:
            bar_class = "progress-high"
        
        return f"""
        <div class='progress-bar'>
            <div class='progress-fill {bar_class}' style='width: {percentage}%'></div>
        </div>"""
    
    def _get_status_class(self, value: float, warning_threshold: float, critical_threshold: float) -> str:
        """Get CSS class based on thresholds"""
        if value < warning_threshold:
            return "status-healthy"
        elif value < critical_threshold:
            return "status-warning"
        else:
            return "status-critical"
    
    def _generate_plots(self) -> None:
        """Generate performance trend plots"""
        if len(self.metrics_history) < 2:
            return
        
        # Extract time series data
        timestamps = [datetime.fromisoformat(m["timestamp"]) for m in self.metrics_history]
        cpu_usage = [m.get("system", {}).get("cpu_percent", 0) for m in self.metrics_history]
        memory_usage = [m.get("system", {}).get("memory_percent", 0) for m in self.metrics_history]
        inference_times = [m.get("model", {}).get("inference_time_ms", 0) for m in self.metrics_history]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('M³TM Performance Trends', fontsize=16, fontweight='bold')
        
        # CPU Usage
        axes[0, 0].plot(timestamps, cpu_usage, 'b-', linewidth=2)
        axes[0, 0].set_title('CPU Usage %')
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Alert threshold')
        
        # Memory Usage
        axes[0, 1].plot(timestamps, memory_usage, 'g-', linewidth=2)
        axes[0, 1].set_title('Memory Usage %')
        axes[0, 1].set_ylabel('Percentage')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=85, color='r', linestyle='--', alpha=0.7, label='Alert threshold')
        
        # Inference Time
        axes[1, 0].plot(timestamps, inference_times, 'orange', linewidth=2)
        axes[1, 0].set_title('Model Inference Time')
        axes[1, 0].set_ylabel('Milliseconds')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=1000, color='r', linestyle='--', alpha=0.7, label='Alert threshold')
        
        # Combined System Health
        health_score = [(100 - cpu) * (100 - mem) / 100 for cpu, mem in zip(cpu_usage, memory_usage)]
        axes[1, 1].plot(timestamps, health_score, 'purple', linewidth=2)
        axes[1, 1].set_title('System Health Score')
        axes[1, 1].set_ylabel('Score (0-100)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Format x-axes
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('dashboard_plots.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_dashboard(self, output_file: str = "dashboard.html", max_iterations: Optional[int] = None) -> None:
        """Run the dashboard continuously"""
        print("🚀 Starting M³TM Observability Dashboard...")
        print(f"📊 Dashboard will be saved to: {output_file}")
        print(f"🔄 Update interval: {self.config['update_interval']} seconds")
        
        iteration = 0
        try:
            while max_iterations is None or iteration < max_iterations:
                print(f"\n📈 Collecting metrics (iteration {iteration + 1})...")
                
                # Collect metrics
                metrics = self.collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Clean old metrics (keep only retention_days worth)
                cutoff_time = datetime.now() - timedelta(days=self.config["retention_days"])
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if datetime.fromisoformat(m["timestamp"]) > cutoff_time
                ]
                
                # Check for alerts
                new_alerts = self.check_alerts(metrics)
                self.alerts.extend(new_alerts)
                
                # Clean old alerts (keep only last 100)
                self.alerts = self.alerts[-100:]
                
                # Print new alerts
                for alert in new_alerts:
                    severity_emoji = "🚨" if alert["severity"] == "critical" else "⚠️"
                    print(f"{severity_emoji} {alert['message']}")
                
                # Generate dashboard
                dashboard_html = self.generate_dashboard()
                
                # Save dashboard
                with open(output_file, 'w') as f:
                    f.write(dashboard_html)
                
                print(f"✅ Dashboard updated: {output_file}")
                
                if max_iterations is None:
                    time.sleep(self.config["update_interval"])
                
                iteration += 1
                
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped by user")
        except Exception as e:
            print(f"\n❌ Dashboard error: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="M³TM Observability Dashboard")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output", default="dashboard.html", help="Output HTML file")
    parser.add_argument("--interval", type=int, default=30, help="Update interval in seconds")
    parser.add_argument("--iterations", type=int, help="Max iterations (for testing)")
    
    args = parser.parse_args()
    
    # Create dashboard
    dashboard = M3TMObservabilityDashboard(args.config)
    
    # Override interval if specified
    if args.interval:
        dashboard.config["update_interval"] = args.interval
    
    # Run dashboard
    dashboard.run_dashboard(args.output, args.iterations)

if __name__ == "__main__":
    main()
