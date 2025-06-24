"""
Advanced Performance Profiler for M³TM Mobile
Context7 Enhanced Implementation with Battery Impact, Thermal Monitoring, and Device Analytics

Features:
- Real-time performance metrics collection
- Battery impact assessment 
- Thermal throttling detection
- Memory usage tracking with detailed breakdowns
- Performance regression detection
- Device capability profiling
- Mobile-specific benchmarking tools
- Continuous monitoring and alerting
"""

import time
import json
import threading
import platform
import subprocess
import psutil
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque, defaultdict
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class DeviceInfo:
    """Device information and capabilities"""
    platform: str
    cpu_cores: int
    cpu_freq_max: float
    memory_total: int
    gpu_available: bool
    gpu_name: str
    neural_engine: bool
    thermal_zones: List[str]
    battery_available: bool

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: float
    inference_time: float
    memory_usage: int
    cpu_usage: float
    gpu_usage: float
    gpu_memory: int
    thermal_state: str
    battery_level: float
    battery_drain_rate: float
    cache_hit_rate: float
    throughput: float
    model_accuracy: float
    
@dataclass
class BenchmarkResult:
    """Benchmark result with detailed metrics"""
    test_name: str
    device_info: DeviceInfo
    avg_inference_time: float
    p50_inference_time: float
    p95_inference_time: float
    p99_inference_time: float
    max_memory_usage: int
    avg_cpu_usage: float
    max_gpu_usage: float
    thermal_throttling_detected: bool
    battery_drain_per_hour: float
    throughput_samples_per_sec: float
    quality_score: float
    optimization_recommendations: List[str]

@dataclass
class ProfilingConfig:
    """Configuration for advanced mobile profiler"""
    level: str = "comprehensive"  # basic, standard, comprehensive
    enable_battery_monitoring: bool = True
    enable_thermal_monitoring: bool = True
    enable_regression_detection: bool = True
    enable_device_analytics: bool = True
    sampling_interval: float = 0.1  # seconds
    metrics_buffer_size: int = 1000
    enable_continuous_monitoring: bool = False
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'cpu_usage': 80.0,
                'memory_usage': 85.0,
                'temperature': 70.0,
                'battery_drain': 10.0
            }

class DeviceCapabilityDetector:
    """Detect device capabilities and optimal configurations"""
    
    @staticmethod
    def detect_device() -> DeviceInfo:
        """Detect comprehensive device information"""
        system = platform.system().lower()
        
        # Basic CPU info
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_max = cpu_freq.max if cpu_freq else 0
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_total = memory.total
        
        # GPU detection
        gpu_available = torch.cuda.is_available()
        gpu_name = ""
        if gpu_available:
            gpu_name = torch.cuda.get_device_name()
        
        # Neural engine detection (iOS/macOS specific)
        neural_engine = False
        if system == "darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"], 
                    capture_output=True, text=True, timeout=5
                )
                neural_engine = "Neural Engine" in result.stdout
            except:
                pass
        
        # Thermal zones detection
        thermal_zones = []
        if hasattr(psutil, "sensors_temperatures"):
            try:
                temps = psutil.sensors_temperatures()
                thermal_zones = list(temps.keys())
            except:
                pass
        
        # Battery detection
        battery_available = hasattr(psutil, "sensors_battery") and psutil.sensors_battery() is not None
        
        return DeviceInfo(
            platform=system,
            cpu_cores=cpu_cores,
            cpu_freq_max=cpu_freq_max,
            memory_total=memory_total,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            neural_engine=neural_engine,
            thermal_zones=thermal_zones,
            battery_available=battery_available
        )
    
    @staticmethod
    def get_optimal_config(device_info: DeviceInfo) -> Dict:
        """Get optimal configuration based on device capabilities"""
        config = {
            'batch_size': 1,
            'num_threads': device_info.cpu_cores,
            'use_gpu': device_info.gpu_available,
            'quantization_backend': 'qnnpack' if 'arm' in device_info.platform else 'x86',
            'memory_fraction': 0.8,
            'enable_fusion': True
        }
        
        # Adjust based on memory
        if device_info.memory_total < 2 * 1024**3:  # Less than 2GB
            config['memory_fraction'] = 0.6
            config['batch_size'] = 1
        elif device_info.memory_total < 4 * 1024**3:  # Less than 4GB
            config['memory_fraction'] = 0.7
            config['batch_size'] = 2
        else:
            config['batch_size'] = 4
        
        # Adjust for mobile platforms
        if device_info.platform in ['android', 'ios']:
            config['num_threads'] = min(4, device_info.cpu_cores)
            config['memory_fraction'] = 0.5
        
        return config

class BatteryMonitor:
    """Monitor battery usage and drain rate"""
    
    def __init__(self):
        self.initial_battery = None
        self.start_time = None
        self.battery_history = deque(maxlen=100)
        
    def start_monitoring(self):
        """Start battery monitoring"""
        if hasattr(psutil, "sensors_battery"):
            battery = psutil.sensors_battery()
            if battery:
                self.initial_battery = battery.percent
                self.start_time = time.time()
                self.battery_history.append((time.time(), battery.percent))
    
    def get_current_battery(self) -> Tuple[float, float]:
        """Get current battery level and drain rate"""
        if not hasattr(psutil, "sensors_battery"):
            return 0.0, 0.0
        
        battery = psutil.sensors_battery()
        if not battery:
            return 0.0, 0.0
        
        current_level = battery.percent
        current_time = time.time()
        
        self.battery_history.append((current_time, current_level))
        
        # Calculate drain rate
        drain_rate = 0.0
        if len(self.battery_history) >= 2:
            # Calculate drain over last measurements
            recent_history = list(self.battery_history)[-10:]  # Last 10 measurements
            if len(recent_history) >= 2:
                time_diff = recent_history[-1][0] - recent_history[0][0]
                battery_diff = recent_history[0][1] - recent_history[-1][1]
                
                if time_diff > 0:
                    drain_rate = (battery_diff / time_diff) * 3600  # Per hour
        
        return current_level, drain_rate

class ThermalMonitor:
    """Monitor thermal state and throttling"""
    
    def __init__(self):
        self.thermal_history = deque(maxlen=50)
        self.throttling_detected = False
        
    def get_thermal_state(self) -> Tuple[str, bool]:
        """Get current thermal state"""
        if not hasattr(psutil, "sensors_temperatures"):
            return "unknown", False
        
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return "unknown", False
            
            max_temp = 0
            for sensor_name, sensor_list in temps.items():
                for sensor in sensor_list:
                    if sensor.current:
                        max_temp = max(max_temp, sensor.current)
            
            self.thermal_history.append((time.time(), max_temp))
            
            # Determine thermal state
            if max_temp > 85:
                thermal_state = "critical"
                self.throttling_detected = True
            elif max_temp > 75:
                thermal_state = "hot"
            elif max_temp > 65:
                thermal_state = "warm"
            else:
                thermal_state = "normal"
            
            # Detect throttling by temperature spikes
            if len(self.thermal_history) >= 5:
                recent_temps = [temp for _, temp in list(self.thermal_history)[-5:]]
                temp_variance = np.var(recent_temps)
                if temp_variance > 25:  # High variance might indicate throttling
                    self.throttling_detected = True
            
            return thermal_state, self.throttling_detected
            
        except Exception as e:
            logger.warning(f"Error reading thermal sensors: {e}")
            return "unknown", False

class AdvancedPerformanceProfiler:
    """Advanced performance profiler with mobile-specific metrics"""
    
    def __init__(self, enable_continuous_monitoring: bool = True):
        self.device_info = DeviceCapabilityDetector.detect_device()
        self.battery_monitor = BatteryMonitor()
        self.thermal_monitor = ThermalMonitor()
        
        self.metrics_history = deque(maxlen=1000)
        self.benchmark_results = []
        
        # Profiling state
        self.profiling_active = False
        self.start_time = None
        self.session_metrics = defaultdict(list)
        
        # Continuous monitoring
        self.enable_continuous_monitoring = enable_continuous_monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Performance baselines
        self.baselines = {}
        
        # Regression detection
        self.regression_threshold = 0.2  # 20% performance degradation
        
    def start_profiling_session(self):
        """Start a profiling session"""
        self.profiling_active = True
        self.start_time = time.time()
        self.session_metrics.clear()
        
        # Reset GPU memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Start battery monitoring
        self.battery_monitor.start_monitoring()
        
        # Start continuous monitoring if enabled
        if self.enable_continuous_monitoring and not self.monitoring_active:
            self.start_continuous_monitoring()
        
        logger.info("Performance profiling session started")
    
    def end_profiling_session(self) -> Dict:
        """End profiling session and return summary"""
        if not self.profiling_active:
            return {}
        
        self.profiling_active = False
        session_duration = time.time() - self.start_time
        
        # Calculate session summary
        summary = {
            'session_duration': session_duration,
            'device_info': asdict(self.device_info),
            'total_inferences': len(self.session_metrics['inference_times']),
            'metrics_summary': {}
        }
        
        # Summarize each metric
        for metric_name, values in self.session_metrics.items():
            if values:
                summary['metrics_summary'][metric_name] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        logger.info(f"Performance profiling session ended. Duration: {session_duration:.2f}s")
        return summary
    
    def profile_inference(self, inference_func: Callable, *args, **kwargs) -> Tuple[Any, PerformanceMetrics]:
        """Profile a single inference with detailed metrics"""
        # Pre-inference metrics
        start_time = time.time()
        
        cpu_before = psutil.cpu_percent()
        memory_before = psutil.Process().memory_info().rss
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_before = torch.cuda.memory_allocated()
        else:
            gpu_memory_before = 0
        
        battery_level, battery_drain = self.battery_monitor.get_current_battery()
        thermal_state, throttling = self.thermal_monitor.get_thermal_state()
        
        # Run inference
        try:
            result = inference_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Inference failed during profiling: {e}")
            raise
        
        # Post-inference metrics
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_after = torch.cuda.memory_allocated()
            gpu_memory_used = gpu_memory_after - gpu_memory_before
        else:
            gpu_memory_used = 0
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        cpu_after = psutil.cpu_percent()
        memory_after = psutil.Process().memory_info().rss
        memory_used = memory_after - memory_before
        
        # Create metrics snapshot
        metrics = PerformanceMetrics(
            timestamp=start_time,
            inference_time=inference_time,
            memory_usage=memory_used,
            cpu_usage=(cpu_before + cpu_after) / 2,
            gpu_usage=0,  # Would need nvidia-smi for accurate GPU usage
            gpu_memory=gpu_memory_used,
            thermal_state=thermal_state,
            battery_level=battery_level,
            battery_drain_rate=battery_drain,
            cache_hit_rate=0,  # Would be provided by cache manager
            throughput=1.0 / inference_time if inference_time > 0 else 0,
            model_accuracy=0  # Would be calculated separately
        )
        
        # Store metrics if profiling session is active
        if self.profiling_active:
            self.session_metrics['inference_times'].append(inference_time)
            self.session_metrics['memory_usage'].append(memory_used)
            self.session_metrics['cpu_usage'].append(metrics.cpu_usage)
            self.session_metrics['gpu_memory'].append(gpu_memory_used)
            self.session_metrics['battery_drain'].append(battery_drain)
            self.session_metrics['throughput'].append(metrics.throughput)
        
        self.metrics_history.append(metrics)
        
        return result, metrics
    
    def benchmark_model(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        num_warmup: int = 10,
        num_iterations: int = 100,
        test_name: str = "default_benchmark"
    ) -> BenchmarkResult:
        """Comprehensive model benchmarking"""
        
        logger.info(f"Starting benchmark: {test_name}")
        
        # Start profiling session
        self.start_profiling_session()
        
        model.eval()
        inference_times = []
        memory_usage = []
        
        # Warmup
        logger.info(f"Warming up for {num_warmup} iterations...")
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= num_warmup:
                    break
                _ = model(data)
        
        # Actual benchmark
        logger.info(f"Running benchmark for {num_iterations} iterations...")
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= num_iterations:
                    break
                
                # Profile inference
                def inference():
                    return model(data)
                
                _, metrics = self.profile_inference(inference)
                inference_times.append(metrics.inference_time)
                memory_usage.append(metrics.memory_usage)
        
        # End profiling session
        session_summary = self.end_profiling_session()
        
        # Calculate benchmark results
        inference_times = np.array(inference_times)
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(session_summary)
        
        result = BenchmarkResult(
            test_name=test_name,
            device_info=self.device_info,
            avg_inference_time=np.mean(inference_times),
            p50_inference_time=np.percentile(inference_times, 50),
            p95_inference_time=np.percentile(inference_times, 95),
            p99_inference_time=np.percentile(inference_times, 99),
            max_memory_usage=max(memory_usage) if memory_usage else 0,
            avg_cpu_usage=session_summary['metrics_summary'].get('cpu_usage', {}).get('mean', 0),
            max_gpu_usage=0,  # Would need proper GPU monitoring
            thermal_throttling_detected=self.thermal_monitor.throttling_detected,
            battery_drain_per_hour=session_summary['metrics_summary'].get('battery_drain', {}).get('mean', 0),
            throughput_samples_per_sec=session_summary['metrics_summary'].get('throughput', {}).get('mean', 0),
            quality_score=0,  # Would need quality assessment
            optimization_recommendations=recommendations
        )
        
        self.benchmark_results.append(result)
        logger.info(f"Benchmark {test_name} completed")
        
        return result
    
    def detect_performance_regression(self, baseline_name: str, current_metrics: Dict) -> bool:
        """Detect performance regression compared to baseline"""
        if baseline_name not in self.baselines:
            # Store as new baseline
            self.baselines[baseline_name] = current_metrics
            return False
        
        baseline = self.baselines[baseline_name]
        
        # Check key performance indicators
        regression_detected = False
        
        for metric in ['avg_inference_time', 'max_memory_usage', 'battery_drain_per_hour']:
            if metric in baseline and metric in current_metrics:
                baseline_value = baseline[metric]
                current_value = current_metrics[metric]
                
                if baseline_value > 0:
                    degradation = (current_value - baseline_value) / baseline_value
                    if degradation > self.regression_threshold:
                        logger.warning(
                            f"Performance regression detected in {metric}: "
                            f"{degradation*100:.1f}% degradation"
                        )
                        regression_detected = True
        
        return regression_detected
    
    def _generate_optimization_recommendations(self, session_summary: Dict) -> List[str]:
        """Generate optimization recommendations based on profiling results"""
        recommendations = []
        
        metrics = session_summary.get('metrics_summary', {})
        
        # Memory recommendations
        avg_memory = metrics.get('memory_usage', {}).get('mean', 0)
        if avg_memory > 100 * 1024 * 1024:  # > 100MB
            recommendations.append("Consider enabling quantization to reduce memory usage")
        
        # Inference time recommendations
        avg_inference_time = metrics.get('inference_times', {}).get('mean', 0)
        if avg_inference_time > 0.2:  # > 200ms
            recommendations.append("Inference time is high - consider model pruning or hardware acceleration")
        
        # CPU usage recommendations
        avg_cpu = metrics.get('cpu_usage', {}).get('mean', 0)
        if avg_cpu > 80:
            recommendations.append("High CPU usage detected - consider reducing batch size or enabling GPU acceleration")
        
        # Battery recommendations
        avg_battery_drain = metrics.get('battery_drain', {}).get('mean', 0)
        if avg_battery_drain > 10:  # > 10% per hour
            recommendations.append("High battery drain - consider enabling power-efficient mode")
        
        # Thermal recommendations
        if self.thermal_monitor.throttling_detected:
            recommendations.append("Thermal throttling detected - consider reducing inference frequency or enabling cooling strategies")
        
        # Device-specific recommendations
        if self.device_info.memory_total < 4 * 1024**3:  # < 4GB
            recommendations.append("Low memory device - enable aggressive memory optimization")
        
        if not self.device_info.gpu_available:
            recommendations.append("No GPU available - focus on CPU optimizations and quantization")
        
        return recommendations
    
    def start_continuous_monitoring(self):
        """Start continuous performance monitoring in background"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Continuous performance monitoring started")
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)
        logger.info("Continuous performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.Process().memory_info().rss
                battery_level, battery_drain = self.battery_monitor.get_current_battery()
                thermal_state, throttling = self.thermal_monitor.get_thermal_state()
                
                # Store metrics
                metrics = PerformanceMetrics(
                    timestamp=time.time(),
                    inference_time=0,
                    memory_usage=memory_usage,
                    cpu_usage=cpu_usage,
                    gpu_usage=0,
                    gpu_memory=0,
                    thermal_state=thermal_state,
                    battery_level=battery_level,
                    battery_drain_rate=battery_drain,
                    cache_hit_rate=0,
                    throughput=0,
                    model_accuracy=0
                )
                
                self.metrics_history.append(metrics)
                
                # Check for alerts
                if throttling:
                    logger.warning("Thermal throttling detected during monitoring")
                
                if battery_drain > 15:  # > 15% per hour
                    logger.warning(f"High battery drain detected: {battery_drain:.1f}% per hour")
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def export_profiling_report(self, output_path: str):
        """Export comprehensive profiling report"""
        report = {
            'device_info': asdict(self.device_info),
            'benchmark_results': [asdict(result) for result in self.benchmark_results],
            'metrics_history': [asdict(metric) for metric in list(self.metrics_history)],
            'baselines': self.baselines,
            'timestamp': time.time(),
            'profiler_config': {
                'continuous_monitoring': self.enable_continuous_monitoring,
                'regression_threshold': self.regression_threshold
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Profiling report exported to {output_path}")
    
    def get_real_time_metrics(self) -> Dict:
        """Get current real-time metrics"""
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        return asdict(latest_metrics)
    
    def get_performance_trends(self, window_size: int = 100) -> Dict:
        """Analyze performance trends over recent measurements"""
        if len(self.metrics_history) < window_size:
            return {}
        
        recent_metrics = list(self.metrics_history)[-window_size:]
        
        # Extract time series for key metrics
        timestamps = [m.timestamp for m in recent_metrics]
        inference_times = [m.inference_time for m in recent_metrics if m.inference_time > 0]
        memory_usage = [m.memory_usage for m in recent_metrics]
        cpu_usage = [m.cpu_usage for m in recent_metrics]
        
        trends = {}
        
        # Calculate trends for each metric
        if len(inference_times) > 1:
            inference_trend = np.polyfit(range(len(inference_times)), inference_times, 1)[0]
            trends['inference_time_trend'] = 'increasing' if inference_trend > 0 else 'decreasing'
        
        if len(memory_usage) > 1:
            memory_trend = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
            trends['memory_trend'] = 'increasing' if memory_trend > 0 else 'decreasing'
        
        if len(cpu_usage) > 1:
            cpu_trend = np.polyfit(range(len(cpu_usage)), cpu_usage, 1)[0]
            trends['cpu_trend'] = 'increasing' if cpu_trend > 0 else 'decreasing'
        
        return trends
