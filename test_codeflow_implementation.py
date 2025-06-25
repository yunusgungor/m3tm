#!/usr/bin/env python3
"""
Implementation test for the Codeflow System - creates the actual directory structure
and validates that all integrated systems can be initialized properly.

This test simulates the "Initialize Project" step from the codeflow instructions
to ensure all systems work together correctly.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any
import tempfile


class CodeflowSystemImplementationTest:
    """Tests the actual implementation of the Codeflow System."""
    
    def __init__(self, test_root: str = None):
        self.test_root = Path(test_root) if test_root else Path(tempfile.mkdtemp(prefix="codeflow_test_"))
        self.project_meta = self.test_root / ".project_meta"
        self.results = {"created_dirs": [], "created_files": [], "errors": []}
    
    def setup_test_environment(self):
        """Set up the test environment."""
        print(f"Setting up test environment in: {self.test_root}")
        self.test_root.mkdir(exist_ok=True, parents=True)
    
    def create_project_structure(self):
        """Create the complete .project_meta structure as defined in instructions."""
        print("Creating .project_meta directory structure...")
        
        # Define the complete directory structure from the instructions
        directories = [
            ".project_meta/.chats",
            ".project_meta/.stories",
            ".project_meta/.stories/mappings",
            ".project_meta/.stories/metrics",
            ".project_meta/.stories/versions",
            ".project_meta/.stories/visualizations",
            ".project_meta/.stories/templates",
            ".project_meta/.docs",
            ".project_meta/.docs/api",
            ".project_meta/.docs/architecture",
            ".project_meta/.docs/guides",
            ".project_meta/.docs/tutorials",
            ".project_meta/.docs/maintenance",
            ".project_meta/.docs/metrics",
            ".project_meta/.docs/versions",
            ".project_meta/.docs/interactive",
            ".project_meta/.docs/validation",
            ".project_meta/.docs/validation/validation_reports",
            ".project_meta/.docs/automation",
            ".project_meta/.docs/analytics",
            ".project_meta/.docs/feedback",
            ".project_meta/.docs/search",
            ".project_meta/.docs/templates",
            ".project_meta/.patterns",
            ".project_meta/.patterns/metrics",
            ".project_meta/.patterns/evolution",
            ".project_meta/.patterns/relationships",
            ".project_meta/.patterns/visualization",
            ".project_meta/.patterns/templates",
            ".project_meta/.patterns/reviews",
            ".project_meta/.architecture",
            ".project_meta/.architecture/architecture_metrics",
            ".project_meta/.architecture/component_specifications",
            ".project_meta/.architecture/models",
            ".project_meta/.architecture/visualizations",
            ".project_meta/.architecture/reviews",
            ".project_meta/.architecture/validation",
            ".project_meta/.architecture/validation/validation_reports",
            ".project_meta/.architecture/automation",
            ".project_meta/.architecture/analytics",
            ".project_meta/.architecture/feedback",
            ".project_meta/.context7",
            ".project_meta/.context7/fetched_docs",
            ".project_meta/.context7/fetched_docs/frameworks",
            ".project_meta/.context7/fetched_docs/libraries",
            ".project_meta/.context7/fetched_docs/apis",
            ".project_meta/.context7/fetched_docs/guides",
            ".project_meta/.context7/validation_reports",
            ".project_meta/.integration",
            ".project_meta/.integration/metrics",
            ".project_meta/.integration/reports",
            ".project_meta/.integration/logs",
            ".project_meta/.integration/logs/integration_execution_logs",
            ".project_meta/.integration/logs/integration_error_logs",
            ".project_meta/.integration/logs/integration_performance_logs",
            ".project_meta/.integration/logs/integration_audit_trail",
            ".project_meta/.integration/visualization",
            ".project_meta/.integration/visualization/integration_dashboard",
            ".project_meta/.integration/visualization/integration_flow_diagrams",
            ".project_meta/.integration/visualization/integration_dependency_maps",
            ".project_meta/.integration/visualization/integration_health_charts",
            ".project_meta/.integration/templates",
            ".project_meta/.integration/templates/integration_test_templates",
            ".project_meta/.integration/templates/integration_strategy_templates",
            ".project_meta/.integration/templates/integration_documentation_templates",
            ".project_meta/.integration/validation",
            ".project_meta/.integration/validation/pre_integration_checks",
            ".project_meta/.integration/validation/integration_compliance_checks",
            ".project_meta/.integration/validation/post_integration_validation",
            ".project_meta/.dependencies",
            ".project_meta/.errors",
            ".project_meta/.errors/metrics",
            ".project_meta/.errors/reports",
            ".project_meta/.errors/reports/cascading_failures",
            ".project_meta/.errors/reports/visualizations",
            ".project_meta/.decisions"
        ]
        
        # Create all directories
        for dir_path in directories:
            full_path = self.test_root / dir_path
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                self.results["created_dirs"].append(str(full_path))
            except Exception as e:
                self.results["errors"].append(f"Failed to create directory {dir_path}: {e}")
    
    def create_initial_json_files(self):
        """Create all initial JSON files with default content as specified in instructions."""
        print("Creating initial JSON files...")
        
        # Define all JSON files with their initial content
        json_files = {
            # Context7 Files
            ".project_meta/.context7/doc_metadata.json": {"technologies": [], "last_updated": None},
            ".project_meta/.context7/tech_stack_docs.json": {"identified_technologies": [], "documentation_status": {}},
            ".project_meta/.context7/last_fetch_timestamps.json": {},
            
            # Stories Files
            ".project_meta/.stories/roadmap.json": {"current_iteration_id": None, "stories": [], "iterations": []},
            ".project_meta/.stories/mappings/story_module_map.json": {},
            ".project_meta/.stories/mappings/story_requirement_map.json": {},
            ".project_meta/.stories/mappings/story_dependency_map.json": {"nodes": [], "edges": []},
            ".project_meta/.stories/mappings/story_architectural_impact_map.json": {},
            ".project_meta/.stories/metrics/velocity_metrics.json": {},
            ".project_meta/.stories/metrics/estimation_accuracy.json": {},
            ".project_meta/.stories/metrics/story_completion_trend.json": {},
            ".project_meta/.stories/metrics/dependency_health_index.json": {},
            ".project_meta/.stories/metrics/milestone_progress.json": {},
            ".project_meta/.stories/metrics/story_quality.json": {},
            ".project_meta/.stories/metrics/forecasting_metrics.json": {},
            
            # Architecture Files
            ".project_meta/.architecture/adr_log.json": [],
            ".project_meta/.architecture/module_definitions.json": {"modules": []},
            ".project_meta/.architecture/architecture_constraints.json": [],
            ".project_meta/.architecture/architecture_strategy.json": {"strategy": "architecture_first", "approach": "continuous", "quality_gates": []},
            ".project_meta/.architecture/architecture_requirements.json": {"requirements": [], "constraints": [], "quality_attributes": {}},
            ".project_meta/.architecture/architecture_schedule.json": {"milestones": [], "deadlines": [], "review_schedule": {}},
            ".project_meta/.architecture/architecture_quality_gates.json": {"quality_gates": [], "validation_points": []},
            ".project_meta/.architecture/real_time_validation.json": {"validators": [], "alerts": [], "thresholds": {}},
            ".project_meta/.architecture/architecture_feedback.json": {"feedback_loops": [], "improvement_suggestions": []},
            ".project_meta/.architecture/architecture_impact_analysis.json": {"impact_tracking": [], "change_analysis": []},
            ".project_meta/.architecture/architecture_automation.json": {"automation_rules": [], "triggers": []},
            ".project_meta/.architecture/architecture_governance.json": {"governance_rules": [], "compliance_checks": []},
            ".project_meta/.architecture/architecture_evolution.json": {"evolution_tracking": [], "version_history": []},
            ".project_meta/.architecture/architecture_templates.json": {"templates": [], "standards": {}},
            ".project_meta/.architecture/architecture_metrics/conformance_score.json": {},
            ".project_meta/.architecture/architecture_metrics/drift_metrics.json": {},
            ".project_meta/.architecture/architecture_metrics/component_conformance.json": {},
            ".project_meta/.architecture/architecture_metrics/coupling_metrics.json": {},
            ".project_meta/.architecture/architecture_metrics/cohesion_metrics.json": {},
            ".project_meta/.architecture/architecture_metrics/performance_qa.json": {},
            ".project_meta/.architecture/architecture_metrics/security_qa.json": {},
            ".project_meta/.architecture/architecture_metrics/maintainability_qa.json": {},
            ".project_meta/.architecture/architecture_metrics/scalability_qa.json": {},
            ".project_meta/.architecture/architecture_metrics/tech_debt.json": {},
            ".project_meta/.architecture/reviews/architecture_review_summary.json": {},
            
            # Integration Files
            ".project_meta/.integration/integration_status.json": {"overall_status": "pending", "last_run": None, "component_status": {}},
            ".project_meta/.integration/integration_strategy.json": {"strategy": "continuous", "approach": "incremental", "quality_gates": []},
            ".project_meta/.integration/integration_requirements.json": {"contracts": [], "interfaces": [], "dependencies": []},
            ".project_meta/.integration/integration_schedule.json": {"milestones": [], "checkpoints": [], "timeline": {}},
            ".project_meta/.integration/integration_checkpoints.json": {"quality_gates": [], "validation_points": []},
            ".project_meta/.integration/real_time_monitoring.json": {"monitors": [], "alerts": [], "thresholds": {}},
            ".project_meta/.integration/integration_feedback.json": {"feedback_loops": [], "recommendations": []},
            ".project_meta/.integration/integration_risk_assessment.json": {"risks": [], "mitigation_strategies": []},
            ".project_meta/.integration/continuous_integration_config.json": {"pipelines": [], "automation_rules": []},
            ".project_meta/.integration/integration_test_suites.json": {"test_suites": [], "test_configurations": []},
            ".project_meta/.integration/integration_environments.json": {"environments": [], "configurations": []},
            ".project_meta/.integration/integration_automation.json": {"automation_rules": [], "triggers": []},
            ".project_meta/.integration/metrics/stability_index.json": {},
            ".project_meta/.integration/metrics/coverage_report.json": {},
            ".project_meta/.integration/metrics/test_performance.json": {},
            ".project_meta/.integration/metrics/interface_compliance.json": {},
            ".project_meta/.integration/metrics/integration_debt.json": {},
            ".project_meta/.integration/metrics/architecture_alignment.json": {},
            ".project_meta/.integration/metrics/environment_health.json": {},
            ".project_meta/.integration/metrics/integration_velocity.json": {},
            ".project_meta/.integration/metrics/integration_quality_score.json": {},
            ".project_meta/.integration/metrics/integration_risk_metrics.json": {},
            ".project_meta/.integration/metrics/integration_efficiency.json": {},
            ".project_meta/.integration/metrics/integration_success_rate.json": {},
            
            # Dependencies Files
            ".project_meta/.dependencies/dependency_graph.json": {"nodes": [], "edges": []},
            ".project_meta/.dependencies/conflict_log.json": [],
            
            # Error Management Files
            ".project_meta/.errors/error_log.json": [],
            ".project_meta/.errors/metrics/effectiveness_score.json": {},
            ".project_meta/.errors/metrics/resolution_efficiency.json": {},
            ".project_meta/.errors/metrics/distribution_analysis.json": {},
            ".project_meta/.errors/metrics/error_trends.json": {},
            ".project_meta/.errors/metrics/prediction_accuracy.json": {},
            ".project_meta/.errors/metrics/critical_error_analysis.json": {},
            
            # Documentation Files
            ".project_meta/.docs/documentation_strategy.json": {"strategy": "documentation_first", "approach": "continuous", "quality_gates": []},
            ".project_meta/.docs/documentation_requirements.json": {"requirements": [], "coverage_targets": {}, "quality_standards": {}},
            ".project_meta/.docs/documentation_schedule.json": {"milestones": [], "deadlines": [], "maintenance_schedule": {}},
            ".project_meta/.docs/documentation_quality_gates.json": {"quality_gates": [], "validation_points": []},
            ".project_meta/.docs/real_time_validation.json": {"validators": [], "alerts": [], "thresholds": {}},
            ".project_meta/.docs/documentation_feedback.json": {"feedback_loops": [], "improvement_suggestions": []},
            ".project_meta/.docs/documentation_usage_analytics.json": {"tracking_enabled": True, "metrics_collection": []},
            ".project_meta/.docs/documentation_automation.json": {"automation_rules": [], "triggers": []},
            ".project_meta/.docs/content_management.json": {"content_lifecycle": [], "review_cycles": []},
            ".project_meta/.docs/documentation_templates.json": {"templates": [], "standards": {}},
            ".project_meta/.docs/metrics/doc_quality_metrics.json": {},
            ".project_meta/.docs/metrics/doc_coverage_report.json": {},
            ".project_meta/.docs/metrics/doc_usage_analytics.json": {},
            ".project_meta/.docs/metrics/doc_freshness_index.json": {},
            
            # Pattern Files
            ".project_meta/.patterns/pattern_catalog.json": [],
            ".project_meta/.patterns/anti_patterns.json": [],
            ".project_meta/.patterns/pattern_schema.json": {"version": "1.0", "schema": {}},
            ".project_meta/.patterns/pattern_usage_tracker.json": {},
            ".project_meta/.patterns/pattern_effectiveness.json": {},
            ".project_meta/.patterns/pattern_recommendations.json": {},
            ".project_meta/.patterns/pre_implementation_checks.json": {"checks": [], "last_run": None},
            ".project_meta/.patterns/post_implementation_analysis.json": {"analyses": [], "last_run": None},
            ".project_meta/.patterns/metrics/pattern_metrics.json": {},
            ".project_meta/.patterns/metrics/pattern_history.json": {},
            ".project_meta/.patterns/metrics/pattern_adoption_rate.json": {},
            ".project_meta/.patterns/metrics/pattern_compliance_score.json": {},
            ".project_meta/.patterns/metrics/pattern_impact_analysis.json": {},
            
            # Decision Files
            ".project_meta/.decisions/decision_log.json": []
        }
        
        # Create all JSON files
        for file_path, content in json_files.items():
            full_path = self.test_root / file_path
            try:
                # Ensure parent directory exists
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write JSON file
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2)
                
                self.results["created_files"].append(str(full_path))
            except Exception as e:
                self.results["errors"].append(f"Failed to create file {file_path}: {e}")
    
    def create_markdown_files(self):
        """Create essential markdown files."""
        print("Creating markdown files...")
        
        markdown_files = {
            ".project_meta/.architecture/coding_standards.md": "# Coding Standards\n\nThis document defines coding standards for the project.\n",
            ".project_meta/.architecture/architecture_principles.md": "# Architecture Principles\n\nThis document defines architecture principles for the project.\n",
            ".project_meta/.architecture/technology_stack.md": "# Technology Stack\n\nThis document defines the technology selections for the project.\n",
            ".project_meta/.architecture/issues.md": "# Architecture Issues\n\nThis document tracks architecture concerns and issues.\n",
            ".project_meta/.docs/index.md": "# Documentation Index\n\nMain entry point for project documentation.\n"
        }
        
        for file_path, content in markdown_files.items():
            full_path = self.test_root / file_path
            try:
                # Ensure parent directory exists
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write markdown file
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.results["created_files"].append(str(full_path))
            except Exception as e:
                self.results["errors"].append(f"Failed to create file {file_path}: {e}")
    
    def validate_structure(self) -> Dict[str, Any]:
        """Validate that the created structure matches the instructions."""
        print("Validating created structure...")
        
        validation_results = {
            "directories_created": len(self.results["created_dirs"]),
            "files_created": len(self.results["created_files"]),
            "errors": len(self.results["errors"]),
            "status": "PASS" if len(self.results["errors"]) == 0 else "FAIL",
            "structure_integrity": True
        }
        
        # Check key directories exist
        key_directories = [
            ".project_meta/.patterns",
            ".project_meta/.integration", 
            ".project_meta/.docs",
            ".project_meta/.architecture",
            ".project_meta/.context7"
        ]
        
        missing_dirs = []
        for dir_path in key_directories:
            full_path = self.test_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
                validation_results["structure_integrity"] = False
        
        validation_results["missing_directories"] = missing_dirs
        
        # Check key files exist and are valid JSON
        key_files = [
            ".project_meta/.patterns/pattern_catalog.json",
            ".project_meta/.integration/integration_status.json",
            ".project_meta/.docs/documentation_strategy.json",
            ".project_meta/.architecture/architecture_strategy.json",
            ".project_meta/.context7/doc_metadata.json"
        ]
        
        missing_files = []
        invalid_json = []
        
        for file_path in key_files:
            full_path = self.test_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
                validation_results["structure_integrity"] = False
            else:
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    invalid_json.append(f"{file_path}: {e}")
                    validation_results["structure_integrity"] = False
        
        validation_results["missing_files"] = missing_files
        validation_results["invalid_json"] = invalid_json
        
        return validation_results
    
    def run_full_test(self) -> Dict[str, Any]:
        """Run the complete implementation test."""
        print("=== Codeflow System Implementation Test ===")
        
        try:
            self.setup_test_environment()
            self.create_project_structure()
            self.create_initial_json_files()
            self.create_markdown_files()
            validation_results = self.validate_structure()
            
            test_results = {
                "test_environment": str(self.test_root),
                "creation_results": self.results,
                "validation_results": validation_results,
                "overall_status": "PASS" if validation_results["status"] == "PASS" and len(self.results["errors"]) == 0 else "FAIL"
            }
            
            return test_results
            
        except Exception as e:
            return {
                "test_environment": str(self.test_root),
                "creation_results": self.results,
                "validation_results": {"status": "FAIL", "error": str(e)},
                "overall_status": "FAIL"
            }
    
    def cleanup(self):
        """Clean up the test environment."""
        if self.test_root.exists():
            shutil.rmtree(self.test_root)
            print(f"Cleaned up test environment: {self.test_root}")


def main():
    """Main function to run the implementation test."""
    test = CodeflowSystemImplementationTest()
    
    try:
        results = test.run_full_test()
        
        # Print results
        print(f"\nTest Environment: {results['test_environment']}")
        print(f"Overall Status: {results['overall_status']}")
        
        if results["creation_results"]["errors"]:
            print("\nCreation Errors:")
            for error in results["creation_results"]["errors"]:
                print(f"  ❌ {error}")
        
        validation = results["validation_results"]
        print(f"\nValidation Results:")
        print(f"  Directories Created: {validation.get('directories_created', 0)}")
        print(f"  Files Created: {validation.get('files_created', 0)}")
        print(f"  Structure Integrity: {validation.get('structure_integrity', False)}")
        
        if validation.get("missing_directories"):
            print("  Missing Directories:")
            for dir_path in validation["missing_directories"]:
                print(f"    ❌ {dir_path}")
        
        if validation.get("missing_files"):
            print("  Missing Files:")
            for file_path in validation["missing_files"]:
                print(f"    ❌ {file_path}")
        
        if validation.get("invalid_json"):
            print("  Invalid JSON Files:")
            for error in validation["invalid_json"]:
                print(f"    ❌ {error}")
        
        # Save results
        results_file = "/Users/yunusgungor/work/mobilemodel/codeflow_implementation_test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
        
        if results["overall_status"] == "PASS":
            print("\n🎉 Implementation test PASSED! The Codeflow System can be successfully initialized.")
        else:
            print("\n❌ Implementation test FAILED! There are issues with the system initialization.")
        
        return 0 if results["overall_status"] == "PASS" else 1
        
    finally:
        # Always ask before cleanup in case user wants to inspect
        user_input = input("\nDo you want to keep the test environment for inspection? (y/N): ")
        if user_input.lower() not in ['y', 'yes']:
            test.cleanup()
        else:
            print(f"Test environment preserved at: {test.test_root}")


if __name__ == "__main__":
    exit(main())
