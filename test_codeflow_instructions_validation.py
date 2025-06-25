#!/usr/bin/env python3
"""
Comprehensive test suite for validating the Codeflow Instructions system integration.

This test validates that all core systems (Pattern-First Development, Continuous Integration,
Documentation-First Development, and Architecture-First Development) are properly integrated
and functioning as designed in the codeflow.instructions.md file.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


class CodeflowInstructionsValidator:
    """Validates the codeflow.instructions.md file for completeness and consistency."""
    
    def __init__(self, instructions_file: str):
        self.instructions_file = Path(instructions_file)
        self.content = ""
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.load_content()
    
    def load_content(self):
        """Load the instructions file content."""
        try:
            with open(self.instructions_file, 'r', encoding='utf-8') as f:
                self.content = f.read()
        except Exception as e:
            self.errors.append(f"Failed to load instructions file: {e}")
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation tests."""
        results = {
            "file_structure": self.validate_file_structure(),
            "core_principles": self.validate_core_principles(),
            "pattern_system": self.validate_pattern_system(),
            "integration_system": self.validate_integration_system(),
            "documentation_system": self.validate_documentation_system(),
            "architecture_system": self.validate_architecture_system(),
            "workflow_steps": self.validate_workflow_steps(),
            "cross_references": self.validate_cross_references(),
            "completeness": self.validate_completeness(),
            "consistency": self.validate_consistency()
        }
        
        # Summary
        total_errors = len(self.errors)
        total_warnings = len(self.warnings)
        results["summary"] = {
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "status": "PASS" if total_errors == 0 else "FAIL",
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        return results
    
    def validate_file_structure(self) -> Dict[str, Any]:
        """Validate the basic file structure and sections."""
        results = {"status": "PASS", "details": []}
        
        required_sections = [
            "# GitHub Copilot Instructions: Codeflow System",
            "## System Overview",
            "### Core Principles",
            "## Project Structure",
            "### `.project_meta` Structure",
            "## Core Workflow Steps"
        ]
        
        for section in required_sections:
            if section not in self.content:
                self.errors.append(f"Missing required section: {section}")
                results["status"] = "FAIL"
            else:
                results["details"].append(f"Found section: {section}")
        
        return results
    
    def validate_core_principles(self) -> Dict[str, Any]:
        """Validate that all core principles are present and properly defined."""
        results = {"status": "PASS", "details": []}
        
        required_principles = [
            "Workflow Stability Principle",
            "Verification Principle", 
            "Traceability Principle",
            "Documentation Principle",
            "Context7 Documentation Principle",
            "Pattern-First Development Principle",
            "Continuous Integration Principle",
            "Documentation-First Development Principle",
            "Architecture-First Development Principle"
        ]
        
        for principle in required_principles:
            if f"**{principle}:**" not in self.content:
                self.errors.append(f"Missing core principle: {principle}")
                results["status"] = "FAIL"
            else:
                results["details"].append(f"Found principle: {principle}")
        
        return results
    
    def validate_pattern_system(self) -> Dict[str, Any]:
        """Validate the Pattern-First Development system integration."""
        results = {"status": "PASS", "details": []}
        
        # Check for pattern directory structure
        pattern_files = [
            "pattern_catalog.json",
            "anti_patterns.json",
            "pattern_schema.json",
            "pattern_usage_tracker.json",
            "pattern_effectiveness.json",
            "pattern_recommendations.json",
            "pre_implementation_checks.json",
            "post_implementation_analysis.json"
        ]
        
        for file in pattern_files:
            if file not in self.content:
                self.errors.append(f"Missing pattern file in structure: {file}")
                results["status"] = "FAIL"
            else:
                results["details"].append(f"Found pattern file: {file}")
        
        # Check for pattern principles implementation
        pattern_requirements = [
            "Pre-Implementation Pattern Consultation",
            "Pattern Application Assessment",
            "Pattern Compliance Verification",
            "Post-Implementation Pattern Discovery",
            "Pattern Catalog Maintenance"
        ]
        
        for req in pattern_requirements:
            if req not in self.content:
                self.errors.append(f"Missing pattern requirement: {req}")
                results["status"] = "FAIL"
            else:
                results["details"].append(f"Found pattern requirement: {req}")
        
        return results
    
    def validate_integration_system(self) -> Dict[str, Any]:
        """Validate the Continuous Integration system implementation."""
        results = {"status": "PASS", "details": []}
        
        # Check for integration directory structure
        integration_files = [
            "integration_status.json",
            "integration_strategy.json",
            "integration_requirements.json",
            "integration_schedule.json",
            "integration_checkpoints.json",
            "real_time_monitoring.json",
            "integration_feedback.json",
            "integration_risk_assessment.json",
            "continuous_integration_config.json"
        ]
        
        for file in integration_files:
            if file not in self.content:
                self.errors.append(f"Missing integration file in structure: {file}")
                results["status"] = "FAIL"
            else:
                results["details"].append(f"Found integration file: {file}")
        
        # Check for integration principles implementation
        integration_requirements = [
            "Pre-Implementation Integration Planning",
            "Development-Time Integration Monitoring",
            "Real-Time Integration Testing",
            "Integration Quality Gates",
            "Continuous Integration Feedback",
            "Integration Risk Management"
        ]
        
        for req in integration_requirements:
            if req not in self.content:
                self.errors.append(f"Missing integration requirement: {req}")
                results["status"] = "FAIL"
            else:
                results["details"].append(f"Found integration requirement: {req}")
        
        return results
    
    def validate_documentation_system(self) -> Dict[str, Any]:
        """Validate the Documentation-First Development system implementation."""
        results = {"status": "PASS", "details": []}
        
        # Check for documentation directory structure
        documentation_files = [
            "documentation_strategy.json",
            "documentation_requirements.json",
            "documentation_schedule.json",
            "documentation_quality_gates.json",
            "real_time_validation.json",
            "documentation_feedback.json",
            "documentation_usage_analytics.json",
            "documentation_automation.json",
            "content_management.json",
            "documentation_templates.json"
        ]
        
        for file in documentation_files:
            if file not in self.content:
                self.errors.append(f"Missing documentation file in structure: {file}")
                results["status"] = "FAIL"
            else:
                results["details"].append(f"Found documentation file: {file}")
        
        # Check for documentation principles implementation
        documentation_requirements = [
            "Pre-Implementation Documentation Planning",
            "Documentation-Driven Development",
            "Real-Time Documentation Maintenance",
            "Documentation Quality Gates",
            "Continuous Documentation Validation",
            "Documentation Usability Testing",
            "Documentation Metrics Monitoring",
            "Documentation Feedback Integration"
        ]
        
        for req in documentation_requirements:
            if req not in self.content:
                self.errors.append(f"Missing documentation requirement: {req}")
                results["status"] = "FAIL"
            else:
                results["details"].append(f"Found documentation requirement: {req}")
        
        return results
    
    def validate_architecture_system(self) -> Dict[str, Any]:
        """Validate the Architecture-First Development system implementation."""
        results = {"status": "PASS", "details": []}
        
        # Check for architecture directory structure
        architecture_files = [
            "architecture_strategy.json",
            "architecture_requirements.json",
            "architecture_schedule.json",
            "architecture_quality_gates.json",
            "real_time_validation.json",
            "architecture_feedback.json",
            "architecture_impact_analysis.json",
            "architecture_automation.json",
            "architecture_governance.json",
            "architecture_evolution.json",
            "architecture_templates.json"
        ]
        
        for file in architecture_files:
            if file not in self.content:
                self.errors.append(f"Missing architecture file in structure: {file}")
                results["status"] = "FAIL"
            else:
                results["details"].append(f"Found architecture file: {file}")
        
        # Check for architecture principles implementation
        architecture_requirements = [
            "Pre-Implementation Architecture Analysis",
            "Architecture-Driven Development",
            "Real-Time Architecture Validation",
            "Architecture Quality Gates",
            "Continuous Architecture Monitoring",
            "Architecture Impact Assessment",
            "Architecture Feedback Integration",
            "Architecture Decision Tracking"
        ]
        
        for req in architecture_requirements:
            if req not in self.content:
                self.errors.append(f"Missing architecture requirement: {req}")
                results["status"] = "FAIL"
            else:
                results["details"].append(f"Found architecture requirement: {req}")
        
        return results
    
    def validate_workflow_steps(self) -> Dict[str, Any]:
        """Validate that all workflow steps are properly integrated."""
        results = {"status": "PASS", "details": []}
        
        # Check for mandatory workflow steps
        required_steps = [
            "### 0. Context7 Documentation Fetch & Analysis",
            "### 1. Initialize Project", 
            "### 2. Load Context from Chats",
            "### 3. Check/Generate PRD",
            "### 4. Analyze Initial Architecture",
            "### 5. Define Modular Structure",
            "### 6. Create Roadmap and Stories",
            "### 7. Execute Next Story"
        ]
        
        for step in required_steps:
            if step not in self.content:
                self.errors.append(f"Missing workflow step: {step}")
                results["status"] = "FAIL"
            else:
                results["details"].append(f"Found workflow step: {step}")
        
        # Check for integration of systems in workflow steps
        workflow_integrations = [
            "MANDATORY Pre-Implementation Architecture Planning",
            "Pattern-First Development",
            "Documentation-First Development", 
            "Architecture-First Development",
            "Integration-Enhanced",
            "Context7-Enhanced"
        ]
        
        for integration in workflow_integrations:
            if integration not in self.content:
                self.warnings.append(f"Workflow integration may be missing: {integration}")
            else:
                results["details"].append(f"Found workflow integration: {integration}")
        
        return results
    
    def validate_cross_references(self) -> Dict[str, Any]:
        """Validate cross-references between systems."""
        results = {"status": "PASS", "details": []}
        
        # Check for proper cross-referencing between systems
        cross_refs = [
            ".project_meta/.patterns/",
            ".project_meta/.integration/",
            ".project_meta/.docs/",
            ".project_meta/.architecture/",
            ".project_meta/.context7/"
        ]
        
        for ref in cross_refs:
            count = self.content.count(ref)
            if count == 0:
                self.errors.append(f"Missing cross-references to: {ref}")
                results["status"] = "FAIL"
            else:
                results["details"].append(f"Found {count} references to: {ref}")
        
        return results
    
    def validate_completeness(self) -> Dict[str, Any]:
        """Validate overall completeness of the instructions."""
        results = {"status": "PASS", "details": []}
        
        # Check for essential components
        essential_components = [
            "Error Handling:",
            "CRITICAL REQUIREMENT:",
            "**Actions:**",
            "**Purpose:**",
            "json",
            "workflow",
            "verification",
            "validation"
        ]
        
        for component in essential_components:
            count = self.content.count(component)
            if count == 0:
                self.warnings.append(f"No instances of essential component: {component}")
            else:
                results["details"].append(f"Found {count} instances of: {component}")
        
        # Check file length and complexity
        line_count = len(self.content.split('\n'))
        if line_count < 2000:
            self.warnings.append(f"Instructions file seems short: {line_count} lines")
        else:
            results["details"].append(f"Instructions file has {line_count} lines")
        
        return results
    
    def validate_consistency(self) -> Dict[str, Any]:
        """Validate consistency across the instructions."""
        results = {"status": "PASS", "details": []}
        
        # Check for consistent terminology
        terminology_checks = [
            ("Pattern-First", "pattern"),
            ("Documentation-First", "documentation"),
            ("Architecture-First", "architecture"),
            ("Integration", "integration"),
            ("Context7", "context7")
        ]
        
        for term, related in terminology_checks:
            term_count = self.content.lower().count(term.lower())
            related_count = self.content.lower().count(related.lower())
            
            if term_count == 0:
                self.errors.append(f"Missing key terminology: {term}")
                results["status"] = "FAIL"
            elif related_count < term_count:
                self.warnings.append(f"Possible inconsistent usage of {term} vs {related}")
            else:
                results["details"].append(f"Consistent usage of {term} ({term_count}) and {related} ({related_count})")
        
        return results


def main():
    """Main function to run the validation."""
    instructions_file = "/Users/yunusgungor/work/mobilemodel/.github/instructions/codeflow.instructions.md"
    
    if not Path(instructions_file).exists():
        print(f"ERROR: Instructions file not found: {instructions_file}")
        sys.exit(1)
    
    print("=== Codeflow Instructions Validation ===")
    print(f"Validating: {instructions_file}")
    print()
    
    validator = CodeflowInstructionsValidator(instructions_file)
    results = validator.validate_all()
    
    # Print detailed results
    for test_name, test_results in results.items():
        if test_name == "summary":
            continue
            
        print(f"## {test_name.replace('_', ' ').title()}")
        print(f"Status: {test_results.get('status', 'UNKNOWN')}")
        
        if 'details' in test_results:
            for detail in test_results['details'][:5]:  # Show first 5 details
                print(f"  ✓ {detail}")
            if len(test_results['details']) > 5:
                print(f"  ... and {len(test_results['details']) - 5} more")
        print()
    
    # Print summary
    summary = results["summary"]
    print("=== VALIDATION SUMMARY ===")
    print(f"Status: {summary['status']}")
    print(f"Total Errors: {summary['total_errors']}")
    print(f"Total Warnings: {summary['total_warnings']}")
    
    if summary['errors']:
        print("\nErrors:")
        for error in summary['errors']:
            print(f"  ❌ {error}")
    
    if summary['warnings']:
        print("\nWarnings:")
        for warning in summary['warnings'][:10]:  # Show first 10 warnings
            print(f"  ⚠️  {warning}")
        if len(summary['warnings']) > 10:
            print(f"  ... and {len(summary['warnings']) - 10} more warnings")
    
    # Generate test report
    report = {
        "validation_results": results,
        "timestamp": "2024-12-26",
        "validator_version": "1.0.0",
        "instructions_file": str(instructions_file)
    }
    
    report_file = "/Users/yunusgungor/work/mobilemodel/codeflow_validation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    return 0 if summary['status'] == 'PASS' else 1


if __name__ == "__main__":
    sys.exit(main())
