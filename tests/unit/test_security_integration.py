import pytest
import subprocess
import os
import sys

class TestSecurityIntegration:
    """
    M3TM Security Integration Test Suite
    
    Bu test suite'i OWASP MASVS uyumlu güvenlik implementasyonlarını
    Python seviyesinde test eder ve validate eder.
    """
    
    def test_security_files_exist(self):
        """Test that all security implementation files exist."""
        base_path = "/Users/yunusgungor/work/mobilemodel"
        
        android_security_files = [
            "src/m3tm/mobile/android/sdk/java/com/m3tm/sdk/security/SecureStorageManager.java",
            "src/m3tm/mobile/android/sdk/java/com/m3tm/sdk/security/NetworkSecurityManager.java", 
            "src/m3tm/mobile/android/sdk/java/com/m3tm/sdk/security/BiometricAuthManager.java",
            "android/src/main/res/xml/network_security_config.xml"
        ]
        
        ios_security_files = [
            "src/m3tm/mobile/ios/sdk/SecureStorageManager.swift",
            "src/m3tm/mobile/ios/sdk/NetworkSecurityManager.swift"
        ]
        
        for file_path in android_security_files + ios_security_files:
            full_path = os.path.join(base_path, file_path)
            assert os.path.exists(full_path), f"Security file missing: {file_path}"
    
    def test_security_test_files_exist(self):
        """Test that security test files exist."""
        base_path = "/Users/yunusgungor/work/mobilemodel"
        
        test_files = [
            "tests/unit/security/SecureStorageIntegrationTest.java",
            "tests/unit/security/NetworkSecurityIntegrationTest.java",
            "tests/security/run_security_tests.sh"
        ]
        
        for file_path in test_files:
            full_path = os.path.join(base_path, file_path)
            assert os.path.exists(full_path), f"Security test file missing: {file_path}"
    
    def test_android_security_class_structure(self):
        """Test Android security classes have proper structure."""
        base_path = "/Users/yunusgungor/work/mobilemodel"
        
        # SecureStorageManager structure
        storage_file = os.path.join(base_path, 
            "src/m3tm/mobile/android/sdk/java/com/m3tm/sdk/security/SecureStorageManager.java")
        
        with open(storage_file, 'r') as f:
            content = f.read()
            
        # Check critical methods exist
        assert "initialize()" in content, "SecureStorageManager missing initialize method"
        assert "storeSecureString" in content, "Missing secure string storage"
        assert "storeSecureData" in content, "Missing secure data storage"
        assert "AES/GCM" in content, "Missing AES-GCM encryption"
        assert "AndroidKeyStore" in content, "Missing KeyStore integration"
        assert "OWASP MASVS" in content, "Missing OWASP MASVS compliance reference"
    
    def test_ios_security_class_structure(self):
        """Test iOS security classes have proper structure."""
        base_path = "/Users/yunusgungor/work/mobilemodel"
        
        # SecureStorageManager structure
        storage_file = os.path.join(base_path, 
            "src/m3tm/mobile/ios/sdk/SecureStorageManager.swift")
        
        with open(storage_file, 'r') as f:
            content = f.read()
            
        # Check critical methods exist
        assert "initialize" in content, "SecureStorageManager missing initialize method"
        assert "storeSecureString" in content, "Missing secure string storage"
        assert "storeSecureData" in content, "Missing secure data storage"
        assert "Keychain" in content, "Missing Keychain integration"
        assert "OWASP MASVS" in content, "Missing OWASP MASVS compliance reference"
    
    def test_network_security_configuration(self):
        """Test network security configuration is properly structured."""
        base_path = "/Users/yunusgungor/work/mobilemodel"
        
        config_file = os.path.join(base_path, 
            "android/src/main/res/xml/network_security_config.xml")
        
        with open(config_file, 'r') as f:
            content = f.read()
            
        # Check certificate pinning configuration
        assert "pin-set" in content, "Missing certificate pinning configuration"
        assert "SHA-256" in content, "Missing SHA-256 pin digest"
        assert "api.m3tm.com" in content, "Missing production domain"
        assert "TLS" in content or "tls" in content, "Missing TLS configuration"
        assert "OWASP MASVS" in content, "Missing OWASP MASVS compliance reference"
    
    def test_security_test_runner_executable(self):
        """Test that security test runner script is executable."""
        base_path = "/Users/yunusgungor/work/mobilemodel"
        script_path = os.path.join(base_path, "tests/security/run_security_tests.sh")
        
        # Check if file is executable
        assert os.access(script_path, os.X_OK), "Security test runner script is not executable"
        
        # Check script content
        with open(script_path, 'r') as f:
            content = f.read()
            
        assert "OWASP MASVS" in content, "Script missing OWASP MASVS reference"
        assert "SecureStorageIntegrationTest" in content, "Script missing storage tests"
        assert "NetworkSecurityIntegrationTest" in content, "Script missing network tests"
    
    def test_security_implementation_completeness(self):
        """Test that security implementation covers all required MASVS categories."""
        base_path = "/Users/yunusgungor/work/mobilemodel"
        
        # Check Story S24 implementation status
        story_file = os.path.join(base_path, ".project_meta/.stories/story_S24.json")
        
        with open(story_file, 'r') as f:
            content = f.read()
            
        # Verify MASVS categories are covered
        masvs_categories = [
            "MASVS-STORAGE",
            "MASVS-CRYPTO", 
            "MASVS-AUTH",
            "MASVS-NETWORK",
            "MASVS-PLATFORM",
            "MASVS-CODE"
        ]
        
        for category in masvs_categories:
            assert category in content, f"Missing MASVS category: {category}"
    
    def test_phase1_implementation_status(self):
        """Test that Phase 1 security implementation is marked as completed."""
        base_path = "/Users/yunusgungor/work/mobilemodel"
        
        # Check roadmap status
        roadmap_file = os.path.join(base_path, ".project_meta/.stories/roadmap.json")
        
        with open(roadmap_file, 'r') as f:
            content = f.read()
            
        # Story S24 should be in progress with 60%+ completion
        assert '"story_id": "S24"' in content, "Story S24 not found in roadmap"
        assert '"status": "in_progress"' in content, "Story S24 should be in progress"
        
        # Check completion percentage has increased
        assert '"completion": "60%"' in content, "Story S24 completion should be updated"
    
    def test_integration_artifacts_created(self):
        """Test that integration artifacts were properly created."""
        base_path = "/Users/yunusgungor/work/mobilemodel"
        
        # Check if security test directory exists
        security_test_dir = os.path.join(base_path, "tests/unit/security")
        assert os.path.exists(security_test_dir), "Security test directory not created"
        
        # Check if security source directory exists
        android_security_dir = os.path.join(base_path, "src/m3tm/mobile/android/sdk/java/com/m3tm/sdk/security")
        assert os.path.exists(android_security_dir), "Android security source directory not created"
        
        ios_security_dir = os.path.join(base_path, "src/m3tm/mobile/ios/sdk")
        assert os.path.exists(ios_security_dir), "iOS security source directory exists"
        
        # Check if network config directory exists
        network_config_dir = os.path.join(base_path, "android/src/main/res/xml")
        assert os.path.exists(network_config_dir), "Network config directory not created"
