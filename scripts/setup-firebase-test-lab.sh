#!/bin/bash

# M³TM Firebase Test Lab Setup Script
# Enhanced with Context7 best practices for mobile testing

set -e

echo "🔧 Setting up Firebase Test Lab for M³TM..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install Google Cloud SDK first."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Authenticate gcloud (if not already authenticated)
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "🔐 Authenticating with Google Cloud..."
    gcloud auth login
fi

# Set project (from environment or prompt)
if [[ -z "$GCP_PROJECT_ID" ]]; then
    echo "📝 Please enter your Google Cloud Project ID:"
    read -r GCP_PROJECT_ID
fi

gcloud config set project "$GCP_PROJECT_ID"

# Enable required APIs
echo "🚀 Enabling required Google Cloud APIs..."
gcloud services enable firebase.googleapis.com
gcloud services enable testing.googleapis.com
gcloud services enable toolresults.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com

# Create Firebase Test Lab results bucket
BUCKET_NAME="${GCP_PROJECT_ID}-firebase-test-results"
echo "📦 Creating Firebase Test Lab results bucket: $BUCKET_NAME"

if ! gsutil ls gs://"$BUCKET_NAME" 2>/dev/null; then
    gsutil mb gs://"$BUCKET_NAME"
    echo "✅ Created results bucket: gs://$BUCKET_NAME"
else
    echo "✅ Results bucket already exists: gs://$BUCKET_NAME"
fi

# Set bucket lifecycle for cost optimization
echo "⚙️ Setting bucket lifecycle for cost optimization..."
cat > lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://"$BUCKET_NAME"
rm lifecycle.json

# List available devices for testing
echo "📱 Available devices for testing:"
gcloud firebase test android models list --limit=10

# Create sample test configuration
cat > firebase-test-config.yml << EOF
# M³TM Firebase Test Lab Configuration
# Run with: gcloud firebase test android run --config=firebase-test-config.yml

test-setup:
  # APK paths (will be set by CI/CD)
  app: ./android/app/build/outputs/apk/debug/app-debug.apk
  test: ./android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk

# Device matrix for comprehensive testing
device:
  - model: Pixel2
    version: 28
    locale: en
    orientation: portrait
  - model: MediumPhone.arm
    version: 30
    locale: en
    orientation: portrait
  - model: Pixel6
    version: 31
    locale: en
    orientation: portrait
  - model: MediumTablet.arm
    version: 30
    locale: en
    orientation: landscape

# Test configuration
test-targets:
  - class com.m3tm.ExampleInstrumentedTest
  - package com.m3tm.tests

# Test execution settings
test-timeout: 15m
results-bucket: $BUCKET_NAME
results-dir: m3tm-test-results-\$(date +%Y%m%d_%H%M%S)

# Performance monitoring
performance-metrics: true
video-quality: high
record-video: true
EOF

echo "✅ Created firebase-test-config.yml"

# Create GitHub Actions secrets setup script
cat > setup-github-secrets.sh << 'EOF'
#!/bin/bash

# Script to help set up GitHub Actions secrets for Firebase Test Lab
echo "🔐 GitHub Actions Secrets Setup for Firebase Test Lab"
echo ""
echo "You need to set the following secrets in your GitHub repository:"
echo ""
echo "1. GCP_SA_KEY - Service Account Key JSON"
echo "   Create a service account with Firebase Test Lab Admin role:"
echo "   gcloud iam service-accounts create firebase-test-lab-sa --display-name=\"Firebase Test Lab Service Account\""
echo "   gcloud projects add-iam-policy-binding $GCP_PROJECT_ID --member=\"serviceAccount:firebase-test-lab-sa@$GCP_PROJECT_ID.iam.gserviceaccount.com\" --role=\"roles/firebase.admin\""
echo "   gcloud iam service-accounts keys create sa-key.json --iam-account=firebase-test-lab-sa@$GCP_PROJECT_ID.iam.gserviceaccount.com"
echo ""
echo "2. GCP_PROJECT_ID - Your Google Cloud Project ID"
echo "   Value: $GCP_PROJECT_ID"
echo ""
echo "3. FIREBASE_TEST_BUCKET - Test results bucket"
echo "   Value: $BUCKET_NAME"
echo ""
echo "Visit: https://github.com/YOUR_USERNAME/YOUR_REPO/settings/secrets/actions"
EOF

chmod +x setup-github-secrets.sh

echo ""
echo "🎉 Firebase Test Lab setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Run './setup-github-secrets.sh' to see GitHub secrets setup instructions"
echo "2. Test locally with: gcloud firebase test android run --config=firebase-test-config.yml"
echo "3. Push to trigger CI/CD pipeline with Firebase Test Lab integration"
echo ""
echo "📁 Files created:"
echo "  - firebase-test-config.yml (test configuration)"
echo "  - setup-github-secrets.sh (secrets setup guide)"
echo ""
echo "📦 Test results will be stored in: gs://$BUCKET_NAME"
