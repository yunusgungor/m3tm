# Quick Start Guide

Get up and running with M³TM in under 5 minutes! This guide will walk you through installing the SDK and building your first AI-powered app.

## Prerequisites

Before you begin, ensure you have:

- [x] A development environment set up (Android Studio, Xcode, or Python)
- [x] Basic knowledge of your chosen platform
- [x] A device or simulator for testing

## Step 1: Installation

Choose your platform and install the M³TM SDK:

=== "Android"

    ### Gradle Setup
    
    Add the M³TM repository to your project's `build.gradle`:
    
    ```gradle
    allprojects {
        repositories {
            google()
            mavenCentral()
            maven { url 'https://repo.m3tm.dev/releases' }
        }
    }
    ```
    
    Add the dependency to your app's `build.gradle`:
    
    ```gradle
    dependencies {
        implementation 'com.m3tm:android-sdk:2.3.0'
    }
    ```
    
    ### Permissions
    
    Add required permissions to your `AndroidManifest.xml`:
    
    ```xml
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.CAMERA" />
    ```

=== "iOS"

    ### Package Manager
    
    Add M³TM to your project using Swift Package Manager:
    
    1. In Xcode, go to File → Add Package Dependencies
    2. Enter the repository URL: `https://github.com/m3tm/ios-sdk`
    3. Select version `2.3.0` or later
    4. Add to your target
    
    ### Alternative: CocoaPods
    
    Add to your `Podfile`:
    
    ```ruby
    pod 'M3TM', '~> 2.3.0'
    ```
    
    Then run:
    
    ```bash
    pod install
    ```
    
    ### Info.plist
    
    Add required permissions:
    
    ```xml
    <key>NSCameraUsageDescription</key>
    <string>This app uses the camera to analyze images</string>
    <key>NSPhotoLibraryUsageDescription</key>
    <string>This app accesses photos for AI processing</string>
    ```

=== "Python"

    ### pip Installation
    
    ```bash
    pip install m3tm
    ```
    
    ### Development Installation
    
    For development with the latest features:
    
    ```bash
    pip install git+https://github.com/m3tm/mobilemodel.git
    ```
    
    ### Virtual Environment (Recommended)
    
    ```bash
    python -m venv m3tm-env
    source m3tm-env/bin/activate  # On Windows: m3tm-env\Scripts\activate
    pip install m3tm
    ```

## Step 2: Initialize Your First Model

Now let's create and initialize your first M³TM model:

=== "Android"

    ```kotlin
    import com.m3tm.android.*
    import com.m3tm.android.config.*
    
    class MainActivity : AppCompatActivity() {
        private lateinit var model: M3TM
        
        override fun onCreate(savedInstanceState: Bundle?) {
            super.onCreate(savedInstanceState)
            setContentView(R.layout.activity_main)
            
            // Initialize M³TM model
            initializeModel()
        }
        
        private fun initializeModel() {
            // Create model configuration
            val config = ModelConfig.builder()
                .setModelSize(ModelSize.COMPACT)  // Optimized for mobile
                .setPrivacyMode(PrivacyMode.STRICT)  // Maximum privacy
                .enableCaching(true)  // Enable result caching
                .build()
            
            // Initialize the model
            model = M3TM.builder()
                .setConfig(config)
                .setContext(this)
                .build()
            
            // Optional: Warm up the model
            model.warmUp()
        }
    }
    ```

=== "iOS"

    ```swift
    import M3TM
    import UIKit
    
    class ViewController: UIViewController {
        private var model: M3TM?
        
        override func viewDidLoad() {
            super.viewDidLoad()
            initializeModel()
        }
        
        private func initializeModel() {
            // Create model configuration
            let config = ModelConfig(
                modelSize: .compact,  // Optimized for mobile
                privacyMode: .strict,  // Maximum privacy
                enableCaching: true   // Enable result caching
            )
            
            do {
                // Initialize the model
                model = try M3TM(config: config)
                
                // Optional: Warm up the model
                try model?.warmUp()
                
                print("M³TM model initialized successfully")
            } catch {
                print("Failed to initialize M³TM: \\(error)")
            }
        }
    }
    ```

=== "Python"

    ```python
    import m3tm
    from m3tm.config import ModelConfig, ModelSize, PrivacyMode
    
    # Create model configuration
    config = ModelConfig(
        model_size=ModelSize.COMPACT,  # Optimized for mobile
        privacy_mode=PrivacyMode.STRICT,  # Maximum privacy
        enable_caching=True  # Enable result caching
    )
    
    # Initialize the model
    model = m3tm.M3TM(config=config)
    
    # Optional: Warm up the model
    model.warm_up()
    
    print("M³TM model initialized successfully")
    ```

## Step 3: Your First Inference

Let's run your first AI inference with both text and image inputs:

=== "Android"

    ```kotlin
    private fun runFirstInference() {
        // Prepare inputs
        val textInput = "A beautiful sunset over the mountains"
        val imageInput = BitmapFactory.decodeResource(resources, R.drawable.sunset_image)
        
        // Create inference request
        val request = InferenceRequest.builder()
            .setText(textInput)
            .setImage(imageInput)
            .setTask(Task.SIMILARITY_SEARCH)  // What we want to do
            .build()
        
        // Run inference asynchronously
        model.inferAsync(request) { result ->
            when (result) {
                is InferenceResult.Success -> {
                    handleResults(result.data)
                }
                is InferenceResult.Error -> {
                    Log.e("M3TM", "Inference failed: ${result.message}")
                }
            }
        }
    }
    
    private fun handleResults(data: InferenceData) {
        // Process the results
        val similarity = data.getSimilarityScore()
        val textEmbedding = data.getTextEmbedding()
        val imageEmbedding = data.getImageEmbedding()
        
        println("Similarity: $similarity")
        println("Text embedding size: ${textEmbedding.size}")
        println("Image embedding size: ${imageEmbedding.size}")
    }
    ```

=== "iOS"

    ```swift
    private func runFirstInference() {
        guard let model = model else { return }
        
        // Prepare inputs
        let textInput = "A beautiful sunset over the mountains"
        let imageInput = UIImage(named: "sunset_image")!
        
        // Create inference request
        let request = InferenceRequest(
            text: textInput,
            image: imageInput,
            task: .similaritySearch  // What we want to do
        )
        
        // Run inference asynchronously
        Task {
            do {
                let result = try await model.infer(request)
                await handleResults(result)
            } catch {
                print("Inference failed: \\(error)")
            }
        }
    }
    
    @MainActor
    private func handleResults(_ data: InferenceData) {
        // Process the results
        let similarity = data.similarityScore
        let textEmbedding = data.textEmbedding
        let imageEmbedding = data.imageEmbedding
        
        print("Similarity: \\(similarity)")
        print("Text embedding size: \\(textEmbedding.count)")
        print("Image embedding size: \\(imageEmbedding.count)")
    }
    ```

=== "Python"

    ```python
    import numpy as np
    from PIL import Image
    
    def run_first_inference():
        # Prepare inputs
        text_input = "A beautiful sunset over the mountains"
        image_input = Image.open("sunset_image.jpg")
        
        # Run inference
        result = model.infer(
            text=text_input,
            image=image_input,
            task="similarity_search"  # What we want to do
        )
        
        # Process the results
        similarity = result.similarity_score
        text_embedding = result.text_embedding
        image_embedding = result.image_embedding
        
        print(f"Similarity: {similarity}")
        print(f"Text embedding shape: {text_embedding.shape}")
        print(f"Image embedding shape: {image_embedding.shape}")
        
        return result
    
    # Run the inference
    inference_result = run_first_inference()
    ```

## Step 4: Understanding the Results

The inference returns several types of data:

### Similarity Score
A value between 0.0 and 1.0 indicating how well the text and image match:
- **0.0-0.3**: Low similarity
- **0.3-0.7**: Moderate similarity  
- **0.7-1.0**: High similarity

### Embeddings
High-dimensional vectors representing the semantic content:
- **Text Embedding**: 384-dimensional vector
- **Image Embedding**: 384-dimensional vector
- **Fused Embedding**: Combined representation

### Usage Example

```python
# Using similarity for search
if result.similarity_score > 0.7:
    print("✅ Strong match found!")
elif result.similarity_score > 0.4:
    print("⚠️ Partial match")
else:
    print("❌ No significant match")

# Using embeddings for clustering
embeddings_db.store(result.fused_embedding, metadata={"text": text_input})
```

## Step 5: Next Steps

Congratulations! You've successfully run your first M³TM inference. Here's what to explore next:

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Build Your First App**

    ---

    Follow our comprehensive tutorial to build a complete AI-powered application.

    [:octicons-arrow-right-24: Start Tutorial](../tutorials/basic-integration.md)

-   :material-tune:{ .lg .middle } **Optimize Performance**

    ---

    Learn how to optimize M³TM for your specific use case and device constraints.

    [:octicons-arrow-right-24: Optimization Guide](../tutorials/optimization.md)

-   :material-code-braces:{ .lg .middle } **Explore Examples**

    ---

    Browse our collection of sample applications and integration patterns.

    [:octicons-arrow-right-24: View Examples](../examples/sample-apps.md)

-   :material-book-open:{ .lg .middle } **API Reference**

    ---

    Dive deep into the complete API documentation and advanced features.

    [:octicons-arrow-right-24: API Docs](../api/core/index.md)

</div>

## Troubleshooting

Having issues? Here are common solutions:

??? question "Model initialization fails"
    
    **Possible causes:**
    - Insufficient device memory
    - Incompatible device architecture
    - Network connectivity issues (first run)
    
    **Solutions:**
    - Use `ModelSize.TINY` for low-memory devices
    - Check device compatibility requirements
    - Ensure internet connection for initial model download

??? question "Inference is slow"
    
    **Possible causes:**
    - Large input images
    - CPU-only execution
    - Background processing
    
    **Solutions:**
    - Resize images to 224x224 or smaller
    - Enable GPU acceleration in config
    - Run inference on background thread

??? question "Results seem inaccurate"
    
    **Possible causes:**
    - Poor quality input images
    - Mismatched text-image pairs
    - Wrong task configuration
    
    **Solutions:**
    - Use high-quality, well-lit images
    - Ensure text describes the image content
    - Check task type matches your use case

## Community Support

Need help? Our community is here for you:

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/m3tm/mobilemodel/issues)
- **Discussions**: [Ask questions and share projects](https://github.com/m3tm/mobilemodel/discussions)
- **Discord**: [Real-time community chat](https://discord.gg/m3tm)
- **Documentation**: [Browse the complete docs](../index.md)

!!! tip "Pro Tip"
    
    Join our Discord server for real-time help from the community and M³TM team members!

---

**Time to Complete**: ~5 minutes  
**Next**: [Build Your First App →](../tutorials/basic-integration.md)
