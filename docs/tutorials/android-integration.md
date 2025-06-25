# Android Integration Tutorial

This step-by-step tutorial guides you through integrating M³TM models into an Android application.

## Prerequisites

- Android Studio 4.0 or later
- Android SDK API level 21 or higher
- Basic knowledge of Android development (Java/Kotlin)
- A trained M³TM model (we'll use `m3tm-small` for this tutorial)

## Tutorial Overview

We'll build a simple photo search app that:
1. Takes text queries and photos from users
2. Generates embeddings using M³TM
3. Finds similar images in a gallery
4. Displays results in real-time

## Step 1: Project Setup

### Create Android Project

1. Open Android Studio and create a new project
2. Choose "Empty Activity" template
3. Set minimum SDK to API 21
4. Choose Kotlin as the language

### Add Dependencies

Add these dependencies to your `app/build.gradle`:

```gradle
dependencies {
    implementation 'androidx.core:core-ktx:1.7.0'
    implementation 'androidx.appcompat:appcompat:1.4.1'
    implementation 'com.google.android.material:material:1.5.0'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.3'
    
    // ONNX Runtime for model inference
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.12.0'
    
    // Image processing
    implementation 'com.github.bumptech.glide:glide:4.13.0'
    
    // Permissions handling
    implementation 'com.karumi:dexter:6.2.3'
    
    // Async operations
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.6.0'
    
    // JSON processing
    implementation 'com.google.code.gson:gson:2.8.9'
    
    // Testing
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.3'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.4.0'
}
```

## Step 2: Prepare the M³TM Model

First, optimize your model for Android:

```python
# optimize_for_android.py
from m3tm.core import M3TMModel
from m3tm.mobile import MobileOptimizer, export_to_onnx

# Load and optimize model
model = M3TMModel.from_pretrained("m3tm-small")
optimizer = MobileOptimizer(
    target_platform="android",
    optimization_level="balanced"
)

optimized_model = optimizer.optimize(model)

# Export to ONNX
onnx_path = export_to_onnx(
    optimized_model,
    "m3tm_android.onnx",
    input_names=["text_tokens", "image_tensor"],
    output_names=["text_embedding", "image_embedding", "fused_embedding"]
)

print(f"Model exported to: {onnx_path}")
```

Place the generated `m3tm_android.onnx` file in your Android project's `assets` folder.

## Step 3: Model Wrapper Class

Create a Kotlin class to handle model inference:

```kotlin
// M3TMInference.kt
package com.example.m3tmdemo

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import java.util.*

class M3TMInference(private val context: Context) {
    private var ortSession: OrtSession? = null
    private var ortEnvironment: OrtEnvironment = OrtEnvironment.getEnvironment()
    
    companion object {
        private const val TAG = "M3TMInference"
        private const val MODEL_NAME = "m3tm_android.onnx"
        private const val IMAGE_SIZE = 224
        private const val MAX_TEXT_LENGTH = 77
    }
    
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            // Load model from assets
            val modelBytes = context.assets.open(MODEL_NAME).readBytes()
            ortSession = ortEnvironment.createSession(modelBytes)
            Log.d(TAG, "Model loaded successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model", e)
            false
        }
    }
    
    suspend fun generateEmbeddings(
        text: String,
        image: Bitmap?
    ): M3TMResult? = withContext(Dispatchers.IO) {
        val session = ortSession ?: return@withContext null
        
        try {
            val inputs = mutableMapOf<String, OnnxTensor>()
            
            // Process text input
            val textTokens = tokenizeText(text)
            val textTensor = OnnxTensor.createTensor(
                ortEnvironment,
                textTokens,
                longArrayOf(1, MAX_TEXT_LENGTH.toLong())
            )
            inputs["text_tokens"] = textTensor
            
            // Process image input
            if (image != null) {
                val imageArray = preprocessImage(image)
                val imageTensor = OnnxTensor.createTensor(
                    ortEnvironment,
                    imageArray,
                    longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
                )
                inputs["image_tensor"] = imageTensor
            }
            
            // Run inference
            val results = session.run(inputs)
            
            // Extract outputs
            val textEmbedding = results.get("text_embedding")?.value as? Array<FloatArray>
            val imageEmbedding = results.get("image_embedding")?.value as? Array<FloatArray>
            val fusedEmbedding = results.get("fused_embedding")?.value as? Array<FloatArray>
            
            // Clean up tensors
            inputs.values.forEach { it.close() }
            results.close()
            
            M3TMResult(
                textEmbedding = textEmbedding?.get(0),
                imageEmbedding = imageEmbedding?.get(0),
                fusedEmbedding = fusedEmbedding?.get(0)
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            null
        }
    }
    
    private fun tokenizeText(text: String): Array<IntArray> {
        // Simple tokenization - in production, use proper tokenizer
        val tokens = IntArray(MAX_TEXT_LENGTH) { 0 }
        val words = text.lowercase().split(" ")
        
        // Simple word-to-id mapping (in production, use proper vocabulary)
        val vocab = mapOf(
            "photo" to 1, "image" to 2, "picture" to 3,
            "cat" to 10, "dog" to 11, "bird" to 12,
            "beautiful" to 20, "amazing" to 21, "wonderful" to 22
        )
        
        words.take(MAX_TEXT_LENGTH).forEachIndexed { index, word ->
            tokens[index] = vocab[word] ?: 100 // Unknown token
        }
        
        return arrayOf(tokens)
    }
    
    private fun preprocessImage(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        // Resize image to model input size
        val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        
        // Convert to RGB array and normalize
        val imageArray = Array(1) { 
            Array(3) { 
                Array(IMAGE_SIZE) { 
                    FloatArray(IMAGE_SIZE) 
                } 
            } 
        }
        
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        resized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        
        for (i in 0 until IMAGE_SIZE) {
            for (j in 0 until IMAGE_SIZE) {
                val pixel = pixels[i * IMAGE_SIZE + j]
                
                // Extract RGB and normalize to [0, 1]
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f
                
                // Store in CHW format (Channel, Height, Width)
                imageArray[0][0][i][j] = r
                imageArray[0][1][i][j] = g
                imageArray[0][2][i][j] = b
            }
        }
        
        return imageArray
    }
    
    fun computeSimilarity(embedding1: FloatArray, embedding2: FloatArray): Float {
        if (embedding1.size != embedding2.size) return 0f
        
        var dotProduct = 0f
        var norm1 = 0f
        var norm2 = 0f
        
        for (i in embedding1.indices) {
            dotProduct += embedding1[i] * embedding2[i]
            norm1 += embedding1[i] * embedding1[i]
            norm2 += embedding2[i] * embedding2[i]
        }
        
        return if (norm1 == 0f || norm2 == 0f) 0f 
               else dotProduct / (kotlin.math.sqrt(norm1) * kotlin.math.sqrt(norm2))
    }
    
    fun cleanup() {
        ortSession?.close()
        ortEnvironment.close()
    }
}

data class M3TMResult(
    val textEmbedding: FloatArray?,
    val imageEmbedding: FloatArray?,
    val fusedEmbedding: FloatArray?
)
```

## Step 4: Main Activity Implementation

```kotlin
// MainActivity.kt
package com.example.m3tmdemo

import android.Manifest
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.karumi.dexter.Dexter
import com.karumi.dexter.MultiplePermissionsReport
import com.karumi.dexter.PermissionToken
import com.karumi.dexter.listener.PermissionRequest
import com.karumi.dexter.listener.multi.MultiplePermissionsListener
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    
    private lateinit var searchEditText: EditText
    private lateinit var selectImageButton: Button
    private lateinit var searchButton: Button
    private lateinit var selectedImageView: ImageView
    private lateinit var resultsRecyclerView: RecyclerView
    private lateinit var progressBar: ProgressBar
    
    private lateinit var m3tmInference: M3TMInference
    private lateinit var imageGallery: ImageGallery
    private lateinit var resultsAdapter: SearchResultsAdapter
    
    private var selectedImageBitmap: Bitmap? = null
    
    companion object {
        private const val PICK_IMAGE_REQUEST = 1001
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initializeViews()
        setupRecyclerView()
        requestPermissions()
        initializeM3TM()
        loadSampleImages()
    }
    
    private fun initializeViews() {
        searchEditText = findViewById(R.id.searchEditText)
        selectImageButton = findViewById(R.id.selectImageButton)
        searchButton = findViewById(R.id.searchButton)
        selectedImageView = findViewById(R.id.selectedImageView)
        resultsRecyclerView = findViewById(R.id.resultsRecyclerView)
        progressBar = findViewById(R.id.progressBar)
        
        selectImageButton.setOnClickListener { selectImage() }
        searchButton.setOnClickListener { performSearch() }
    }
    
    private fun setupRecyclerView() {
        resultsAdapter = SearchResultsAdapter { imageItem ->
            // Handle image selection from results
            Toast.makeText(this, "Selected: ${imageItem.name}", Toast.LENGTH_SHORT).show()
        }
        
        resultsRecyclerView.apply {
            layoutManager = GridLayoutManager(this@MainActivity, 2)
            adapter = resultsAdapter
        }
    }
    
    private fun requestPermissions() {
        Dexter.withContext(this)
            .withPermissions(
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.CAMERA
            )
            .withListener(object : MultiplePermissionsListener {
                override fun onPermissionsChecked(report: MultiplePermissionsReport?) {
                    if (report?.areAllPermissionsGranted() == true) {
                        Toast.makeText(this@MainActivity, "Permissions granted", Toast.LENGTH_SHORT).show()
                    }
                }
                
                override fun onPermissionRationaleShouldBeShown(
                    permissions: MutableList<PermissionRequest>?,
                    token: PermissionToken?
                ) {
                    token?.continuePermissionRequest()
                }
            })
            .check()
    }
    
    private fun initializeM3TM() {
        m3tmInference = M3TMInference(this)
        
        lifecycleScope.launch {
            progressBar.visibility = ProgressBar.VISIBLE
            val success = m3tmInference.initialize()
            progressBar.visibility = ProgressBar.GONE
            
            if (success) {
                Toast.makeText(this@MainActivity, "M³TM model loaded", Toast.LENGTH_SHORT).show()
                searchButton.isEnabled = true
            } else {
                Toast.makeText(this@MainActivity, "Failed to load model", Toast.LENGTH_LONG).show()
            }
        }
    }
    
    private fun loadSampleImages() {
        imageGallery = ImageGallery(this)
        lifecycleScope.launch {
            imageGallery.loadSampleImages()
        }
    }
    
    private fun selectImage() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(intent, PICK_IMAGE_REQUEST)
    }
    
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK) {
            data?.data?.let { uri ->
                selectedImageBitmap = getBitmapFromUri(uri)
                selectedImageView.setImageBitmap(selectedImageBitmap)
                selectedImageView.visibility = ImageView.VISIBLE
            }
        }
    }
    
    private fun getBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            val inputStream = contentResolver.openInputStream(uri)
            BitmapFactory.decodeStream(inputStream)
        } catch (e: Exception) {
            null
        }
    }
    
    private fun performSearch() {
        val query = searchEditText.text.toString().trim()
        
        if (query.isEmpty() && selectedImageBitmap == null) {
            Toast.makeText(this, "Please enter text or select an image", Toast.LENGTH_SHORT).show()
            return
        }
        
        lifecycleScope.launch {
            progressBar.visibility = ProgressBar.VISIBLE
            searchButton.isEnabled = false
            
            try {
                // Generate embeddings for query
                val queryResult = m3tmInference.generateEmbeddings(query, selectedImageBitmap)
                
                if (queryResult != null) {
                    // Search in image gallery
                    val results = imageGallery.searchSimilar(queryResult, m3tmInference)
                    
                    // Update UI with results
                    resultsAdapter.updateResults(results)
                    
                    Toast.makeText(this@MainActivity, 
                        "Found ${results.size} similar images", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this@MainActivity, 
                        "Failed to generate embeddings", Toast.LENGTH_SHORT).show()
                }
                
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, 
                    "Search failed: ${e.message}", Toast.LENGTH_LONG).show()
            } finally {
                progressBar.visibility = ProgressBar.GONE
                searchButton.isEnabled = true
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        m3tmInference.cleanup()
    }
}
```

## Step 5: Image Gallery and Search Logic

```kotlin
// ImageGallery.kt
package com.example.m3tmdemo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class ImageGallery(private val context: Context) {
    private val images = mutableListOf<ImageItem>()
    private val embeddings = mutableMapOf<String, FloatArray>()
    
    suspend fun loadSampleImages() = withContext(Dispatchers.IO) {
        // Load sample images from assets
        val assetManager = context.assets
        val imageFiles = assetManager.list("sample_images") ?: emptyArray()
        
        imageFiles.forEach { fileName ->
            try {
                val bitmap = assetManager.open("sample_images/$fileName").use { inputStream ->
                    BitmapFactory.decodeStream(inputStream)
                }
                
                val imageItem = ImageItem(
                    id = fileName,
                    name = fileName.substringBeforeLast('.'),
                    bitmap = bitmap
                )
                
                images.add(imageItem)
            } catch (e: Exception) {
                // Handle error loading image
            }
        }
    }
    
    suspend fun precomputeEmbeddings(m3tmInference: M3TMInference) = withContext(Dispatchers.IO) {
        images.forEach { imageItem ->
            val result = m3tmInference.generateEmbeddings("", imageItem.bitmap)
            result?.imageEmbedding?.let { embedding ->
                embeddings[imageItem.id] = embedding
            }
        }
    }
    
    suspend fun searchSimilar(
        queryResult: M3TMResult,
        m3tmInference: M3TMInference,
        topK: Int = 10
    ): List<SearchResult> = withContext(Dispatchers.IO) {
        
        val queryEmbedding = queryResult.fusedEmbedding 
            ?: queryResult.imageEmbedding 
            ?: queryResult.textEmbedding
            ?: return@withContext emptyList()
        
        val similarities = mutableListOf<SearchResult>()
        
        images.forEach { imageItem ->
            // Get or compute embedding for this image
            val imageEmbedding = embeddings[imageItem.id] ?: run {
                val result = m3tmInference.generateEmbeddings("", imageItem.bitmap)
                result?.imageEmbedding?.also { 
                    embeddings[imageItem.id] = it 
                }
            }
            
            imageEmbedding?.let { embedding ->
                val similarity = m3tmInference.computeSimilarity(queryEmbedding, embedding)
                similarities.add(SearchResult(imageItem, similarity))
            }
        }
        
        // Sort by similarity and return top K
        similarities.sortedByDescending { it.similarity }.take(topK)
    }
}

data class ImageItem(
    val id: String,
    val name: String,
    val bitmap: Bitmap
)

data class SearchResult(
    val imageItem: ImageItem,
    val similarity: Float
)
```

## Step 6: RecyclerView Adapter

```kotlin
// SearchResultsAdapter.kt
package com.example.m3tmdemo

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class SearchResultsAdapter(
    private val onItemClick: (ImageItem) -> Unit
) : RecyclerView.Adapter<SearchResultsAdapter.ViewHolder>() {
    
    private var results: List<SearchResult> = emptyList()
    
    fun updateResults(newResults: List<SearchResult>) {
        results = newResults
        notifyDataSetChanged()
    }
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_search_result, parent, false)
        return ViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(results[position])
    }
    
    override fun getItemCount(): Int = results.size
    
    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val imageView: ImageView = itemView.findViewById(R.id.resultImageView)
        private val nameTextView: TextView = itemView.findViewById(R.id.resultNameTextView)
        private val similarityTextView: TextView = itemView.findViewById(R.id.resultSimilarityTextView)
        
        fun bind(searchResult: SearchResult) {
            imageView.setImageBitmap(searchResult.imageItem.bitmap)
            nameTextView.text = searchResult.imageItem.name
            similarityTextView.text = String.format("%.2f", searchResult.similarity)
            
            itemView.setOnClickListener {
                onItemClick(searchResult.imageItem)
            }
        }
    }
}
```

## Step 7: Layout Files

Create the layout files:

### activity_main.xml
```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <EditText
        android:id="@+id/searchEditText"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:hint="Enter search query..."
        android:minHeight="48dp" />

    <Button
        android:id="@+id/selectImageButton"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Select Image"
        android:layout_marginTop="8dp" />

    <ImageView
        android:id="@+id/selectedImageView"
        android:layout_width="200dp"
        android:layout_height="200dp"
        android:layout_gravity="center"
        android:layout_marginTop="8dp"
        android:scaleType="centerCrop"
        android:visibility="gone"
        android:contentDescription="Selected image" />

    <Button
        android:id="@+id/searchButton"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Search"
        android:layout_marginTop="8dp"
        android:enabled="false" />

    <ProgressBar
        android:id="@+id/progressBar"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_gravity="center"
        android:layout_marginTop="16dp"
        android:visibility="gone" />

    <androidx.recyclerview.widget.RecyclerView
        android:id="@+id/resultsRecyclerView"
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1"
        android:layout_marginTop="16dp" />

</LinearLayout>
```

### item_search_result.xml
```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:orientation="vertical"
    android:padding="8dp"
    android:background="?android:attr/selectableItemBackground">

    <ImageView
        android:id="@+id/resultImageView"
        android:layout_width="match_parent"
        android:layout_height="120dp"
        android:scaleType="centerCrop"
        android:contentDescription="Search result image" />

    <TextView
        android:id="@+id/resultNameTextView"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginTop="4dp"
        android:textSize="14sp"
        android:textStyle="bold" />

    <TextView
        android:id="@+id/resultSimilarityTextView"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginTop="2dp"
        android:textSize="12sp"
        android:textColor="#666666" />

</LinearLayout>
```

## Step 8: Testing and Optimization

### Performance Testing

Add performance monitoring to your app:

```kotlin
// PerformanceMonitor.kt
class PerformanceMonitor {
    private val measurements = mutableListOf<Long>()
    
    inline fun <T> measureTime(operation: () -> T): T {
        val startTime = System.currentTimeMillis()
        val result = operation()
        val endTime = System.currentTimeMillis()
        measurements.add(endTime - startTime)
        return result
    }
    
    fun getAverageTime(): Double {
        return if (measurements.isEmpty()) 0.0 
               else measurements.average()
    }
    
    fun getStats(): String {
        return if (measurements.isEmpty()) "No measurements"
        else "Avg: ${getAverageTime():.1f}ms, Count: ${measurements.size}"
    }
}
```

Use it in your inference:

```kotlin
private val performanceMonitor = PerformanceMonitor()

private fun performSearch() {
    lifecycleScope.launch {
        val queryResult = performanceMonitor.measureTime {
            m3tmInference.generateEmbeddings(query, selectedImageBitmap)
        }
        
        Log.d("Performance", performanceMonitor.getStats())
    }
}
```

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check that the ONNX file is in the assets folder
   - Verify the model was exported correctly
   - Check device compatibility

2. **Out of Memory Errors**
   - Reduce image resolution
   - Process images in smaller batches
   - Enable model quantization

3. **Slow Inference**
   - Enable NNAPI acceleration
   - Use GPU acceleration if available
   - Optimize model with mobile-specific settings

### Performance Tips

1. **Precompute embeddings** for your image gallery
2. **Cache results** to avoid recomputation
3. **Use background threads** for inference
4. **Implement lazy loading** for large image sets

## Next Steps

- Add more sophisticated tokenization
- Implement real-time camera search
- Add cloud synchronization for embeddings
- Optimize for specific Android devices

This tutorial provides a complete foundation for integrating M³TM into Android applications. You can extend it with additional features like real-time search, cloud synchronization, and advanced UI components.

## Related Resources

- [iOS Integration Tutorial](./ios-integration.md)
- [Mobile Deployment Guide](../guides/mobile-deployment.md)
- [Performance Optimization](../guides/performance.md)
- [API Reference](../api/overview.md)
