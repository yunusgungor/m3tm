package com.m3tm.sdk.example;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.util.Log;

import com.m3tm.sdk.InferenceException;
import com.m3tm.sdk.M3TMModel;
import com.m3tm.sdk.M3TMModelManager;
import com.m3tm.sdk.M3TMTrainingManager;
import com.m3tm.sdk.ModelException;
import com.m3tm.sdk.ModelInfo;
import com.m3tm.sdk.TrainingCallback;
import com.m3tm.sdk.TrainingException;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * M3TM SDK'nın kullanımını gösteren örnek sınıf.
 * Bu sınıf gerçek bir uygulamada doğrudan kullanılmamalıdır, sadece referans amaçlıdır.
 */
public class M3TMExample {
    private static final String TAG = "M3TMExample";
    private static final String DEFAULT_MODEL_ID = "m3tm_base";
    
    private M3TMModelManager modelManager;
    private M3TMTrainingManager trainingManager;
    private M3TMModel model;
    
    /**
     * M3TM SDK'yı başlatır ve model yükler.
     *
     * @param modelCacheDir Model önbellek dizini
     * @return Başarılı olup olmadığı
     */
    public boolean initialize(String modelCacheDir) {
        try {
            // Model yöneticisini başlat
            modelManager = new M3TMModelManager(modelCacheDir);
            
            // Kullanılabilir modelleri listele
            List<ModelInfo> availableModels = modelManager.listAvailableModels();
            
            for (ModelInfo info : availableModels) {
                Log.d(TAG, "Mevcut model: " + info.getModelId() + ", versiyon: " + info.getVersion());
            }
            
            // Modeli yükle
            model = modelManager.loadModel(DEFAULT_MODEL_ID);
            Log.d(TAG, "Model yüklendi: " + model.getModelInfo().getModelId());
            
            // Eğitim yöneticisini başlat
            trainingManager = new M3TMTrainingManager(modelCacheDir);
            
            return true;
        } catch (ModelException e) {
            Log.e(TAG, "Model yüklenirken hata: " + e.getMessage(), e);
            return false;
        } catch (Exception e) {
            Log.e(TAG, "SDK başlatılırken hata: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Metni işler ve sonuçları döndürür.
     *
     * @param text İşlenecek metin
     * @return İşlem sonuçları
     */
    public Map<String, Object> processTextExample(String text) {
        try {
            if (model == null) {
                Log.e(TAG, "Model yüklenmedi!");
                return null;
            }
            
            // Metin işle
            Map<String, Object> result = model.processText(text);
            
            // Sonuçları logla
            Log.d(TAG, "Metin işlendi: " + text);
            for (Map.Entry<String, Object> entry : result.entrySet()) {
                Log.d(TAG, entry.getKey() + " = " + entry.getValue());
            }
            
            return result;
        } catch (InferenceException e) {
            Log.e(TAG, "Metin işlenirken hata: " + e.getMessage(), e);
            return null;
        }
    }
    
    /**
     * Görüntüyü işler ve sonuçları döndürür.
     *
     * @param imagePath İşlenecek görüntü dosyasının yolu
     * @return İşlem sonuçları
     */
    public Map<String, Object> processImageExample(String imagePath) {
        try {
            if (model == null) {
                Log.e(TAG, "Model yüklenmedi!");
                return null;
            }
            
            // Görüntüyü yükle
            Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
            if (bitmap == null) {
                Log.e(TAG, "Görüntü yüklenemedi: " + imagePath);
                return null;
            }
            
            // Görüntüyü işle
            Map<String, Object> result = model.processImage(bitmap);
            
            // Sonuçları logla
            Log.d(TAG, "Görüntü işlendi: " + imagePath);
            for (Map.Entry<String, Object> entry : result.entrySet()) {
                Log.d(TAG, entry.getKey() + " = " + entry.getValue());
            }
            
            return result;
        } catch (InferenceException e) {
            Log.e(TAG, "Görüntü işlenirken hata: " + e.getMessage(), e);
            return null;
        }
    }
    
    /**
     * Metin ve görüntüyü birlikte işler ve sonuçları döndürür.
     *
     * @param text      İşlenecek metin
     * @param imagePath İşlenecek görüntü dosyasının yolu
     * @return İşlem sonuçları
     */
    public Map<String, Object> processMultimodalExample(String text, String imagePath) {
        try {
            if (model == null) {
                Log.e(TAG, "Model yüklenmedi!");
                return null;
            }
            
            // Görüntüyü yükle
            Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
            if (bitmap == null) {
                Log.e(TAG, "Görüntü yüklenemedi: " + imagePath);
                return null;
            }
            
            // Çoklu-modalite işle
            Map<String, Object> result = model.processMultimodal(text, bitmap);
            
            // Sonuçları logla
            Log.d(TAG, "Çoklu-modalite işlendi - Metin: " + text + ", Görüntü: " + imagePath);
            for (Map.Entry<String, Object> entry : result.entrySet()) {
                Log.d(TAG, entry.getKey() + " = " + entry.getValue());
            }
            
            return result;
        } catch (InferenceException e) {
            Log.e(TAG, "Çoklu-modalite işlenirken hata: " + e.getMessage(), e);
            return null;
        }
    }
    
    /**
     * Model eğitimi örneği.
     *
     * @param textSamples Eğitim için metin örnekleri
     * @param labels      Her metin örneği için etiketler
     * @return Eğitim sonuçları
     */
    public Map<String, Object> trainingExample(List<String> textSamples, List<String> labels) {
        try {
            if (trainingManager == null) {
                Log.e(TAG, "Eğitim yöneticisi başlatılmadı!");
                return null;
            }
            
            if (textSamples.size() != labels.size()) {
                Log.e(TAG, "Metin örnekleri ve etiketler aynı sayıda olmalıdır!");
                return null;
            }
            
            // Eğitim yapılandırması
            Map<String, Object> trainingConfig = new HashMap<>();
            trainingConfig.put("learning_rate", 0.0001);
            trainingConfig.put("epochs", 3);
            trainingConfig.put("batch_size", 4);
            
            // Eğitim oturumu oluştur
            String sessionId = trainingManager.createTrainingSession(DEFAULT_MODEL_ID, trainingConfig);
            Log.d(TAG, "Eğitim oturumu oluşturuldu: " + sessionId);
            
            // Eğitim verileri
            Map<String, Object> trainingData = new HashMap<>();
            List<Map<String, Object>> samples = new ArrayList<>();
            
            // Eğitim örnekleri ekle
            for (int i = 0; i < textSamples.size(); i++) {
                Map<String, Object> sample = new HashMap<>();
                sample.put("text", textSamples.get(i));
                sample.put("label", labels.get(i));
                samples.add(sample);
            }
            
            trainingData.put("samples", samples);
            
            // Eğitim ilerleme callback'i
            TrainingCallback callback = new TrainingCallback() {
                @Override
                public void onBatchComplete(int batch, Map<String, Object> metrics) {
                    Log.d(TAG, "Batch " + batch + " tamamlandı");
                    for (Map.Entry<String, Object> entry : metrics.entrySet()) {
                        Log.d(TAG, entry.getKey() + " = " + entry.getValue());
                    }
                }
                
                @Override
                public void onEpochComplete(int epoch, Map<String, Object> metrics) {
                    Log.d(TAG, "Epoch " + epoch + " tamamlandı");
                    for (Map.Entry<String, Object> entry : metrics.entrySet()) {
                        Log.d(TAG, entry.getKey() + " = " + entry.getValue());
                    }
                }
                
                @Override
                public void onTrainingComplete(Map<String, Object> metrics) {
                    Log.d(TAG, "Eğitim tamamlandı");
                    for (Map.Entry<String, Object> entry : metrics.entrySet()) {
                        Log.d(TAG, entry.getKey() + " = " + entry.getValue());
                    }
                }
            };
            
            // Eğitimi başlat
            Map<String, Object> trainingResults = trainingManager.startTraining(sessionId, trainingData, callback);
            
            // Eğitilmiş modeli kaydet
            String customModelPath = Environment.getExternalStorageDirectory() + "/m3tm_models";
            File dir = new File(customModelPath);
            if (!dir.exists()) {
                dir.mkdirs();
            }
            
            Map<String, Object> saveResults = trainingManager.saveTrainedModel(sessionId, customModelPath, "m3tm_custom");
            
            // Eğitim oturumunu sil
            trainingManager.deleteTrainingSession(sessionId);
            
            return trainingResults;
        } catch (TrainingException e) {
            Log.e(TAG, "Eğitim sırasında hata: " + e.getMessage(), e);
            return null;
        }
    }
    
    /**
     * Kaynakları serbest bırakır.
     */
    public void close() {
        if (model != null) {
            model.close();
            model = null;
        }
        
        if (modelManager != null) {
            modelManager.close();
            modelManager = null;
        }
        
        if (trainingManager != null) {
            trainingManager.close();
            trainingManager = null;
        }
        
        Log.d(TAG, "M3TM SDK kaynakları serbest bırakıldı");
    }
} 