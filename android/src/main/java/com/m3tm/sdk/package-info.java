/**
 * M³TM v2.3 - Android SDK
 * 
 * <p>
 * Bu paket, M³TM v2.3 (Mobil Multi-Modal Modüler Transformer) modelinin Android uygulamalarında 
 * kullanılmasını sağlayan SDK'yı içerir. SDK, kullanıcı verisiyle modeli cihaz üzerinde eğitmek,
 * çıkarım işlemleri yapmak ve veriler üzerinde semantik arama yapmak için API'ler sağlar.
 * </p>
 * 
 * <p>
 * Temel bileşenler:
 * <ul>
 *   <li>{@link com.m3tm.sdk.M3TMModel} - Yüklenen modeli temsil eder ve çıkarım işlevleri sağlar</li>
 *   <li>{@link com.m3tm.sdk.M3TMModelManager} - Model yükleme ve yönetimi için işlevler sağlar</li>
 *   <li>{@link com.m3tm.sdk.M3TMTrainingManager} - Eğitim ve ince ayarlama için işlevler sağlar</li>
 *   <li>{@link com.m3tm.sdk.ModelInfo} - Model meta verilerini temsil eder</li>
 *   <li>{@link com.m3tm.sdk.TrainingCallback} - Eğitim ilerleme geri çağrıları için arayüz</li>
 * </ul>
 * </p>
 * 
 * <p>
 * Tipik kullanım:
 * <pre>
 * // Model yöneticisini başlat
 * M3TMModelManager modelManager = new M3TMModelManager();
 * 
 * // Modeli yükle
 * M3TMModel model = modelManager.loadModel("m3tm_base");
 * 
 * // Metin işleme
 * try {
 *     Map<String, Object> result = model.processText("Örnek metin");
 *     // Sonuçları işle
 * } catch (InferenceException e) {
 *     // Hatayı işle
 * }
 * 
 * // Kaynakları serbest bırak
 * model.close();
 * modelManager.close();
 * </pre>
 * </p>
 * 
 * @since 1.0
 */
package com.m3tm.sdk; 