package com.m3tm.sdk;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * Model meta verilerini temsil eden sınıf.
 */
public class ModelInfo {
    private final String modelId;
    private final String version;
    private final String name;
    private final String description;
    private final Map<String, Object> properties;

    /**
     * ModelInfo sınıfını başlatır.
     *
     * @param modelId     Model benzersiz tanımlayıcısı
     * @param version     Model versiyonu
     * @param name        Model ismi
     * @param description Model açıklaması
     * @param properties  Ek model özellikleri
     */
    public ModelInfo(String modelId, String version, String name, String description, Map<String, Object> properties) {
        this.modelId = modelId;
        this.version = version;
        this.name = name != null ? name : modelId;
        this.description = description != null ? description : "";
        this.properties = properties != null ? properties : new HashMap<>();
    }

    /**
     * ModelInfo sınıfını başlatır.
     *
     * @param modelId Model benzersiz tanımlayıcısı
     * @param version Model versiyonu
     */
    public ModelInfo(String modelId, String version) {
        this(modelId, version, null, null, null);
    }

    /**
     * Model ID'sini döndürür.
     *
     * @return Model ID'si
     */
    public String getModelId() {
        return modelId;
    }

    /**
     * Model versiyonunu döndürür.
     *
     * @return Model versiyonu
     */
    public String getVersion() {
        return version;
    }

    /**
     * Model ismini döndürür.
     *
     * @return Model ismi
     */
    public String getName() {
        return name;
    }

    /**
     * Model açıklamasını döndürür.
     *
     * @return Model açıklaması
     */
    public String getDescription() {
        return description;
    }

    /**
     * Model özelliklerini döndürür.
     *
     * @return Model özellikleri
     */
    public Map<String, Object> getProperties() {
        return new HashMap<>(properties);
    }

    /**
     * Model bilgisini JSON formatında döndürür.
     *
     * @return JSON formatında model bilgisi
     */
    public String toJson() {
        try {
            JSONObject jsonObject = new JSONObject();
            jsonObject.put("model_id", modelId);
            jsonObject.put("version", version);
            jsonObject.put("name", name);
            jsonObject.put("description", description);
            
            JSONObject propertiesJson = new JSONObject();
            for (Map.Entry<String, Object> entry : properties.entrySet()) {
                propertiesJson.put(entry.getKey(), entry.getValue());
            }
            jsonObject.put("properties", propertiesJson);
            
            return jsonObject.toString();
        } catch (JSONException e) {
            return "{}";
        }
    }

    /**
     * JSON formatından ModelInfo oluşturur.
     *
     * @param jsonStr JSON formatında model bilgisi
     * @return Oluşturulan ModelInfo nesnesi
     * @throws ModelException JSON dönüşüm hatası olduğunda
     */
    public static ModelInfo fromJson(String jsonStr) throws ModelException {
        try {
            JSONObject jsonObject = new JSONObject(jsonStr);
            String modelId = jsonObject.getString("model_id");
            String version = jsonObject.getString("version");
            String name = jsonObject.optString("name", modelId);
            String description = jsonObject.optString("description", "");
            
            Map<String, Object> properties = new HashMap<>();
            if (jsonObject.has("properties")) {
                JSONObject propertiesJson = jsonObject.getJSONObject("properties");
                Iterator<String> keys = propertiesJson.keys();
                while (keys.hasNext()) {
                    String key = keys.next();
                    properties.put(key, propertiesJson.get(key));
                }
            }
            
            return new ModelInfo(modelId, version, name, description, properties);
        } catch (JSONException e) {
            throw new ModelException("ModelInfo JSON'dan oluşturulamadı: " + e.getMessage(), e);
        }
    }

    @Override
    public String toString() {
        return "ModelInfo{" +
                "modelId='" + modelId + '\'' +
                ", version='" + version + '\'' +
                ", name='" + name + '\'' +
                ", description='" + description + '\'' +
                ", properties=" + properties +
                '}';
    }
} 