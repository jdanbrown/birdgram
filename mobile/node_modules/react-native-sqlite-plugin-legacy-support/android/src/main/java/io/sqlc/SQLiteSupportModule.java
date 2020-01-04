package io.sqlc;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Callback;

import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableNativeArray;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;


public class SQLiteSupportModule extends ReactContextBaseJavaModule {

    private final ReactApplicationContext reactContext;

    private final SQLitePlugin p;

    public SQLiteSupportModule(ReactApplicationContext reactContext) {
        super(reactContext);
        this.reactContext = reactContext;
        this.p = new SQLitePlugin(reactContext);
    }

    @Override
    public String getName() {
        return "SQLiteSupport";
    }

    @ReactMethod
    public void sampleMethod(String stringArgument, int numberArgument, Callback callback) {
        // TODO: Implement some real useful functionality
        callback.invoke("Received numberArgument: " + numberArgument + " stringArgument: " + stringArgument);
    }

    @ReactMethod
    public void echoStringValue(ReadableArray values, Callback callback, Callback ecbIgnored) {
        ReadableMap m1 = values.getMap(0);
        String v = m1.getString("value");

        try {
            JSONArray a1 = new JSONArray();
            JSONObject o1 = new JSONObject();
            o1.put("value", v);
            a1.put(o1);

            final Callback cb = callback;

            p.execute("echoStringValue", a1, new CallbackContext() {
                public void success(String value) {
                    cb.invoke(value);
                }
            });
        } catch(Exception e) { }
    }

    @ReactMethod
    public void open(ReadableArray values, Callback callback, Callback ecb) {
        ReadableMap m1 = values.getMap(0);
        String v = m1.getString("name");

        try {
            JSONArray a1 = new JSONArray();
            JSONObject o1 = new JSONObject();
            o1.put("name", v);
            a1.put(o1);

            final Callback cb = callback;

            p.execute("open", a1, new CallbackContext() {
                public void success() {
                    cb.invoke();
                }
            });
        } catch(Exception e) { }
    }

    @ReactMethod
    public void close(ReadableArray values, Callback callback, Callback ecb) {
        ReadableMap m1 = values.getMap(0);
        String v = m1.getString("path");

        try {
            JSONArray a1 = new JSONArray();
            JSONObject o1 = new JSONObject();
            o1.put("path", v);
            a1.put(o1);

            final Callback cb = callback;

            p.execute("close", a1, new CallbackContext() {
                public void success() {
                    cb.invoke();
                }
            });
        } catch(Exception e) { }
    }

    @ReactMethod
    public void delete(ReadableArray values, Callback callback, Callback errorcb) {
        ReadableMap m1 = values.getMap(0);
        String v = m1.getString("path");

        try {
            JSONArray a1 = new JSONArray();
            JSONObject o1 = new JSONObject();
            o1.put("path", v);
            a1.put(o1);

            final Callback cb = callback;
            final Callback ecb = errorcb;

            p.execute("delete", a1, new CallbackContext() {
                public void success() {
                    cb.invoke();
                }
                public void error(String message) {
                    ecb.invoke(message);
                }
            });
        } catch(Exception e) { }
    }

    @ReactMethod
    public void backgroundExecuteSqlBatch(ReadableArray values, Callback callback, Callback ecb) {
        try {
            JSONArray a = ReactNativeJson.convertArrayToJson(values);

            final Callback cb = callback;

            p.execute("backgroundExecuteSqlBatch", a, new CallbackContext() {
                public void success(JSONArray a2) {
                    try {
                        cb.invoke(ReactNativeJson.convertJsonToArray(a2));
                    } catch(Exception e) { }
                }
            });
        } catch(Exception e) { }
    }

}
