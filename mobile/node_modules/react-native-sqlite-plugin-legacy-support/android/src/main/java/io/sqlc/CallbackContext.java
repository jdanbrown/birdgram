package io.sqlc;

import org.json.JSONArray;

public class CallbackContext {
    public void success() {}
    public void success(String value) {}
    public void success(JSONArray value) {}
    public void error(String message) {}
}
