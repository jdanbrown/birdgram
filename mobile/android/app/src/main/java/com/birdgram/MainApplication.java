package com.birdgram;

import android.app.Application;

import com.facebook.react.ReactApplication;
import io.sqlc.SQLiteSupportPackage;
import io.sentry.RNSentryPackage;
import com.learnium.RNDeviceInfo.RNDeviceInfo;
import com.lugg.ReactNativeConfig.ReactNativeConfigPackage;
import com.goodatlas.audiorecord.RNAudioRecordPackage;
import com.swmansion.gesturehandler.react.RNGestureHandlerPackage;
import com.RNFetchBlob.RNFetchBlobPackage;
import com.zmxv.RNSound.RNSoundPackage;
import com.oblador.vectoricons.VectorIconsPackage;
import com.dylanvann.fastimage.FastImageViewPackage;
import com.corbt.keepawake.KCKeepAwakePackage;
import com.github.chadsmith.MicrophoneStream.MicrophoneStreamPackage;
import com.facebook.react.ReactNativeHost;
import com.facebook.react.ReactPackage;
import com.facebook.react.shell.MainReactPackage;
import com.facebook.soloader.SoLoader;

import java.util.Arrays;
import java.util.List;

public class MainApplication extends Application implements ReactApplication {

  private final ReactNativeHost mReactNativeHost = new ReactNativeHost(this) {
    @Override
    public boolean getUseDeveloperSupport() {
      return BuildConfig.DEBUG;
    }

    @Override
    protected List<ReactPackage> getPackages() {
      return Arrays.<ReactPackage>asList(
          new MainReactPackage(),
            new SQLiteSupportPackage(),
            new RNSentryPackage(),
            new RNDeviceInfo(),
            new ReactNativeConfigPackage(),
            new RNAudioRecordPackage(),
            new RNGestureHandlerPackage(),
            new RNFetchBlobPackage(),
            new RNSoundPackage(),
            new VectorIconsPackage(),
            new FastImageViewPackage(),
            new KCKeepAwakePackage(),
            new MicrophoneStreamPackage()
      );
    }

    @Override
    protected String getJSMainModuleName() {
      return "index";
    }
  };

  @Override
  public ReactNativeHost getReactNativeHost() {
    return mReactNativeHost;
  }

  @Override
  public void onCreate() {
    super.onCreate();
    SoLoader.init(this, /* native exopackage */ false);
  }
}
