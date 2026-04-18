package com.dawn.libimage;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import com.dawn.face_check.CheckFaceManage;
import com.dawn.face_check.CheckFaceManage2;
import com.dawn.face_check.FaceCheckResult;
import com.dawn.face_check.FaceChecker;

import java.io.InputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private ExecutorService executor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ImageView imageView1 = findViewById(R.id.imageView1);
        TextView tvResult1 = findViewById(R.id.tvResult1);
        ImageView imageView2 = findViewById(R.id.imageView2);
        TextView tvResult2 = findViewById(R.id.tvResult2);
        ImageView imageView3 = findViewById(R.id.imageView3);
        TextView tvResult3 = findViewById(R.id.tvResult3);

        executor = Executors.newSingleThreadExecutor();
        executor.execute(() -> {
            try {
                // 从 assets 读取测试图片
                Bitmap bitmap;
                try (InputStream is = getAssets().open("test_face.jpg")) {
                    bitmap = BitmapFactory.decodeStream(is);
                }
                if (bitmap == null) {
                    Log.e(TAG, "测试图片加载失败");
                    return;
                }

                // ============ 1. FaceChecker - 人脸关键点检测 ============
                try (FaceChecker checker = FaceChecker.create(this)) {
                    FaceCheckResult result = checker.detectAndDraw(bitmap);
                    String info1 = String.format("检测到人脸: %s\n关键点数量: %d\n左耳可见: %s\n右耳可见: %s",
                            result.isFaceDetected(), result.getLandmarkCount(),
                            result.isLeftEarVisible(), result.isRightEarVisible());
                    Log.i(TAG, info1);
                    if (!isFinishing()) {
                        runOnUiThread(() -> {
                            imageView1.setImageBitmap(result.getAnnotatedBitmap());
                            tvResult1.setText(info1);
                        });
                    }
                }

                // ============ 2. CheckFaceManage - 眼嘴鼻检测 ============
                try {
                    CheckFaceManage manage = new CheckFaceManage();
                    CheckFaceManage.FaceStateResult faceState = manage.renderFacePreview(this, bitmap);
                    if (faceState != null && !isFinishing()) {
                        runOnUiThread(() -> {
                            if (faceState.getAnnotatedBitmap() != null) {
                                imageView2.setImageBitmap(faceState.getAnnotatedBitmap());
                            }
                            tvResult2.setText(faceState.getDescription());
                        });
                    } else if (!isFinishing()) {
                        runOnUiThread(() -> tvResult2.setText("眼嘴鼻检测失败，未检测到人脸"));
                    }
                } catch (Exception e) {
                    Log.e(TAG, "CheckFaceManage 失败: " + e.getMessage(), e);
                    if (!isFinishing()) {
                        runOnUiThread(() -> tvResult2.setText("CheckFaceManage 异常: " + e.getMessage()));
                    }
                }

                // ============ 3. CheckFaceManage2 - 人脸语义分割 ============
                try {
                    CheckFaceManage2 manage2 = new CheckFaceManage2();
                    CheckFaceManage2.SegResult segResult = manage2.analyzeFromAssets(this, bitmap);
                    String overlayPath = segResult.getOverlayPath();
                    Bitmap overlayBmp = overlayPath != null ? BitmapFactory.decodeFile(overlayPath) : null;
                    Log.i(TAG, "分割结果:\n" + segResult.getText());
                    if (!isFinishing()) {
                        runOnUiThread(() -> {
                            tvResult3.setText(segResult.getText());
                            if (overlayBmp != null) {
                                imageView3.setImageBitmap(overlayBmp);
                            }
                        });
                    }
                } catch (Exception e) {
                    Log.e(TAG, "CheckFaceManage2 失败: " + e.getMessage(), e);
                    if (!isFinishing()) {
                        runOnUiThread(() -> tvResult3.setText("CheckFaceManage2 异常: " + e.getMessage()));
                    }
                }

            } catch (Exception e) {
                Log.e(TAG, "人脸检测失败: " + e.getMessage(), e);
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (executor != null) {
            executor.shutdownNow();
        }
    }
}