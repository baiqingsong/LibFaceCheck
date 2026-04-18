package com.dawn.face_check;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.util.Log;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult;

import java.io.Closeable;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 人脸关键点检测封装类。
 * <p>
 * 基于 MediaPipe FaceLandmarker，从 assets 加载模型，提供一行代码完成人脸检测的能力。
 * 支持检测人脸关键点、判断左右耳可见性、以及在图片上绘制标注。
 * <p>
 * 使用示例：
 * <pre>
 *   try (FaceChecker checker = FaceChecker.create(context)) {
 *       FaceCheckResult result = checker.detectAndDraw(bitmap);
 *       if (result.isFaceDetected()) {
 *           imageView.setImageBitmap(result.getAnnotatedBitmap());
 *       }
 *   }
 * </pre>
 */
public class FaceChecker implements Closeable {

    private static final String TAG = "FaceChecker";

    // FaceMesh 标准左右耳索引
    private static final int FM_LEFT_EAR = 234;
    private static final int FM_RIGHT_EAR = 454;

    private final FaceLandmarker faceLandmarker;

    private FaceChecker(FaceLandmarker faceLandmarker) {
        this.faceLandmarker = faceLandmarker;
    }

    /**
     * 创建 FaceChecker 实例。
     * 从库的 assets 中自动查找并加载 FaceLandmarker 模型。
     *
     * @param context Android Context
     * @return FaceChecker 实例
     * @throws Exception 如果模型文件未找到或加载失败
     */
    public static FaceChecker create(Context context) throws Exception {
        return create(context, 1);
    }

    /**
     * 创建 FaceChecker 实例，可指定最大人脸数。
     *
     * @param context  Android Context
     * @param numFaces 最大检测人脸数量
     * @return FaceChecker 实例
     * @throws Exception 如果模型文件未找到或加载失败
     */
    public static FaceChecker create(Context context, int numFaces) throws Exception {
        AssetManager am = context.getAssets();

        String chosenPath = FaceModelHelper.findModelPath(am);
        if (chosenPath == null) {
            throw new IllegalStateException(
                    "未在 assets 中找到 FaceLandmarker 模型文件。" +
                    "请将 face_landmarker.task 或 face_landmarker_v2_with_blendshapes.task " +
                    "放入 face_check 模块的 src/main/assets/models/ 目录。");
        }

        ByteBuffer modelBuffer = FaceModelHelper.readAssetToDirectBuffer(am, chosenPath);
        Log.i(TAG, "已加载模型: " + chosenPath);

        FaceLandmarker.FaceLandmarkerOptions options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetBuffer(modelBuffer).build())
                .setNumFaces(numFaces)
                .setOutputFaceBlendshapes(false)
                .build();

        FaceLandmarker landmarker = FaceLandmarker.createFromOptions(context, options);
        return new FaceChecker(landmarker);
    }

    /**
     * 检测图片中的人脸关键点。
     *
     * @param bitmap 输入图片
     * @return 检测结果（不含标注图）
     */
    public FaceCheckResult detect(Bitmap bitmap) {
        MPImage mpImage = new BitmapImageBuilder(bitmap).build();
        FaceLandmarkerResult result = faceLandmarker.detect(mpImage);
        return buildResult(bitmap, result);
    }

    /**
     * 检测人脸关键点并在图片上绘制标注。
     * 返回的 FaceCheckResult 中 getAnnotatedBitmap() 可获取标注后的图片。
     *
     * @param bitmap 输入图片
     * @return 检测结果（含标注图）
     */
    public FaceCheckResult detectAndDraw(Bitmap bitmap) {
        FaceCheckResult result = detect(bitmap);
        Bitmap annotated = drawLandmarks(bitmap, result);
        result.setAnnotatedBitmap(annotated);
        return result;
    }

    /**
     * 在图片上绘制人脸关键点和耳朵可见性标注。
     *
     * @param src    原图
     * @param result 检测结果
     * @return 标注后的新 Bitmap
     */
    public Bitmap drawLandmarks(Bitmap src, FaceCheckResult result) {
        Bitmap out = src.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(out);

        // 初始化画笔
        Paint pt = new Paint(Paint.ANTI_ALIAS_FLAG);
        pt.setStyle(Paint.Style.FILL);
        pt.setColor(Color.WHITE);
        float r = Math.max(3f, Math.min(out.getWidth(), out.getHeight()) * 0.006f);

        Paint text = new Paint(Paint.ANTI_ALIAS_FLAG);
        text.setColor(Color.YELLOW);
        text.setTextSize(Math.max(12f, Math.min(out.getWidth(), out.getHeight()) * 0.02f));
        text.setShadowLayer(2f, 1f, 1f, Color.BLACK);

        if (!result.isFaceDetected() || result.getLandmarks().isEmpty()) {
            canvas.drawText("未检测到人脸关键点", r * 2, r * 8, text);
            return out;
        }

        List<PointF> landmarks = result.getLandmarks();
        int n = landmarks.size();

        // 绘制所有关键点（白底红心双层圆点）
        for (PointF p : landmarks) {
            pt.setColor(Color.WHITE);
            canvas.drawCircle(p.x, p.y, r + 1f, pt);
            pt.setColor(Color.RED);
            canvas.drawCircle(p.x, p.y, r, pt);
        }

        // 计算人脸边界框
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = -Float.MAX_VALUE, maxY = -Float.MAX_VALUE;
        for (PointF p : landmarks) {
            if (p.x < minX) minX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.x > maxX) maxX = p.x;
            if (p.y > maxY) maxY = p.y;
        }
        float faceH = Math.max(1f, maxY - minY);

        // 绘制耳朵标注
        float oldTextSize = text.getTextSize();
        text.setTextSize(Math.max(10f, r * 0.9f));

        // 确定左右耳索引
        int leftEarIdx, rightEarIdx;
        if (n > FM_RIGHT_EAR) {
            leftEarIdx = FM_LEFT_EAR;
            rightEarIdx = FM_RIGHT_EAR;
        } else {
            leftEarIdx = 0;
            rightEarIdx = 0;
            float lMin = Float.MAX_VALUE, rMax = -Float.MAX_VALUE;
            for (int i = 0; i < n; i++) {
                if (landmarks.get(i).x < lMin) { lMin = landmarks.get(i).x; leftEarIdx = i; }
                if (landmarks.get(i).x > rMax) { rMax = landmarks.get(i).x; rightEarIdx = i; }
            }
        }

        pt.setColor(Color.BLUE);
        if (result.isLeftEarVisible()) {
            PointF lp = landmarks.get(leftEarIdx);
            canvas.drawCircle(lp.x, lp.y, r * 2.0f, pt);
            text.setColor(Color.WHITE);
            canvas.drawText("L", lp.x - r * 0.6f, lp.y - r * 1.2f, text);
        } else {
            text.setColor(Color.LTGRAY);
            float lx = Math.max(4f, minX - r * 3.5f);
            float ly = minY + faceH * 0.5f;
            canvas.drawText("L?", lx, ly, text);
        }

        if (result.isRightEarVisible()) {
            PointF rp = landmarks.get(rightEarIdx);
            canvas.drawCircle(rp.x, rp.y, r * 2.0f, pt);
            text.setColor(Color.WHITE);
            canvas.drawText("R", rp.x - r * 0.6f, rp.y - r * 1.2f, text);
        } else {
            text.setColor(Color.LTGRAY);
            float rx = Math.min(out.getWidth() - 4f, maxX + r * 2.5f);
            float ry = minY + faceH * 0.5f;
            canvas.drawText("R?", rx, ry, text);
        }

        text.setTextSize(oldTextSize);
        return out;
    }

    @Override
    public void close() {
        if (faceLandmarker != null) {
            faceLandmarker.close();
        }
    }

    // ======================== 内部方法 ========================

    /**
     * 从 MediaPipe 检测结果构建 FaceCheckResult
     */
    private FaceCheckResult buildResult(Bitmap bitmap, FaceLandmarkerResult result) {
        int w = bitmap.getWidth();
        int h = bitmap.getHeight();

        if (result == null || result.faceLandmarks().isEmpty()) {
            return new FaceCheckResult(false, Collections.emptyList(),
                    false, false, w, h);
        }

        List<NormalizedLandmark> lms = result.faceLandmarks().get(0);
        int n = lms.size();

        // 将归一化坐标转换为像素坐标
        List<PointF> landmarks = new ArrayList<>(n);
        float[] xs = new float[n];
        float[] ys = new float[n];
        for (int i = 0; i < n; i++) {
            NormalizedLandmark lm = lms.get(i);
            float x = lm.x() * w;
            float y = lm.y() * h;
            xs[i] = x;
            ys[i] = y;
            landmarks.add(new PointF(x, y));
        }

        // 判断左右耳可见性
        boolean leftEarVisible = false;
        boolean rightEarVisible = false;

        int leftEarIdx, rightEarIdx;
        if (n > FM_RIGHT_EAR) {
            leftEarIdx = FM_LEFT_EAR;
            rightEarIdx = FM_RIGHT_EAR;
        } else {
            leftEarIdx = 0;
            rightEarIdx = 0;
            float lMin = Float.MAX_VALUE, rMax = -Float.MAX_VALUE;
            for (int i = 0; i < n; i++) {
                if (xs[i] < lMin) { lMin = xs[i]; leftEarIdx = i; }
                if (xs[i] > rMax) { rMax = xs[i]; rightEarIdx = i; }
            }
        }

        // 计算人脸边界框和中心
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = -Float.MAX_VALUE, maxY = -Float.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            if (xs[i] < minX) minX = xs[i];
            if (ys[i] < minY) minY = ys[i];
            if (xs[i] > maxX) maxX = xs[i];
            if (ys[i] > maxY) maxY = ys[i];
        }
        float faceW = Math.max(1f, maxX - minX);
        float faceH = Math.max(1f, maxY - minY);
        float centerX = (minX + maxX) * 0.5f;

        // 启发式耳朵可见性判断
        if (leftEarIdx >= 0 && leftEarIdx < n) {
            float dx = centerX - xs[leftEarIdx];
            float dy = Math.abs(ys[leftEarIdx] - (minY + faceH * 0.5f));
            leftEarVisible = dx > faceW * 0.35f && dy < faceH * 0.6f;
        }
        if (rightEarIdx >= 0 && rightEarIdx < n) {
            float dx = xs[rightEarIdx] - centerX;
            float dy = Math.abs(ys[rightEarIdx] - (minY + faceH * 0.5f));
            rightEarVisible = dx > faceW * 0.35f && dy < faceH * 0.6f;
        }

        return new FaceCheckResult(true, landmarks, leftEarVisible, rightEarVisible, w, h);
    }

}
