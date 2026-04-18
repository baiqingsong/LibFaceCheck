package com.dawn.face_check;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult;

import java.io.Closeable;
import java.nio.ByteBuffer;
import java.util.List;

/**
 * 人脸综合检测器。
 * <p>
 * 传入 Bitmap，一次性返回人脸检测、眼睛睁闭、五官可见性、嘴巴张闭、姿态端正、耳朵可见性等结果。
 * 传入的 Bitmap 不会被回收，外部可继续使用。
 * <p>
 * 使用示例：
 * <pre>
 *   try (FaceDetector detector = FaceDetector.create(context)) {
 *       FaceDetectResult result = detector.detect(bitmap);
 *       if (result.isFaceDetected()) {
 *           boolean eyesOpen = result.isLeftEyeOpen() && result.isRightEyeOpen();
 *           boolean mouthClosed = result.isMouthClosed();
 *           boolean straight = result.isFaceStraight();
 *       }
 *   }
 * </pre>
 */
public class FaceDetector implements Closeable {

    private static final String TAG = "FaceDetector";

    // ===== FaceMesh 关键点索引 =====
    // 左眼
    private static final int LE_V1A = 159, LE_V1B = 145;
    private static final int LE_V2A = 158, LE_V2B = 153;
    private static final int LE_H1 = 33, LE_H2 = 133;
    // 右眼
    private static final int RE_V1A = 386, RE_V1B = 374;
    private static final int RE_V2A = 385, RE_V2B = 380;
    private static final int RE_H1 = 362, RE_H2 = 263;
    // 嘴巴
    private static final int M_V1 = 13, M_V2 = 14;
    private static final int M_H1 = 61, M_H2 = 291;
    // 鼻子
    private static final int N_TIP = 1, N_BASE = 2;
    // 耳朵
    private static final int FM_LEFT_EAR = 234;
    private static final int FM_RIGHT_EAR = 454;

    // ===== 阈值 =====
    private static final float EAR_THRESH = 0.18f;
    private static final float MAR_THRESH = 0.55f;
    private static final float EAR_PIXEL_THRESH = 20f;
    private static final float MAR_PIXEL_THRESH = 25f;
    private static final float POSTURE_TOLERANCE = 25f;
    private static final float EAR_VISIBLE_RATIO = 0.35f;
    private static final float EAR_VISIBLE_VERTICAL = 0.6f;
    /** 眼睛可见性：EAR 水平距离占人脸宽度的最低比例 */
    private static final float EYE_VISIBLE_WIDTH_RATIO = 0.08f;
    /** 鼻子可见性：鼻子长度占人脸高度的最低比例 */
    private static final float NOSE_VISIBLE_HEIGHT_RATIO = 0.03f;

    private final FaceLandmarker faceLandmarker;

    private FaceDetector(FaceLandmarker faceLandmarker) {
        this.faceLandmarker = faceLandmarker;
    }

    /**
     * 创建 FaceDetector 实例，从 assets 自动加载模型。
     *
     * @param context Android Context
     * @return FaceDetector 实例
     * @throws Exception 模型文件未找到或加载失败
     */
    public static FaceDetector create(Context context) throws Exception {
        AssetManager am = context.getAssets();
        String modelPath = FaceModelHelper.findModelPath(am);
        if (modelPath == null) {
            throw new IllegalStateException(
                    "未在 assets 中找到 FaceLandmarker 模型文件。" +
                    "请将模型文件放入 face_check 模块的 src/main/assets/models/ 目录。");
        }

        ByteBuffer modelBuffer = FaceModelHelper.readAssetToDirectBuffer(am, modelPath);
        Log.i(TAG, "已加载模型: " + modelPath);

        FaceLandmarker.FaceLandmarkerOptions options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetBuffer(modelBuffer).build())
                .setNumFaces(1)
                .setOutputFaceBlendshapes(true)
                .build();

        FaceLandmarker landmarker = FaceLandmarker.createFromOptions(context, options);
        return new FaceDetector(landmarker);
    }

    /**
     * 对 Bitmap 进行人脸综合检测。
     * <p>
     * 传入的 Bitmap 不会被回收，调用方可继续使用。
     *
     * @param bitmap 输入图片，不会被修改或回收
     * @return 检测结果，包含所有人脸状态数据
     */
    public FaceDetectResult detect(Bitmap bitmap) {
        if (bitmap == null || bitmap.isRecycled()) {
            return FaceDetectResult.noFace();
        }

        MPImage mpImage = new BitmapImageBuilder(bitmap).build();
        FaceLandmarkerResult result;
        try {
            result = faceLandmarker.detect(mpImage);
        } catch (Exception e) {
            Log.e(TAG, "检测失败: " + e.getMessage());
            return FaceDetectResult.noFace();
        }

        if (result == null || result.faceLandmarks().isEmpty()) {
            return FaceDetectResult.noFace();
        }

        List<NormalizedLandmark> lms = result.faceLandmarks().get(0);
        int w = bitmap.getWidth();
        int h = bitmap.getHeight();

        return analyzeface(lms, w, h);
    }

    @Override
    public void close() {
        if (faceLandmarker != null) {
            faceLandmarker.close();
        }
    }

    // ======================== 内部分析逻辑 ========================

    private FaceDetectResult analyzeface(List<NormalizedLandmark> lms, int w, int h) {
        // ---------- 计算人脸边界框 ----------
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = -Float.MAX_VALUE, maxY = -Float.MAX_VALUE;
        for (NormalizedLandmark lm : lms) {
            float x = lm.x() * w, y = lm.y() * h;
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
        }
        float faceW = Math.max(1f, maxX - minX);
        float faceH = Math.max(1f, maxY - minY);
        float centerX = (minX + maxX) * 0.5f;

        // ---------- 眼睛 EAR（Eye Aspect Ratio） ----------
        float leftV1 = dist(lms, LE_V1A, LE_V1B, w, h);
        float leftV2 = dist(lms, LE_V2A, LE_V2B, w, h);
        float leftHor = dist(lms, LE_H1, LE_H2, w, h);
        float rightV1 = dist(lms, RE_V1A, RE_V1B, w, h);
        float rightV2 = dist(lms, RE_V2A, RE_V2B, w, h);
        float rightHor = dist(lms, RE_H1, RE_H2, w, h);

        float leftEAR = (leftV1 + leftV2) / (2f * Math.max(1e-6f, leftHor));
        float rightEAR = (rightV1 + rightV2) / (2f * Math.max(1e-6f, rightHor));

        boolean leftEyeOpen = leftEAR > EAR_THRESH || (leftV1 > EAR_PIXEL_THRESH && leftV2 > EAR_PIXEL_THRESH);
        boolean rightEyeOpen = rightEAR > EAR_THRESH || (rightV1 > EAR_PIXEL_THRESH && rightV2 > EAR_PIXEL_THRESH);

        // ---------- 眼睛可见性 ----------
        boolean leftEyeVisible = leftHor > faceW * EYE_VISIBLE_WIDTH_RATIO;
        boolean rightEyeVisible = rightHor > faceW * EYE_VISIBLE_WIDTH_RATIO;

        // ---------- 嘴巴 MAR（Mouth Aspect Ratio） ----------
        float mouthVert = dist(lms, M_V1, M_V2, w, h);
        float mouthHor = dist(lms, M_H1, M_H2, w, h);
        float mar = mouthVert / Math.max(1e-6f, mouthHor);
        boolean mouthClosed = mar <= MAR_THRESH && mouthVert <= MAR_PIXEL_THRESH;

        // ---------- 鼻子可见性 ----------
        float noseLen = dist(lms, N_TIP, N_BASE, w, h);
        boolean noseVisible = noseLen > faceH * NOSE_VISIBLE_HEIGHT_RATIO;

        // ---------- 人脸端正（鼻子-眼睛对称性） ----------
        float leftEyeCenterX = (lms.get(LE_H1).x() + lms.get(LE_H2).x()) / 2f * w;
        float rightEyeCenterX = (lms.get(RE_H1).x() + lms.get(RE_H2).x()) / 2f * w;
        float noseX = lms.get(N_TIP).x() * w;
        float distLeft = Math.abs(noseX - leftEyeCenterX);
        float distRight = Math.abs(rightEyeCenterX - noseX);
        boolean faceStraight = Math.abs(distLeft - distRight) <= POSTURE_TOLERANCE;

        // ---------- 耳朵可见性 ----------
        float leftEarX = lms.get(FM_LEFT_EAR).x() * w;
        float leftEarY = lms.get(FM_LEFT_EAR).y() * h;
        float rightEarX = lms.get(FM_RIGHT_EAR).x() * w;
        float rightEarY = lms.get(FM_RIGHT_EAR).y() * h;
        float faceMidY = minY + faceH * 0.5f;

        boolean leftEarVisible = (centerX - leftEarX) > faceW * EAR_VISIBLE_RATIO
                && Math.abs(leftEarY - faceMidY) < faceH * EAR_VISIBLE_VERTICAL;
        boolean rightEarVisible = (rightEarX - centerX) > faceW * EAR_VISIBLE_RATIO
                && Math.abs(rightEarY - faceMidY) < faceH * EAR_VISIBLE_VERTICAL;

        return new FaceDetectResult(
                true,
                leftEyeOpen, rightEyeOpen,
                leftEyeVisible, rightEyeVisible,
                noseVisible,
                mouthClosed,
                faceStraight,
                leftEarVisible, rightEarVisible,
                leftEAR, rightEAR, mar
        );
    }

    private float dist(List<NormalizedLandmark> lms, int i, int j, int w, int h) {
        float dx = (lms.get(i).x() - lms.get(j).x()) * w;
        float dy = (lms.get(i).y() - lms.get(j).y()) * h;
        return (float) Math.hypot(dx, dy);
    }
}
