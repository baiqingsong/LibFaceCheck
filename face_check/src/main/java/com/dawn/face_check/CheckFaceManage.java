package com.dawn.face_check;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Locale;

/**
 * 人脸五官状态检测管理类。
 * <p>
 * 基于 MediaPipe FaceLandmarker 检测眼睛（睁开/闭合）、嘴巴（张开/闭合）、鼻子位置，
 * 以及坐姿是否端正（基于鼻子-眼睛对称性判断）。
 * <p>
 * 使用示例：
 * <pre>
 *   CheckFaceManage manage = new CheckFaceManage();
 *   Bitmap annotated = manage.renderFacePreview(context, "face.jpg");
 * </pre>
 */
public class CheckFaceManage {

    private static final String TAG = "CheckFaceManage";

    /**
     * 对 assets 中的图片进行人脸五官检测与标注。
     *
     * @param context  Android Context
     * @param fileName assets 中的图片文件名
     * @return 标注后的 Bitmap，检测失败时返回 null
     */
    public FaceStateResult renderFacePreview(Context context, String fileName) {
        AssetManager am = context.getAssets();
        String chosenPath = FaceModelHelper.findModelPath(am);
        if (chosenPath == null) { Log.e(TAG, "未找到模型，预览终止"); return null; }

        ByteBuffer modelBuffer;
        try { modelBuffer = FaceModelHelper.readAssetToDirectBuffer(am, chosenPath); } catch (Exception e) { Log.e(TAG, "读模型失败:" + e.getMessage()); return null; }

        FaceLandmarker.FaceLandmarkerOptions options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetBuffer(modelBuffer).build())
                .setNumFaces(1)
                .setOutputFaceBlendshapes(true)
                .build();

        try (FaceLandmarker faceLandmarker = FaceLandmarker.createFromOptions(context, options)) {
            Bitmap srcBmp;
            try (InputStream is = am.open(fileName)) { srcBmp = BitmapFactory.decodeStream(is); } catch (Exception e) { Log.e(TAG, "读取 " + fileName + " 失败:" + e.getMessage()); return null; }
            if (srcBmp == null) { Log.e(TAG, "图片为空"); return null; }

            MPImage mpImage = new BitmapImageBuilder(srcBmp).build();
            FaceLandmarkerResult result;
            try { result = faceLandmarker.detect(mpImage); } catch (Exception e) { Log.e(TAG, "检测失败:" + e.getMessage()); return null; }

            return drawEyeMouthKeypoints(srcBmp, result);
        } catch (Exception e) {
            Log.e(TAG, "预览失败:" + e.getMessage());
        }
        return null;
    }

    /**
     * 对 Bitmap 进行人脸五官检测与标注。
     *
     * @param context Android Context
     * @param srcBmp  输入图片
     * @return 标注后的 Bitmap，检测失败时返回 null
     */
    public FaceStateResult renderFacePreview(Context context, Bitmap srcBmp) {
        if (srcBmp == null) return null;
        AssetManager am = context.getAssets();
        String chosenPath = FaceModelHelper.findModelPath(am);
        if (chosenPath == null) { Log.e(TAG, "未找到模型，预览终止"); return null; }

        ByteBuffer modelBuffer;
        try { modelBuffer = FaceModelHelper.readAssetToDirectBuffer(am, chosenPath); } catch (Exception e) { Log.e(TAG, "读模型失败:" + e.getMessage()); return null; }

        FaceLandmarker.FaceLandmarkerOptions options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetBuffer(modelBuffer).build())
                .setNumFaces(1)
                .setOutputFaceBlendshapes(true)
                .build();

        try (FaceLandmarker faceLandmarker = FaceLandmarker.createFromOptions(context, options)) {
            MPImage mpImage = new BitmapImageBuilder(srcBmp).build();
            FaceLandmarkerResult result;
            try { result = faceLandmarker.detect(mpImage); } catch (Exception e) { Log.e(TAG, "检测失败:" + e.getMessage()); return null; }
            return drawEyeMouthKeypoints(srcBmp, result);
        } catch (Exception e) {
            Log.e(TAG, "预览失败:" + e.getMessage());
        }
        return null;
    }

    // 绘制眼睛/嘴巴/鼻子关键点与引导线，并标注 EAR/MAR
    private FaceStateResult drawEyeMouthKeypoints(Bitmap src, FaceLandmarkerResult result) {
        Bitmap out = src.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(out);
        int w = out.getWidth(), h = out.getHeight();
        String description = "未检测到人脸";
        Paint pt = new Paint(Paint.ANTI_ALIAS_FLAG);
        pt.setStyle(Paint.Style.FILL);
        float r = Math.max(4f, Math.min(w, h) * 0.006f);
        Paint line = new Paint(Paint.ANTI_ALIAS_FLAG);
        line.setStyle(Paint.Style.STROKE);
        line.setStrokeWidth(Math.max(3f, Math.min(w, h) * 0.004f));
        Paint text = new Paint(Paint.ANTI_ALIAS_FLAG);
        text.setColor(Color.WHITE);
        text.setTextSize(Math.max(18f, Math.min(w, h) * 0.024f));
        text.setShadowLayer(3f, 1.5f, 1.5f, Color.BLACK);

        if (result == null || result.faceLandmarks().isEmpty()) {
            canvas.drawText("未检测到人脸", r * 2, r * 8, text);
            return new FaceStateResult(out, description);
        }
        List<NormalizedLandmark> lms = result.faceLandmarks().get(0);
        try {
            // FaceMesh 索引
            int LE_v1a = 159, LE_v1b = 145, LE_v2a = 158, LE_v2b = 153, LE_h1 = 33, LE_h2 = 133;
            int RE_v1a = 386, RE_v1b = 374, RE_v2a = 385, RE_v2b = 380, RE_h1 = 362, RE_h2 = 263;
            int M_v1 = 13, M_v2 = 14, M_h1 = 61, M_h2 = 291;
            int N_tip = 1, N_base = 2;

            // 像素坐标
            float[] pLE_v1a = px(lms.get(LE_v1a), w, h), pLE_v1b = px(lms.get(LE_v1b), w, h);
            float[] pLE_v2a = px(lms.get(LE_v2a), w, h), pLE_v2b = px(lms.get(LE_v2b), w, h);
            float[] pRE_v1a = px(lms.get(RE_v1a), w, h), pRE_v1b = px(lms.get(RE_v1b), w, h);
            float[] pRE_v2a = px(lms.get(RE_v2a), w, h), pRE_v2b = px(lms.get(RE_v2b), w, h);
            float[] pM_v1 = px(lms.get(M_v1), w, h), pM_v2 = px(lms.get(M_v2), w, h);
            float[] pN_tip = px(lms.get(N_tip), w, h), pN_base = px(lms.get(N_base), w, h);

            int colLeftEye = Color.parseColor("#3DDC84");
            int colRightEye = Color.parseColor("#33B5E5");
            int colMouth = Color.parseColor("#FF4081");
            int colNose = Color.parseColor("#FFD600");

            // 左眼
            line.setColor(colLeftEye);
            canvas.drawLine(pLE_v1a[0], pLE_v1a[1], pLE_v1b[0], pLE_v1b[1], line);
            canvas.drawLine(pLE_v2a[0], pLE_v2a[1], pLE_v2b[0], pLE_v2b[1], line);
            drawPtLabel(canvas, pt, text, r, LE_v1a, pLE_v1a);
            drawPtLabel(canvas, pt, text, r, LE_v1b, pLE_v1b);
            drawPtLabel(canvas, pt, text, r, LE_v2a, pLE_v2a);
            drawPtLabel(canvas, pt, text, r, LE_v2b, pLE_v2b);

            // 右眼
            line.setColor(colRightEye);
            canvas.drawLine(pRE_v1a[0], pRE_v1a[1], pRE_v1b[0], pRE_v1b[1], line);
            canvas.drawLine(pRE_v2a[0], pRE_v2a[1], pRE_v2b[0], pRE_v2b[1], line);
            drawPtLabel(canvas, pt, text, r, RE_v1a, pRE_v1a);
            drawPtLabel(canvas, pt, text, r, RE_v1b, pRE_v1b);
            drawPtLabel(canvas, pt, text, r, RE_v2a, pRE_v2a);
            drawPtLabel(canvas, pt, text, r, RE_v2b, pRE_v2b);

            // 嘴
            line.setColor(colMouth);
            canvas.drawLine(pM_v1[0], pM_v1[1], pM_v2[0], pM_v2[1], line);
            drawPtLabel(canvas, pt, text, r, M_v1, pM_v1);
            drawPtLabel(canvas, pt, text, r, M_v2, pM_v2);

            // 鼻子
            line.setColor(colNose);
            canvas.drawLine(pN_tip[0], pN_tip[1], pN_base[0], pN_base[1], line);
            drawPtLabel(canvas, pt, text, r, N_tip, pN_tip);
            drawPtLabel(canvas, pt, text, r, N_base, pN_base);

            // 计算 EAR/MAR
            float leftV1 = distPx(lms, LE_v1a, LE_v1b, w, h);
            float leftV2 = distPx(lms, LE_v2a, LE_v2b, w, h);
            float leftH = distPx(lms, LE_h1, LE_h2, w, h);
            float rightV1 = distPx(lms, RE_v1a, RE_v1b, w, h);
            float rightV2 = distPx(lms, RE_v2a, RE_v2b, w, h);
            float rightH = distPx(lms, RE_h1, RE_h2, w, h);
            float leftEAR = (leftV1 + leftV2) / (2f * Math.max(1e-6f, leftH));
            float rightEAR = (rightV1 + rightV2) / (2f * Math.max(1e-6f, rightH));
            float mar = distPx(lms, M_v1, M_v2, w, h) / Math.max(1e-6f, distPx(lms, M_h1, M_h2, w, h));
            float noseLen = distPx(lms, N_tip, N_base, w, h);

            // 坐姿判断
            float leftEyeCenterX = (lms.get(LE_h1).x() + lms.get(LE_h2).x()) / 2f * w;
            float rightEyeCenterX = (lms.get(RE_h1).x() + lms.get(RE_h2).x()) / 2f * w;
            float noseX = lms.get(N_tip).x() * w;
            float distLeft = Math.abs(noseX - leftEyeCenterX);
            float distRight = Math.abs(rightEyeCenterX - noseX);

            final float EAR_THRESH = 0.18f;
            final float MAR_THRESH = 0.55f;
            boolean leftOpen = leftEAR > EAR_THRESH || (leftV1 > 20f && leftV2 > 20f);
            boolean rightOpen = rightEAR > EAR_THRESH || (rightV1 > 20f && rightV2 > 20f);
            boolean mouthOpen = mar > MAR_THRESH || distPx(lms, M_v1, M_v2, w, h) > 25f;
            boolean postureOk = Math.abs(distLeft - distRight) <= 25f;

            Log.i(TAG, "左眼 EAR=" + String.format(Locale.US, "%.3f", leftEAR) + " " + (leftOpen ? "睁开" : "闭合"));
            Log.i(TAG, "右眼 EAR=" + String.format(Locale.US, "%.3f", rightEAR) + " " + (rightOpen ? "睁开" : "闭合"));
            Log.i(TAG, "嘴 MAR=" + String.format(Locale.US, "%.3f", mar) + " " + (mouthOpen ? "张开" : "闭合"));
            Log.i(TAG, "坐姿: " + (postureOk ? "正确" : "不正"));

            description = String.format(Locale.US,
                    "左眼: %s (EAR=%.3f)\n右眼: %s (EAR=%.3f)\n嘴巴: %s (MAR=%.3f)\n鼻子长度: %.1f\n坐姿: %s",
                    leftOpen ? "睁开" : "闭合", leftEAR,
                    rightOpen ? "睁开" : "闭合", rightEAR,
                    mouthOpen ? "张开" : "闭合", mar,
                    noseLen,
                    postureOk ? "正确" : "不正");

            // 文字说明
            float ty = r * 6;
            canvas.drawText("左眼 EAR=" + String.format(Locale.US, "%.3f", leftEAR), r * 2, ty, text);
            ty += text.getTextSize() * 1.4f;
            canvas.drawText("右眼 EAR=" + String.format(Locale.US, "%.3f", rightEAR), r * 2, ty, text);
            ty += text.getTextSize() * 1.4f;
            canvas.drawText("嘴部 MAR=" + String.format(Locale.US, "%.3f", mar), r * 2, ty, text);
            ty += text.getTextSize() * 1.4f;
            canvas.drawText("鼻子长度=" + String.format(Locale.US, "%.1f", noseLen), r * 2, ty, text);

        } catch (Exception e) {
            Log.e(TAG, "绘制关键点失败:" + e.getMessage());
            description = "检测失败: " + e.getMessage();
        }
        return new FaceStateResult(out, description);
    }

    private void drawPtLabel(Canvas canvas, Paint pt, Paint text, float r, int idx, float[] p) {
        pt.setColor(Color.WHITE);
        canvas.drawCircle(p[0], p[1], r + 1f, pt);
        pt.setColor(Color.BLACK);
        canvas.drawCircle(p[0], p[1], r - 1f, pt);
        canvas.drawText(String.valueOf(idx), p[0] + r * 1.2f, p[1] - r * 1.2f, text);
    }

    private float[] px(NormalizedLandmark lm, int w, int h) {
        return new float[]{lm.x() * w, lm.y() * h};
    }

    private float distPx(List<NormalizedLandmark> lms, int i, int j, int w, int h) {
        float dx = (lms.get(i).x() - lms.get(j).x()) * w;
        float dy = (lms.get(i).y() - lms.get(j).y()) * h;
        return (float) Math.hypot(dx, dy);
    }

    /**
     * 人脸五官检测结果封装类。
     */
    public static class FaceStateResult {
        private final Bitmap annotatedBitmap;
        private final String description;

        public FaceStateResult(Bitmap annotatedBitmap, String description) {
            this.annotatedBitmap = annotatedBitmap;
            this.description = description;
        }

        /** 标注后的 Bitmap */
        public Bitmap getAnnotatedBitmap() { return annotatedBitmap; }

        /** 检测结果的文字描述 */
        public String getDescription() { return description; }
    }
}
