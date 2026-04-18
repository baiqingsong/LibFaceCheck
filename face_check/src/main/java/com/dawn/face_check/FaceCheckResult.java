package com.dawn.face_check;

import android.graphics.Bitmap;
import android.graphics.PointF;

import java.util.List;

/**
 * 人脸检测结果封装类。
 * 包含人脸关键点坐标（像素坐标）、耳朵可见性判断、以及可选的标注后图片。
 */
public class FaceCheckResult {

    private final boolean faceDetected;
    private final List<PointF> landmarks;
    private final boolean leftEarVisible;
    private final boolean rightEarVisible;
    private final int imageWidth;
    private final int imageHeight;
    private Bitmap annotatedBitmap;

    public FaceCheckResult(boolean faceDetected,
                           List<PointF> landmarks,
                           boolean leftEarVisible,
                           boolean rightEarVisible,
                           int imageWidth,
                           int imageHeight) {
        this.faceDetected = faceDetected;
        this.landmarks = landmarks;
        this.leftEarVisible = leftEarVisible;
        this.rightEarVisible = rightEarVisible;
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
    }

    /** 是否检测到人脸 */
    public boolean isFaceDetected() {
        return faceDetected;
    }

    /** 人脸关键点列表（像素坐标），未检测到时为空列表 */
    public List<PointF> getLandmarks() {
        return landmarks;
    }

    /** 左耳是否可见（启发式判断） */
    public boolean isLeftEarVisible() {
        return leftEarVisible;
    }

    /** 右耳是否可见（启发式判断） */
    public boolean isRightEarVisible() {
        return rightEarVisible;
    }

    /** 输入图片宽度 */
    public int getImageWidth() {
        return imageWidth;
    }

    /** 输入图片高度 */
    public int getImageHeight() {
        return imageHeight;
    }

    /** 标注后的 Bitmap（仅在调用 detectAndDraw 后有值） */
    public Bitmap getAnnotatedBitmap() {
        return annotatedBitmap;
    }

    void setAnnotatedBitmap(Bitmap annotatedBitmap) {
        this.annotatedBitmap = annotatedBitmap;
    }

    /** 关键点数量 */
    public int getLandmarkCount() {
        return landmarks != null ? landmarks.size() : 0;
    }
}
