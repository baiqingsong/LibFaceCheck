package com.dawn.face_check;

/**
 * 人脸综合检测结果。
 * <p>
 * 包含人脸检测、五官可见性、眼睛睁闭、嘴巴张闭、姿态端正等所有检测数据。
 */
public class FaceDetectResult {

    /** 是否检测到人脸 */
    private final boolean faceDetected;

    /** 左眼是否睁开 */
    private final boolean leftEyeOpen;
    /** 右眼是否睁开 */
    private final boolean rightEyeOpen;

    /** 左眼是否可见（能看到） */
    private final boolean leftEyeVisible;
    /** 右眼是否可见（能看到） */
    private final boolean rightEyeVisible;

    /** 鼻子是否可见 */
    private final boolean noseVisible;

    /** 嘴巴是否闭合 */
    private final boolean mouthClosed;

    /** 人脸是否端正（基于鼻子-眼睛对称性） */
    private final boolean faceStraight;

    /** 左耳是否可见 */
    private final boolean leftEarVisible;
    /** 右耳是否可见 */
    private final boolean rightEarVisible;

    /** 左眼 EAR 值（Eye Aspect Ratio） */
    private final float leftEyeEAR;
    /** 右眼 EAR 值 */
    private final float rightEyeEAR;
    /** 嘴巴 MAR 值（Mouth Aspect Ratio） */
    private final float mouthMAR;

    FaceDetectResult(boolean faceDetected,
                     boolean leftEyeOpen, boolean rightEyeOpen,
                     boolean leftEyeVisible, boolean rightEyeVisible,
                     boolean noseVisible,
                     boolean mouthClosed,
                     boolean faceStraight,
                     boolean leftEarVisible, boolean rightEarVisible,
                     float leftEyeEAR, float rightEyeEAR, float mouthMAR) {
        this.faceDetected = faceDetected;
        this.leftEyeOpen = leftEyeOpen;
        this.rightEyeOpen = rightEyeOpen;
        this.leftEyeVisible = leftEyeVisible;
        this.rightEyeVisible = rightEyeVisible;
        this.noseVisible = noseVisible;
        this.mouthClosed = mouthClosed;
        this.faceStraight = faceStraight;
        this.leftEarVisible = leftEarVisible;
        this.rightEarVisible = rightEarVisible;
        this.leftEyeEAR = leftEyeEAR;
        this.rightEyeEAR = rightEyeEAR;
        this.mouthMAR = mouthMAR;
    }

    /** 未检测到人脸时的默认结果 */
    static FaceDetectResult noFace() {
        return new FaceDetectResult(false,
                false, false, false, false,
                false, false, false, false, false,
                0f, 0f, 0f);
    }

    public boolean isFaceDetected() { return faceDetected; }

    public boolean isLeftEyeOpen() { return leftEyeOpen; }
    public boolean isRightEyeOpen() { return rightEyeOpen; }

    public boolean isLeftEyeVisible() { return leftEyeVisible; }
    public boolean isRightEyeVisible() { return rightEyeVisible; }
    /** 双眼是否都可见 */
    public boolean isBothEyesVisible() { return leftEyeVisible && rightEyeVisible; }

    public boolean isNoseVisible() { return noseVisible; }

    public boolean isMouthClosed() { return mouthClosed; }

    public boolean isFaceStraight() { return faceStraight; }

    public boolean isLeftEarVisible() { return leftEarVisible; }
    public boolean isRightEarVisible() { return rightEarVisible; }
    /** 双耳是否都可见 */
    public boolean isBothEarsVisible() { return leftEarVisible && rightEarVisible; }

    public float getLeftEyeEAR() { return leftEyeEAR; }
    public float getRightEyeEAR() { return rightEyeEAR; }
    public float getMouthMAR() { return mouthMAR; }

    @Override
    public String toString() {
        if (!faceDetected) return "FaceDetectResult{未检测到人脸}";
        return "FaceDetectResult{" +
                "左眼=" + (leftEyeOpen ? "睁开" : "闭合") + "(EAR=" + String.format("%.3f", leftEyeEAR) + ")" +
                ", 右眼=" + (rightEyeOpen ? "睁开" : "闭合") + "(EAR=" + String.format("%.3f", rightEyeEAR) + ")" +
                ", 左眼可见=" + leftEyeVisible +
                ", 右眼可见=" + rightEyeVisible +
                ", 鼻子可见=" + noseVisible +
                ", 嘴巴=" + (mouthClosed ? "闭合" : "张开") + "(MAR=" + String.format("%.3f", mouthMAR) + ")" +
                ", 人脸端正=" + faceStraight +
                ", 左耳可见=" + leftEarVisible +
                ", 右耳可见=" + rightEarVisible +
                '}';
    }
}
