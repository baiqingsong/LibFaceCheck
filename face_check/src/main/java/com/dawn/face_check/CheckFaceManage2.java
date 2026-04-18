package com.dawn.face_check;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Locale;

/**
 * 人脸语义分割管理类。
 * <p>
 * 基于 PyTorch bisenet 模型进行人脸语义分割（19 类），支持检测：
 * 眼镜(6)、左耳(7)、右耳(8)、鼻子(10) 等部位，并进行遮挡与对称性判断。
 * <p>
 * 类别对照表：
 * 0=背景, 1=脸, 2=左眉, 3=右眉, 4=左眼, 5=右眼, 6=眼镜, 7=左耳, 8=右耳,
 * 9=耳环, 10=鼻子, 11=牙齿, 12=上唇, 13=下唇, 14=脖子, 15=项链, 16=衣服, 17=头发, 18=帽子
 * <p>
 * 使用示例：
 * <pre>
 *   CheckFaceManage2 manage2 = new CheckFaceManage2();
 *   Module module = manage2.loadModel(context, "bisenet_512.pt");
 *   CheckFaceManage2.SegResult result = manage2.runSegmentationAndAnalyze(context, module, bitmap);
 * </pre>
 */
public class CheckFaceManage2 {

    private static final String TAG = "CheckFaceManage2";

    /**
     * 从 assets 加载 PyTorch 模型。
     *
     * @param context  Android Context
     * @param fileName assets 中的模型文件名（如 "bisenet_512.pt"）
     * @return 已加载的 PyTorch Module
     */
    public Module loadModel(Context context, String fileName) throws IOException {
        File modelFile = new File(context.getFilesDir(), fileName);
        if (!modelFile.exists()) {
            try (InputStream is = context.getAssets().open(fileName);
                 OutputStream os = new FileOutputStream(modelFile)) {
                byte[] buffer = new byte[8192];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
            }
        }
        return Module.load(modelFile.getAbsolutePath());
    }

    /**
     * 便捷方法：加载 bisenet_512.pt 模型并运行分割分析。
     *
     * @param context Android Context
     * @param bitmap  输入 Bitmap
     * @return SegResult
     */
    public SegResult analyzeFromAssets(Context context, Bitmap bitmap) throws IOException {
        Module module = loadModel(context, "bisenet_512.pt");
        return runSegmentationAndAnalyze(context, module, bitmap);
    }

    /**
     * 运行分割模型并对结果进行分析。
     *
     * @param context Android Context
     * @param module  已加载的 PyTorch Module（分割模型）
     * @param finalBm 输入 Bitmap（原始图片）
     * @return 包含结果描述文本和叠加图路径的 SegResult
     */
    public SegResult runSegmentationAndAnalyze(Context context, Module module, Bitmap finalBm) {
        Log.i(TAG, "开始图像分割推理...");
        int origW = finalBm.getWidth();
        int origH = finalBm.getHeight();

        // 将图片缩放到 512x512 以匹配模型输入尺寸并减少内存占用
        final int MODEL_SIZE = 512;
        Bitmap scaledBm = Bitmap.createScaledBitmap(finalBm, MODEL_SIZE, MODEL_SIZE, true);
        int scaledW = scaledBm.getWidth();
        int scaledH = scaledBm.getHeight();
        int padH = ((scaledH + 31) / 32) * 32;
        int padW = ((scaledW + 31) / 32) * 32;

        // 边界反射填充
        int[] srcPixels = new int[scaledW * scaledH];
        scaledBm.getPixels(srcPixels, 0, scaledW, 0, 0, scaledW, scaledH);
        scaledBm.recycle();

        int[] paddedPixels = new int[padW * padH];
        for (int y = 0; y < padH; y++) {
            int srcY = y < scaledH ? y : (2 * scaledH - y - 1);
            if (srcY < 0) srcY = 0;
            if (srcY >= scaledH) srcY = scaledH - 1;
            for (int x = 0; x < padW; x++) {
                int srcX = x < scaledW ? x : (2 * scaledW - x - 1);
                if (srcX < 0) srcX = 0;
                if (srcX >= scaledW) srcX = scaledW - 1;
                paddedPixels[y * padW + x] = srcPixels[srcY * scaledW + srcX];
            }
        }

        // 归一化 + CHW 格式转换
        float[] mean = new float[]{0.485f, 0.456f, 0.406f};
        float[] std = new float[]{0.229f, 0.224f, 0.225f};
        int phw = padH * padW;
        float[] inputData = new float[3 * phw];
        for (int y = 0; y < padH; y++) {
            for (int x = 0; x < padW; x++) {
                int argb = paddedPixels[y * padW + x];
                int rr = (argb >> 16) & 0xFF;
                int gg = (argb >> 8) & 0xFF;
                int bb = argb & 0xFF;
                int idx = y * padW + x;
                inputData[idx] = (rr / 255.0f - mean[0]) / std[0];
                inputData[phw + idx] = (gg / 255.0f - mean[1]) / std[1];
                inputData[2 * phw + idx] = (bb / 255.0f - mean[2]) / std[2];
            }
        }

        Tensor inputTensor = Tensor.fromBlob(inputData, new long[]{1, 3, padH, padW});
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        long[] outShape = outputTensor.shape();

        int[] parsing = new int[scaledH * scaledW];

        if (outShape.length == 4) {
            int numClasses = (int) outShape[1];
            int outH = (int) outShape[2];
            int outW = (int) outShape[3];
            float[] outFloats = null;
            long[] outLongs = null;
            try { outFloats = outputTensor.getDataAsFloatArray(); } catch (Exception e) {
                try { outLongs = outputTensor.getDataAsLongArray(); } catch (Exception ex) {
                    throw new RuntimeException("无法解析模型输出: " + ex.getMessage());
                }
            }
            int hw = outH * outW;
            if (outH == padH && outW == padW) {
                for (int yy = 0; yy < scaledH; yy++) {
                    for (int xx = 0; xx < scaledW; xx++) {
                        int bestClass = 0;
                        float bestVal = Float.NEGATIVE_INFINITY;
                        int idxInMap = yy * outW + xx;
                        if (outFloats != null) {
                            for (int cls = 0; cls < numClasses; cls++) {
                                float val = outFloats[cls * hw + idxInMap];
                                if (val > bestVal) { bestVal = val; bestClass = cls; }
                            }
                        } else if (outLongs != null) {
                            bestClass = (int) outLongs[idxInMap];
                        }
                        parsing[yy * scaledW + xx] = bestClass;
                    }
                }
            } else {
                for (int yy = 0; yy < scaledH; yy++) {
                    for (int xx = 0; xx < scaledW; xx++) {
                        int bestClass = 0;
                        float bestVal = Float.NEGATIVE_INFINITY;
                        int mappedY = Math.min(outH - 1, (int) ((yy / (float) scaledH) * outH));
                        int mappedX = Math.min(outW - 1, (int) ((xx / (float) scaledW) * outW));
                        int idxInMap = mappedY * outW + mappedX;
                        if (outFloats != null) {
                            for (int cls = 0; cls < numClasses; cls++) {
                                float val = outFloats[cls * hw + idxInMap];
                                if (val > bestVal) { bestVal = val; bestClass = cls; }
                            }
                        } else if (outLongs != null) {
                            bestClass = (int) outLongs[idxInMap];
                        }
                        parsing[yy * scaledW + xx] = bestClass;
                    }
                }
            }
        } else if (outShape.length == 3) {
            int outH = (int) outShape[1];
            int outW = (int) outShape[2];
            try {
                long[] outLongs = outputTensor.getDataAsLongArray();
                if (outH == padH && outW == padW) {
                    for (int yy = 0; yy < scaledH; yy++) {
                        for (int xx = 0; xx < scaledW; xx++) {
                            int idxInMap = yy * outW + xx;
                            parsing[yy * scaledW + xx] = (idxInMap >= 0 && idxInMap < outLongs.length) ? (int) outLongs[idxInMap] : 0;
                        }
                    }
                } else {
                    for (int yy = 0; yy < scaledH; yy++) {
                        for (int xx = 0; xx < scaledW; xx++) {
                            int mappedY = Math.min(outH - 1, (int) ((yy * (long) outH) / (long) scaledH));
                            int mappedX = Math.min(outW - 1, (int) ((xx * (long) outW) / (long) scaledW));
                            int idx = mappedY * outW + mappedX;
                            parsing[yy * scaledW + xx] = (idx >= 0 && idx < outLongs.length) ? (int) outLongs[idx] : 0;
                        }
                    }
                }
            } catch (Exception e) {
                float[] outFloats = outputTensor.getDataAsFloatArray();
                if (outFloats == null) throw new RuntimeException("无法解析模型输出为 float[]");
                if (outH == padH && outW == padW) {
                    for (int yy = 0; yy < scaledH; yy++) {
                        for (int xx = 0; xx < scaledW; xx++) {
                            int idxInMap = yy * outW + xx;
                            parsing[yy * scaledW + xx] = (idxInMap >= 0 && idxInMap < outFloats.length) ? (int) outFloats[idxInMap] : 0;
                        }
                    }
                } else {
                    for (int yy = 0; yy < scaledH; yy++) {
                        for (int xx = 0; xx < scaledW; xx++) {
                            int mappedY = Math.min(outH - 1, (int) ((yy * (long) outH) / (long) scaledH));
                            int mappedX = Math.min(outW - 1, (int) ((xx * (long) outW) / (long) scaledW));
                            int idx = mappedY * outW + mappedX;
                            parsing[yy * scaledW + xx] = (idx >= 0 && idx < outFloats.length) ? (int) outFloats[idx] : 0;
                        }
                    }
                }
            }
        }

        // 统计每类像素数
        int maxClasses = 30;
        int[] classCounts = new int[maxClasses];
        for (int cls : parsing) {
            if (cls >= 0 && cls < maxClasses) classCounts[cls]++;
        }

        // 保存特定类别掩码（眼镜6、左耳7、右耳8、鼻子10）
        for (int cls : new int[]{6, 7, 8, 10}) {
            if (classCounts[cls] <= 0) continue;
            int[] maskPixels = new int[scaledW * scaledH];
            for (int pi = 0; pi < scaledW * scaledH; pi++) {
                maskPixels[pi] = (parsing[pi] == cls) ? 0xFFFFFFFF : 0xFF000000;
            }
            Bitmap maskBmp = Bitmap.createBitmap(scaledW, scaledH, Bitmap.Config.ARGB_8888);
            maskBmp.setPixels(maskPixels, 0, scaledW, 0, 0, scaledW, scaledH);
            File maskFile = new File(context.getFilesDir(), String.format(Locale.US, "mask_class_%02d.png", cls));
            try (FileOutputStream fos = new FileOutputStream(maskFile)) {
                maskBmp.compress(Bitmap.CompressFormat.PNG, 100, fos);
                fos.flush();
                Log.i(TAG, "已保存类别掩码: " + maskFile.getAbsolutePath() + " (像素=" + classCounts[cls] + ")");
            } catch (Exception e) {
                Log.e(TAG, "保存 mask_class_" + cls + " 失败: " + e.getMessage());
            }
            maskBmp.recycle();
        }
        Log.i(TAG, "类别像素统计: " + Arrays.toString(classCounts));

        // 构造文本结果
        StringBuilder sb = new StringBuilder();
        sb.append("类别像素统计:\n");
        for (int i = 0; i < Math.min(classCounts.length, 19); i++) {
            sb.append(String.format(Locale.US, "类 %02d: %d\n", i, classCounts[i]));
        }

        // 生成叠加图
        int[] colorMap = new int[]{
                0xFF000000, 0xFFCC0000, 0xFF4C9900, 0xFFCCCC00,
                0xFF3333FF, 0xFFCC00CC, 0xFF00FFFF, 0xFFFFCCCC,
                0xFF663300, 0xFFFF0000, 0xFF66CC00, 0xFFFFFF00,
                0xFF000099, 0xFF0000CC, 0xFFFF3399, 0xFF00CCCC,
                0xFF003300, 0xFFFF9933, 0xFF00CC00
        };
        int[] overlayPixels = new int[scaledW * scaledH];
        for (int i = 0; i < scaledW * scaledH; i++) {
            int cls = parsing[i];
            int maskColor = colorMap[cls < colorMap.length ? cls : 0];
            int orig = srcPixels[i];
            int or2 = (orig >> 16) & 0xFF, og = (orig >> 8) & 0xFF, ob = orig & 0xFF;
            int mr = (maskColor >> 16) & 0xFF, mg = (maskColor >> 8) & 0xFF, mb = maskColor & 0xFF;
            overlayPixels[i] = 0xFF000000 | (((or2 + mr) / 2) << 16) | (((og + mg) / 2) << 8) | ((ob + mb) / 2);
        }
        Bitmap overlayBmp = Bitmap.createBitmap(scaledW, scaledH, Bitmap.Config.ARGB_8888);
        overlayBmp.setPixels(overlayPixels, 0, scaledW, 0, 0, scaledW, scaledH);
        File outFile = new File(context.getFilesDir(), "result_overlay.jpg");
        try (FileOutputStream fos = new FileOutputStream(outFile)) {
            overlayBmp.compress(Bitmap.CompressFormat.JPEG, 90, fos);
            fos.flush();
            Log.i(TAG, "已保存叠加图: " + outFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e(TAG, "保存叠加图失败: " + e.getMessage());
        }
        overlayBmp.recycle();
        String overlayPath = outFile.getAbsolutePath();

        // 自动运行 checkFace 做进一步判定
        try {
            String checkResult = checkFace(context);
            sb.append("\n").append(checkResult);
        } catch (Exception e) {
            Log.e(TAG, "自动运行 checkFace 失败: " + e.getMessage());
        }

        return new SegResult(sb.toString(), overlayPath);
    }

    /**
     * 分割结果封装类。
     */
    public static class SegResult {
        private final String text;
        private final String overlayPath;

        public SegResult(String text, String overlayPath) {
            this.text = text;
            this.overlayPath = overlayPath;
        }

        /** 检测文字描述 */
        public String getText() { return text; }

        /** 叠加图的文件路径 */
        public String getOverlayPath() { return overlayPath; }
    }

    // ======================== 内部方法 ========================

    private String checkFace(Context context) {
        StringBuilder result = new StringBuilder();
        try {
            File dir = context.getFilesDir();
            // 鼻子 mask (class 10)
            File noseFile = new File(dir, String.format(Locale.US, "mask_class_%02d.png", 10));
            if (!noseFile.exists()) { Log.e(TAG, "未找到鼻子掩码"); return "未找到鼻子掩码"; }
            Bitmap noseBmp = BitmapFactory.decodeFile(noseFile.getAbsolutePath());
            if (noseBmp == null) { Log.e(TAG, "鼻子掩码解码失败"); return "鼻子掩码解码失败"; }
            int w = noseBmp.getWidth(), h = noseBmp.getHeight();
            float[] noseCent = computeCentroid(noseBmp);
            noseBmp.recycle();
            if (noseCent == null) { Log.e(TAG, "鼻子掩码中未检测到像素"); return "鼻子掩码中未检测到像素"; }
            Log.i(TAG, String.format(Locale.US, "鼻子质心: (%.1f, %.1f)", noseCent[0], noseCent[1]));
            result.append(String.format(Locale.US, "鼻子质心: (%.1f, %.1f)\n", noseCent[0], noseCent[1]));

            // 眼镜 mask (class 6)
            File glassFile = new File(dir, String.format(Locale.US, "mask_class_%02d.png", 6));
            if (glassFile.exists()) {
                Bitmap glassBmp = BitmapFactory.decodeFile(glassFile.getAbsolutePath());
                if (glassBmp != null) {
                    int glassCount = countWhitePixels(glassBmp);
                    Rect glassBox = computeBoundingBox(glassBmp);
                    glassBmp.recycle();
                    if (glassBox != null) {
                        float glassCx = glassBox.centerX();
                        float areaRatio = glassCount / (float) (w * h);
                        boolean centerOk = Math.abs(glassCx - noseCent[0]) <= w * 0.15f;
                        boolean areaOk = areaRatio >= 0.002f;
                        String glassInfo = String.format(Locale.US, "眼镜: 像素=%d, 面积比=%.4f, 居中=%s, 面积足够=%s",
                                glassCount, areaRatio, centerOk ? "是" : "否", areaOk ? "是" : "否");
                        Log.i(TAG, glassInfo);
                        result.append(glassInfo).append("\n");
                    }
                }
            }

            // 左右耳合并并判断
            File leftEarFile = new File(dir, String.format(Locale.US, "mask_class_%02d.png", 7));
            File rightEarFile = new File(dir, String.format(Locale.US, "mask_class_%02d.png", 8));
            Bitmap leftBmp = leftEarFile.exists() ? BitmapFactory.decodeFile(leftEarFile.getAbsolutePath()) : null;
            Bitmap rightBmp = rightEarFile.exists() ? BitmapFactory.decodeFile(rightEarFile.getAbsolutePath()) : null;

            if (leftBmp == null && rightBmp == null) {
                Log.e(TAG, "未检测到左右耳掩码");
                result.append("未检测到左右耳掩码\n");
            } else {
                Bitmap merged = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                int[] mergedPixels = new int[w * h];
                int[] leftPixels = null, rightPixels = null;
                if (leftBmp != null) { leftPixels = new int[w * h]; leftBmp.getPixels(leftPixels, 0, w, 0, 0, w, h); leftBmp.recycle(); }
                if (rightBmp != null) { rightPixels = new int[w * h]; rightBmp.getPixels(rightPixels, 0, w, 0, 0, w, h); rightBmp.recycle(); }
                for (int i = 0; i < w * h; i++) {
                    boolean l = leftPixels != null && (leftPixels[i] & 0x00FFFFFF) != 0;
                    boolean r = rightPixels != null && (rightPixels[i] & 0x00FFFFFF) != 0;
                    mergedPixels[i] = (l || r) ? 0xFFFFFFFF : 0xFF000000;
                }
                merged.setPixels(mergedPixels, 0, w, 0, 0, w, h);
                File mergedFile = new File(dir, "merged_ears.png");
                try (FileOutputStream fos = new FileOutputStream(mergedFile)) {
                    merged.compress(Bitmap.CompressFormat.PNG, 100, fos);
                    fos.flush();
                    Log.i(TAG, "已保存合并耳朵图片: " + mergedFile.getAbsolutePath());
                } catch (Exception e) { Log.e(TAG, "保存合并耳朵失败:" + e.getMessage()); }
                merged.recycle();

                // 耳对称性判断
                int leftCount = 0, rightCount = 0;
                long sumLx = 0, sumLy = 0, sumRx = 0, sumRy = 0;
                for (int y = 0; y < h; y++) {
                    int base = y * w;
                    for (int x = 0; x < w; x++) {
                        int v = mergedPixels[base + x] & 0x00FFFFFF;
                        if (v != 0) {
                            if (x < noseCent[0]) { sumLx += x; sumLy += y; leftCount++; }
                            else { sumRx += x; sumRy += y; rightCount++; }
                        }
                    }
                }
                float[] leftCent = leftCount > 0 ? new float[]{sumLx / (float) leftCount, sumLy / (float) leftCount} : null;
                float[] rightCent = rightCount > 0 ? new float[]{sumRx / (float) rightCount, sumRy / (float) rightCount} : null;

                boolean symmetryOk = false;
                boolean sizeOk = false;
                if (leftCent != null && rightCent != null) {
                    float noseX = noseCent[0];
                    float noseY = noseCent[1];
                    float mirroredLX = 2 * noseX - leftCent[0];
                    float dx = Math.abs(mirroredLX - rightCent[0]);
                    float dy = Math.abs(leftCent[1] - rightCent[1]);
                    float nx = dx / (float) w;
                    float ny = dy / (float) h;
                    float sizeRatio = Math.abs(leftCount - rightCount) / (float) Math.max(1, (leftCount + rightCount) / 2);
                    double angleL = Math.atan2(leftCent[1] - noseY, leftCent[0] - noseX);
                    double angleR = Math.atan2(rightCent[1] - noseY, rightCent[0] - noseX);
                    double mirroredAngleL = Math.PI - angleL;
                    double dAngle = Math.abs(mirroredAngleL - angleR);
                    while (dAngle > Math.PI) dAngle = Math.abs(dAngle - 2 * Math.PI);

                    symmetryOk = (nx <= 0.10f) && (ny <= 0.15f) && (dAngle <= 0.9);
                    sizeOk = sizeRatio <= 0.5f;

                    String symInfo = String.format(Locale.US, "耳朵对称性: nx=%.3f ny=%.3f sizeRatio=%.3f dAngle=%.3f => 对称=%s 大小相近=%s",
                            nx, ny, sizeRatio, dAngle, symmetryOk ? "是" : "否", sizeOk ? "是" : "否");
                    Log.i(TAG, symInfo);
                    result.append(symInfo).append("\n");
                }
                String earResult = "最终耳朵判定: 对称=" + (symmetryOk ? "是" : "否") + ", 大小相近=" + (sizeOk ? "是" : "否");
                Log.i(TAG, earResult);
                result.append(earResult).append("\n");
            }
        } catch (Exception e) {
            Log.e(TAG, "checkFace 失败: " + e.getMessage());
            result.append("分析失败: ").append(e.getMessage());
        }
        return result.toString();
    }

    private float[] computeCentroid(Bitmap mask) {
        int w = mask.getWidth(), h = mask.getHeight();
        int[] px = new int[w * h];
        mask.getPixels(px, 0, w, 0, 0, w, h);
        long sumX = 0, sumY = 0, cnt = 0;
        for (int y = 0; y < h; y++) {
            int base = y * w;
            for (int x = 0; x < w; x++) {
                if ((px[base + x] & 0x00FFFFFF) != 0) { sumX += x; sumY += y; cnt++; }
            }
        }
        return cnt == 0 ? null : new float[]{sumX / (float) cnt, sumY / (float) cnt};
    }

    private Rect computeBoundingBox(Bitmap mask) {
        int w = mask.getWidth(), h = mask.getHeight();
        int[] px = new int[w * h];
        mask.getPixels(px, 0, w, 0, 0, w, h);
        int minX = w, minY = h, maxX = 0, maxY = 0;
        boolean any = false;
        for (int y = 0; y < h; y++) {
            int base = y * w;
            for (int x = 0; x < w; x++) {
                if ((px[base + x] & 0x00FFFFFF) != 0) {
                    any = true;
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }
        }
        return any ? new Rect(minX, minY, maxX + 1, maxY + 1) : null;
    }

    private int countWhitePixels(Bitmap mask) {
        int w = mask.getWidth(), h = mask.getHeight();
        int[] px = new int[w * h];
        mask.getPixels(px, 0, w, 0, 0, w, h);
        int cnt = 0;
        for (int v : px) if ((v & 0x00FFFFFF) != 0) cnt++;
        return cnt;
    }
}
