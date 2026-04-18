package com.dawn.face_check;

import android.content.res.AssetManager;

import java.io.InputStream;
import java.nio.ByteBuffer;

/**
 * 模型加载工具类，提供 FaceLandmarker 模型查找和加载能力。
 */
class FaceModelHelper {

    private static final String[] MODEL_CANDIDATE_PATHS = {
            "models/face_landmarker_v2_with_blendshapes.task",
            "models/face_landmarker.task",
            "face_landmarker_v2_with_blendshapes.task",
            "face_landmarker.task"
    };

    /**
     * 从 assets 中查找可用的 FaceLandmarker 模型文件路径。
     *
     * @param am AssetManager
     * @return 找到的模型路径，未找到时返回 null
     */
    static String findModelPath(AssetManager am) {
        for (String path : MODEL_CANDIDATE_PATHS) {
            try {
                am.open(path).close();
                return path;
            } catch (Exception ignored) {
            }
        }
        return null;
    }

    /**
     * 将 assets 文件读入 Direct ByteBuffer（MediaPipe 要求 Direct Buffer）。
     *
     * @param am   AssetManager
     * @param path assets 中的文件路径
     * @return Direct ByteBuffer
     */
    static ByteBuffer readAssetToDirectBuffer(AssetManager am, String path) throws Exception {
        try (InputStream is = am.open(path)) {
            int capacity = 1 << 20;
            ByteBuffer buf = ByteBuffer.allocateDirect(capacity);
            byte[] tmp = new byte[8192];
            while (true) {
                int read = is.read(tmp);
                if (read == -1) break;
                if (buf.remaining() < read) {
                    int newCap = Math.max(buf.capacity() * 2, buf.capacity() + read);
                    ByteBuffer newBuf = ByteBuffer.allocateDirect(newCap);
                    buf.flip();
                    newBuf.put(buf);
                    buf = newBuf;
                }
                buf.put(tmp, 0, read);
            }
            buf.flip();
            return buf;
        }
    }
}
